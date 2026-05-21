import asyncio
from unittest.mock import patch

from codebase_mcp.server import mcp


def _get_tool(name):
    """Get tool function from FastMCP server."""
    tool = asyncio.run(mcp.get_tool(name))
    if tool is None:
        raise KeyError(f"Tool {name!r} not registered")
    return tool.fn


def test_get_file_outline_python(tmp_path):
    src = tmp_path / "foo.py"
    src.write_text("def alpha():\n    pass\n\ndef beta(x, y):\n    return x + y\n")
    fn = _get_tool("get_file_outline")
    result = fn(file_path=str(src))
    assert "alpha" in result
    assert "beta" in result
    assert "1" in result  # line number


def test_get_file_outline_unsupported_file(tmp_path):
    src = tmp_path / "data.json"
    src.write_text('{"key": "value"}')
    fn = _get_tool("get_file_outline")
    result = fn(file_path=str(src))
    assert "No AST outline" in result or "data.json" in result


def test_get_file_outline_missing_file():
    fn = _get_tool("get_file_outline")
    result = fn(file_path="/nonexistent/path/foo.py")
    assert "not found" in result.lower() or "File not found" in result


def test_search_symbols_no_repos():
    fn = _get_tool("search_symbols")
    with patch("codebase_mcp.server.get_all_repos", return_value={}):
        result = fn(name="anything")
    assert "No repos" in result


def test_search_symbols_returns_string():
    fn = _get_tool("search_symbols")
    with patch("codebase_mcp.server.get_all_repos", return_value={}):
        result = fn(name="foo")
    assert isinstance(result, str)


def test_find_todos_no_repos():
    fn = _get_tool("find_todos")
    with patch("codebase_mcp.server.get_all_repos", return_value={}):
        result = fn()
    assert "No repos" in result


def test_find_todos_finds_comment(tmp_path):
    src = tmp_path / "app.py"
    src.write_text("x = 1  # TODO: fix this\ny = 2\n# FIXME broken\n")
    fn = _get_tool("find_todos")
    with patch(
        "codebase_mcp.server.get_all_repos",
        return_value={
            str(tmp_path): {
                "repo_id": "abc",
                "last_indexed": "2024-01-01T00:00:00Z",
                "chunk_count": 1,
            }
        },
    ):
        result = fn(repo_path=str(tmp_path))
    assert "TODO" in result or "FIXME" in result
