import pytest
from pathlib import Path


@pytest.fixture
def fixture_repo(tmp_path):
    """Small repo with known files for deterministic testing."""
    (tmp_path / "main.py").write_text("def hello():\n    return 'hi'\n")
    (tmp_path / "utils.py").write_text("\n".join(f"# line {i}" for i in range(150)))
    (tmp_path / "README.md").write_text("# Docs\n\nSome docs here.\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "pkg.js").write_text("should be skipped")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("should be skipped")
    (tmp_path / "binary.bin").write_bytes(b"\x00\x01\x02")
    return tmp_path


def test_iter_files_skips_hidden_dirs(fixture_repo):
    from codebase_mcp.indexer import iter_files
    files = list(iter_files(fixture_repo))
    paths = [str(f) for f in files]
    assert not any("node_modules" in p for p in paths)
    assert not any(".git" in p for p in paths)
    assert not any(".bin" in p for p in paths)


def test_iter_files_finds_source_files(fixture_repo):
    from codebase_mcp.indexer import iter_files
    names = {f.name for f in iter_files(fixture_repo)}
    assert "main.py" in names
    assert "utils.py" in names
    assert "README.md" in names


def test_chunk_file_short_file_single_chunk():
    from codebase_mcp.indexer import chunk_file
    content = "\n".join(f"line {i}" for i in range(10))
    chunks = chunk_file(content, "short.py", "/repo")
    assert len(chunks) == 1
    assert chunks[0]["start_line"] == 1
    assert chunks[0]["file"] == "short.py"
    assert chunks[0]["repo_path"] == "/repo"


def test_chunk_file_long_file_multiple_chunks():
    from codebase_mcp.indexer import chunk_file
    content = "\n".join(f"line {i}" for i in range(200))
    chunks = chunk_file(content, "long.py", "/repo")
    assert len(chunks) > 1
    # chunks overlap: second chunk starts before first ends
    assert chunks[1]["start_line"] < chunks[0]["end_line"]


def test_chunk_file_overlap():
    from codebase_mcp.indexer import chunk_file, CHUNK_LINES, OVERLAP_LINES
    content = "\n".join(f"line {i}" for i in range(CHUNK_LINES * 2))
    chunks = chunk_file(content, "f.py", "/r")
    step = CHUNK_LINES - OVERLAP_LINES
    assert chunks[1]["start_line"] == step + 1
