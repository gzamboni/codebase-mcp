from unittest.mock import MagicMock, patch

import pytest

from yacodebase_mcp.store import load_file_hashes, save_file_hashes


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("YACODEBASE_MCP_DATA_DIR", str(tmp_path))
    return tmp_path


def test_save_and_load_file_hashes(isolated):
    from yacodebase_mcp.store import add_repo

    add_repo("/test/repo", 10)

    hashes = {"src/a.py": "abc123", "src/b.py": "def456"}
    save_file_hashes("/test/repo", hashes)

    loaded = load_file_hashes("/test/repo")
    assert loaded == hashes


def test_load_file_hashes_missing_repo(isolated):
    result = load_file_hashes("/nonexistent/repo")
    assert result == {}


def test_save_file_hashes_unknown_repo_is_noop(isolated):
    # Should not raise, just silently skip
    save_file_hashes("/not/indexed", {"file.py": "hash"})
    result = load_file_hashes("/not/indexed")
    assert result == {}


def test_index_incremental_skips_unchanged(isolated):
    repo = isolated / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("def foo(): pass\n")

    from yacodebase_mcp.indexer import index_repo_incremental
    from yacodebase_mcp.store import add_repo

    # Pre-register the repo so incremental can find it
    add_repo(str(repo), 0)

    mock_qdrant = MagicMock()
    mock_qdrant.collection_exists.return_value = True
    mock_qdrant.get_collection.return_value = MagicMock(
        config=MagicMock(params=MagicMock(vectors=MagicMock(size=1536)))
    )
    mock_qdrant.scroll.return_value = ([], None)

    mock_openai = MagicMock()
    mock_openai.embeddings.create.return_value = MagicMock(data=[MagicMock(embedding=[0.1] * 1536)])

    with (
        patch("yacodebase_mcp.indexer.get_client", return_value=mock_qdrant),
        patch("yacodebase_mcp.indexer.OpenAI", return_value=mock_openai),
    ):
        count1 = index_repo_incremental(str(repo))
        count2 = index_repo_incremental(str(repo))  # second call: nothing changed

    assert count1 > 0  # first run indexed something
    assert count2 == 0  # second run: file unchanged, nothing re-embedded
