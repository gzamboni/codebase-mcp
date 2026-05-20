# Codebase Search MCP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI tool + MCP server that indexes repos via vector embeddings and lets Claude search codebases via FastMCP tools.

**Architecture:** User runs `codebase-mcp index /path` to embed repo chunks into a local Qdrant DB. The MCP server (`codebase-mcp serve`) exposes `search_codebase` and `list_indexed_repos` tools — it only reads from the index, never writes. OpenAI `text-embedding-3-small` handles all embeddings.

**Tech Stack:** Python 3.11+, FastMCP 2.x, qdrant-client, openai SDK, click, rich, pytest

---

## File Map

| File | Responsibility |
|---|---|
| `pyproject.toml` | Project metadata, deps, entry point |
| `src/codebase_mcp/__init__.py` | Empty package marker |
| `src/codebase_mcp/store.py` | Qdrant client factory + config.json CRUD |
| `src/codebase_mcp/indexer.py` | File walking, chunking, embedding, upsert |
| `src/codebase_mcp/searcher.py` | Query embedding + vector search + formatting |
| `src/codebase_mcp/server.py` | FastMCP server with two tools |
| `src/codebase_mcp/cli.py` | click CLI (`index`, `reindex`, `list`, `remove`, `serve`) |
| `tests/__init__.py` | Empty package marker |
| `tests/conftest.py` | Shared fixtures (tmp data dir, fixture repo) |
| `tests/test_store.py` | Unit tests for store.py |
| `tests/test_indexer.py` | Unit tests for chunking + indexing (mocked OpenAI) |
| `tests/test_searcher.py` | Unit tests for search (mocked OpenAI) |
| `tests/test_integration.py` | End-to-end: index → search → reindex → remove |

---

## Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `src/codebase_mcp/__init__.py`
- Create: `tests/__init__.py`
- Create: `.gitignore`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "codebase-mcp"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "fastmcp>=2.0",
    "qdrant-client>=1.9",
    "openai>=1.0",
    "click>=8.0",
    "rich>=13.0",
    "tiktoken>=0.7",
]

[project.scripts]
codebase-mcp = "codebase_mcp.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-mock>=3.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create package files**

```bash
mkdir -p src/codebase_mcp tests
touch src/codebase_mcp/__init__.py tests/__init__.py
```

- [ ] **Step 3: Create .gitignore**

```
__pycache__/
*.pyc
.venv/
dist/
*.egg-info/
.pytest_cache/
```

- [ ] **Step 4: Install deps and verify**

```bash
uv sync --dev
uv run python -c "import fastmcp, qdrant_client, openai, click, rich; print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/ tests/ .gitignore
git commit -m "chore: project scaffold"
```

---

## Task 2: Store Layer

**Files:**
- Create: `src/codebase_mcp/store.py`
- Create: `tests/test_store.py`

### Key design decision

`store.py` reads the data directory from the `CODEBASE_MCP_DATA_DIR` env var (falling back to `~/.codebase-mcp`). Tests set this env var to `tmp_path` so they never touch the real store.

- [ ] **Step 1: Write failing tests**

Create `tests/test_store.py`:

```python
import json
import os
import pytest
from pathlib import Path


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEBASE_MCP_DATA_DIR", str(tmp_path))
    # Re-import after env var is set so module picks up new path
    import importlib
    import codebase_mcp.store as store_mod
    importlib.reload(store_mod)
    return tmp_path


def test_get_repo_id_is_stable():
    from codebase_mcp.store import get_repo_id
    assert get_repo_id("/some/path") == get_repo_id("/some/path")
    assert get_repo_id("/some/path") != get_repo_id("/other/path")


def test_config_roundtrip(tmp_path):
    from codebase_mcp.store import add_repo, load_config, is_indexed
    path = str(tmp_path / "myrepo")
    assert not is_indexed(path)
    add_repo(path, chunk_count=42)
    assert is_indexed(path)
    config = load_config()
    assert path in config
    assert config[path]["chunk_count"] == 42


def test_remove_repo(tmp_path):
    from codebase_mcp.store import add_repo, remove_repo, is_indexed
    path = str(tmp_path / "myrepo")
    add_repo(path, chunk_count=10)
    assert is_indexed(path)
    remove_repo(path)
    assert not is_indexed(path)


def test_get_all_repos(tmp_path):
    from codebase_mcp.store import add_repo, get_all_repos
    p1 = str(tmp_path / "repo1")
    p2 = str(tmp_path / "repo2")
    add_repo(p1, chunk_count=5)
    add_repo(p2, chunk_count=15)
    repos = get_all_repos()
    assert p1 in repos
    assert p2 in repos
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_store.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` (store.py doesn't exist yet)

- [ ] **Step 3: Implement store.py**

Create `src/codebase_mcp/store.py`:

```python
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

VECTOR_SIZE = 1536  # text-embedding-3-small


def _data_dir() -> Path:
    return Path(os.environ.get("CODEBASE_MCP_DATA_DIR", str(Path.home() / ".codebase-mcp")))


def _config_path() -> Path:
    return _data_dir() / "config.json"


def _qdrant_path() -> Path:
    return _data_dir() / "qdrant"


def get_repo_id(repo_path: str) -> str:
    return hashlib.md5(repo_path.encode()).hexdigest()[:16]


def get_client() -> QdrantClient:
    _qdrant_path().mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(_qdrant_path()))


def load_config() -> dict:
    path = _config_path()
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_config(config: dict) -> None:
    _data_dir().mkdir(parents=True, exist_ok=True)
    _config_path().write_text(json.dumps(config, indent=2))


def is_indexed(repo_path: str) -> bool:
    return repo_path in load_config()


def add_repo(repo_path: str, chunk_count: int) -> None:
    config = load_config()
    config[repo_path] = {
        "repo_id": get_repo_id(repo_path),
        "last_indexed": datetime.now(timezone.utc).isoformat(),
        "chunk_count": chunk_count,
    }
    save_config(config)


def remove_repo(repo_path: str) -> None:
    config = load_config()
    config.pop(repo_path, None)
    save_config(config)


def get_all_repos() -> dict:
    return load_config()


def ensure_collection(client: QdrantClient, repo_id: str) -> None:
    try:
        client.delete_collection(collection_name=repo_id)
    except Exception:
        pass
    client.create_collection(
        collection_name=repo_id,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_store.py -v
```

Expected: all 4 tests pass

- [ ] **Step 5: Commit**

```bash
git add src/codebase_mcp/store.py tests/test_store.py
git commit -m "feat: store layer (Qdrant wrapper + config.json)"
```

---

## Task 3: File Walking + Chunking

**Files:**
- Create: `src/codebase_mcp/indexer.py` (chunking only — no OpenAI yet)
- Create: `tests/test_indexer.py` (chunking tests)

- [ ] **Step 1: Write failing tests**

Create `tests/test_indexer.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_indexer.py -v
```

Expected: `ImportError` (indexer.py doesn't exist)

- [ ] **Step 3: Implement file walking + chunking in indexer.py**

Create `src/codebase_mcp/indexer.py`:

```python
import os
import time
from pathlib import Path

from openai import OpenAI
from qdrant_client.models import PointStruct

from .store import (
    VECTOR_SIZE,
    add_repo,
    ensure_collection,
    get_client,
    get_repo_id,
    is_indexed,
    load_config,
)

INDEXED_EXTENSIONS = {
    ".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".rb",
    ".java", ".cpp", ".c", ".h", ".md", ".yaml", ".yml", ".toml", ".json",
}
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", "dist", "build",
    ".venv", "venv", ".mypy_cache", ".pytest_cache", ".ruff_cache",
}
CHUNK_LINES = 100
OVERLAP_LINES = 20
MIN_LINES_FOR_SPLIT = 20
BATCH_SIZE = 100


def iter_files(repo_path: Path):
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for filename in files:
            filepath = Path(root) / filename
            if filepath.suffix in INDEXED_EXTENSIONS:
                yield filepath


def chunk_file(content: str, filepath: str, repo_path: str) -> list[dict]:
    lines = content.splitlines()
    if len(lines) < MIN_LINES_FOR_SPLIT:
        return [{
            "text": content,
            "file": filepath,
            "start_line": 1,
            "end_line": len(lines),
            "repo_path": repo_path,
        }]

    chunks = []
    step = CHUNK_LINES - OVERLAP_LINES
    for i in range(0, len(lines), step):
        chunk_lines = lines[i:i + CHUNK_LINES]
        if not any(line.strip() for line in chunk_lines):
            continue
        chunks.append({
            "text": "\n".join(chunk_lines),
            "file": filepath,
            "start_line": i + 1,
            "end_line": i + len(chunk_lines),
            "repo_path": repo_path,
        })
    return chunks


def _embed_batch(texts: list[str], client: OpenAI) -> list[list[float]]:
    for attempt in range(3):
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=texts,
            )
            return [r.embedding for r in response.data]
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("Embedding failed after 3 attempts")


def index_repo(repo_path: str) -> int:
    """Index a repo. Always replaces any existing index for this path."""
    abs_path = str(Path(repo_path).resolve())
    repo_id = get_repo_id(abs_path)
    openai_client = OpenAI()
    qdrant = get_client()

    all_chunks: list[dict] = []
    for filepath in iter_files(Path(abs_path)):
        try:
            content = filepath.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        rel_path = str(filepath.relative_to(abs_path))
        all_chunks.extend(chunk_file(content, rel_path, abs_path))

    ensure_collection(qdrant, repo_id)

    point_id = 0
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i:i + BATCH_SIZE]
        embeddings = _embed_batch([c["text"] for c in batch], openai_client)
        points = [
            PointStruct(id=point_id + j, vector=emb, payload=chunk)
            for j, (chunk, emb) in enumerate(zip(batch, embeddings))
        ]
        qdrant.upsert(collection_name=repo_id, points=points)
        point_id += len(batch)

    add_repo(abs_path, len(all_chunks))
    return len(all_chunks)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_indexer.py -v
```

Expected: all 6 tests pass

- [ ] **Step 5: Commit**

```bash
git add src/codebase_mcp/indexer.py tests/test_indexer.py
git commit -m "feat: file walking and chunking"
```

---

## Task 4: Embedding + Indexing (mocked OpenAI)

**Files:**
- Modify: `tests/test_indexer.py` — add embedding + index_repo tests
- No changes to indexer.py (already complete)

- [ ] **Step 1: Add index_repo tests with mocked OpenAI**

Append to `tests/test_indexer.py`:

```python
import os
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEBASE_MCP_DATA_DIR", str(tmp_path))
    import importlib
    import codebase_mcp.store as store_mod
    importlib.reload(store_mod)


def _fake_embedding(size: int = 1536) -> list[float]:
    return [0.1] * size


def _mock_openai(num_texts: int):
    mock = MagicMock()
    mock.embeddings.create.return_value.data = [
        MagicMock(embedding=_fake_embedding()) for _ in range(num_texts)
    ]
    return mock


def test_index_repo_returns_chunk_count(fixture_repo, tmp_path):
    from codebase_mcp.indexer import index_repo, BATCH_SIZE

    # Count expected chunks
    from codebase_mcp.indexer import iter_files, chunk_file
    chunks = []
    for f in iter_files(fixture_repo):
        content = f.read_text(encoding="utf-8", errors="ignore")
        chunks.extend(chunk_file(content, str(f), str(fixture_repo)))
    expected = len(chunks)

    with patch("codebase_mcp.indexer.OpenAI") as MockOpenAI:
        MockOpenAI.return_value = _mock_openai(BATCH_SIZE)
        count = index_repo(str(fixture_repo))

    assert count == expected


def test_index_repo_saves_to_config(fixture_repo, tmp_path):
    from codebase_mcp.indexer import index_repo
    from codebase_mcp.store import is_indexed

    with patch("codebase_mcp.indexer.OpenAI") as MockOpenAI:
        MockOpenAI.return_value = _mock_openai(100)
        index_repo(str(fixture_repo))

    assert is_indexed(str(fixture_repo.resolve()))


def test_index_repo_replaces_existing(fixture_repo, tmp_path):
    from codebase_mcp.indexer import index_repo
    from codebase_mcp.store import load_config

    with patch("codebase_mcp.indexer.OpenAI") as MockOpenAI:
        MockOpenAI.return_value = _mock_openai(100)
        first_count = index_repo(str(fixture_repo))
        second_count = index_repo(str(fixture_repo))

    assert first_count == second_count
    config = load_config()
    assert str(fixture_repo.resolve()) in config
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
uv run pytest tests/test_indexer.py -v
```

Expected: all tests pass (9 total)

- [ ] **Step 3: Commit**

```bash
git add tests/test_indexer.py
git commit -m "test: index_repo with mocked OpenAI"
```

---

## Task 5: Searcher

**Files:**
- Create: `src/codebase_mcp/searcher.py`
- Create: `tests/test_searcher.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_searcher.py`:

```python
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEBASE_MCP_DATA_DIR", str(tmp_path))
    import importlib
    import codebase_mcp.store as store_mod
    importlib.reload(store_mod)


def _fake_embedding(size: int = 1536) -> list[float]:
    return [0.1] * size


def _mock_openai():
    mock = MagicMock()
    mock.embeddings.create.return_value.data = [MagicMock(embedding=_fake_embedding())]
    return mock


def _seeded_store(tmp_path, repo_path: str):
    """Put a fake repo entry in config + a real Qdrant collection with one point."""
    from codebase_mcp.store import add_repo, get_client, ensure_collection, get_repo_id
    from qdrant_client.models import PointStruct

    abs_path = str(Path(repo_path).resolve())
    repo_id = get_repo_id(abs_path)
    add_repo(abs_path, chunk_count=1)
    client = get_client()
    ensure_collection(client, repo_id)
    client.upsert(
        collection_name=repo_id,
        points=[PointStruct(
            id=0,
            vector=_fake_embedding(),
            payload={"file": "main.py", "start_line": 1, "end_line": 5, "repo_path": abs_path},
        )],
    )
    return abs_path


def test_search_returns_results(tmp_path):
    abs_path = _seeded_store(tmp_path, str(tmp_path / "myrepo"))

    with patch("codebase_mcp.searcher.OpenAI") as MockOpenAI:
        MockOpenAI.return_value = _mock_openai()
        from codebase_mcp.searcher import search
        result = search("something", repo_path=abs_path)

    assert "main.py" in result
    assert "lines 1-5" in result


def test_search_all_repos_when_no_path(tmp_path):
    abs_path = _seeded_store(tmp_path, str(tmp_path / "myrepo"))

    with patch("codebase_mcp.searcher.OpenAI") as MockOpenAI:
        MockOpenAI.return_value = _mock_openai()
        from codebase_mcp.searcher import search
        result = search("something")

    assert "main.py" in result


def test_search_not_indexed_repo(tmp_path):
    with patch("codebase_mcp.searcher.OpenAI") as MockOpenAI:
        MockOpenAI.return_value = _mock_openai()
        from codebase_mcp.searcher import search
        result = search("something", repo_path=str(tmp_path / "nonexistent"))

    assert "not indexed" in result.lower()


def test_search_no_repos_indexed(tmp_path):
    with patch("codebase_mcp.searcher.OpenAI") as MockOpenAI:
        MockOpenAI.return_value = _mock_openai()
        from codebase_mcp.searcher import search
        result = search("something")

    assert "no repos" in result.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_searcher.py -v
```

Expected: `ImportError` (searcher.py doesn't exist)

- [ ] **Step 3: Implement searcher.py**

Create `src/codebase_mcp/searcher.py`:

```python
from pathlib import Path

from openai import OpenAI

from .store import get_all_repos, get_client, get_repo_id, load_config

TOP_K = 8


def search(query: str, repo_path: str | None = None) -> str:
    openai_client = OpenAI()
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=[query],
    )
    query_vector = response.data[0].embedding

    config = load_config()

    if repo_path:
        abs_path = str(Path(repo_path).resolve())
        if abs_path not in config:
            return f"Repo not indexed. Run: codebase-mcp index {repo_path}"
        repo_ids = [config[abs_path]["repo_id"]]
    else:
        if not config:
            return "No repos indexed. Run: codebase-mcp index /path/to/repo"
        repo_ids = [v["repo_id"] for v in config.values()]

    qdrant = get_client()
    all_results = []
    for repo_id in repo_ids:
        try:
            results = qdrant.search(
                collection_name=repo_id,
                query_vector=query_vector,
                limit=TOP_K,
            )
            all_results.extend(results)
        except Exception:
            continue

    if not all_results:
        return "No results found."

    all_results.sort(key=lambda r: r.score, reverse=True)
    top = all_results[:TOP_K]

    parts = []
    for r in top:
        p = r.payload
        parts.append(
            f"### {p['file']} (lines {p['start_line']}-{p['end_line']}) — score: {r.score:.3f}\n"
            f"```\n{p['text']}\n```"
        )
    return "\n\n".join(parts)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_searcher.py -v
```

Expected: all 4 tests pass

- [ ] **Step 5: Commit**

```bash
git add src/codebase_mcp/searcher.py tests/test_searcher.py
git commit -m "feat: searcher (query embed + vector search + formatting)"
```

---

## Task 6: MCP Server

**Files:**
- Create: `src/codebase_mcp/server.py`

No separate test file — the server functions delegate entirely to `searcher.py` and `store.py` which are already tested. Smoke test included.

- [ ] **Step 1: Implement server.py**

Create `src/codebase_mcp/server.py`:

```python
from fastmcp import FastMCP

from . import searcher
from .store import get_all_repos

mcp = FastMCP("codebase-search")


@mcp.tool()
def search_codebase(query: str, repo_path: str | None = None) -> str:
    """Search indexed codebase for relevant code and docs.

    Args:
        query: Natural language description of what to find.
        repo_path: Absolute path to a specific repo. Omit to search all indexed repos.
    """
    return searcher.search(query, repo_path)


@mcp.tool()
def list_indexed_repos() -> str:
    """List all indexed repositories with chunk count and last indexed time."""
    repos = get_all_repos()
    if not repos:
        return "No repos indexed. Run: codebase-mcp index /path/to/repo"
    lines = [
        f"- {path}  ({meta['chunk_count']} chunks, indexed {meta['last_indexed']})"
        for path, meta in repos.items()
    ]
    return "\n".join(lines)
```

- [ ] **Step 2: Smoke test**

```bash
uv run python -c "from codebase_mcp.server import mcp; print('MCP tools:', [t.name for t in mcp._tools.values()])"
```

Expected output contains: `search_codebase` and `list_indexed_repos`

- [ ] **Step 3: Commit**

```bash
git add src/codebase_mcp/server.py
git commit -m "feat: MCP server with search_codebase and list_indexed_repos tools"
```

---

## Task 7: CLI

**Files:**
- Create: `src/codebase_mcp/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_cli.py`:

```python
import pytest
from click.testing import CliRunner
from unittest.mock import patch
from pathlib import Path


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEBASE_MCP_DATA_DIR", str(tmp_path))
    import importlib
    import codebase_mcp.store as store_mod
    importlib.reload(store_mod)


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def sample_repo(tmp_path):
    repo = tmp_path / "myrepo"
    repo.mkdir()
    (repo / "main.py").write_text("def hello(): pass\n")
    return repo


def test_index_command(runner, sample_repo):
    from codebase_mcp.cli import main

    with patch("codebase_mcp.indexer.OpenAI") as MockOpenAI:
        mock = MockOpenAI.return_value
        mock.embeddings.create.return_value.data = []
        result = runner.invoke(main, ["index", str(sample_repo)])

    assert result.exit_code == 0
    assert "Indexed" in result.output


def test_index_twice_fails(runner, sample_repo):
    from codebase_mcp.cli import main

    with patch("codebase_mcp.indexer.OpenAI") as MockOpenAI:
        mock = MockOpenAI.return_value
        mock.embeddings.create.return_value.data = []
        runner.invoke(main, ["index", str(sample_repo)])
        result = runner.invoke(main, ["index", str(sample_repo)])

    assert result.exit_code != 0
    assert "reindex" in result.output.lower()


def test_reindex_command(runner, sample_repo):
    from codebase_mcp.cli import main

    with patch("codebase_mcp.indexer.OpenAI") as MockOpenAI:
        mock = MockOpenAI.return_value
        mock.embeddings.create.return_value.data = []
        runner.invoke(main, ["index", str(sample_repo)])
        result = runner.invoke(main, ["reindex", str(sample_repo)])

    assert result.exit_code == 0
    assert "Re-indexed" in result.output


def test_list_command_empty(runner):
    from codebase_mcp.cli import main
    result = runner.invoke(main, ["list"])
    assert result.exit_code == 0
    assert "No repos" in result.output


def test_list_command_shows_repo(runner, sample_repo):
    from codebase_mcp.cli import main

    with patch("codebase_mcp.indexer.OpenAI") as MockOpenAI:
        mock = MockOpenAI.return_value
        mock.embeddings.create.return_value.data = []
        runner.invoke(main, ["index", str(sample_repo)])
        result = runner.invoke(main, ["list"])

    assert result.exit_code == 0
    assert str(sample_repo.resolve()) in result.output


def test_remove_command(runner, sample_repo):
    from codebase_mcp.cli import main

    with patch("codebase_mcp.indexer.OpenAI") as MockOpenAI:
        mock = MockOpenAI.return_value
        mock.embeddings.create.return_value.data = []
        runner.invoke(main, ["index", str(sample_repo)])
        result = runner.invoke(main, ["remove", str(sample_repo)])

    assert result.exit_code == 0
    assert "Removed" in result.output


def test_remove_nonexistent_fails(runner, tmp_path):
    from codebase_mcp.cli import main
    result = runner.invoke(main, ["remove", str(tmp_path / "ghost")])
    assert result.exit_code != 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_cli.py -v
```

Expected: `ImportError` (cli.py doesn't exist)

- [ ] **Step 3: Implement cli.py**

Create `src/codebase_mcp/cli.py`:

```python
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from . import indexer
from .store import get_all_repos, get_client, get_repo_id, is_indexed, load_config, remove_repo

console = Console()


@click.group()
def main():
    """Codebase vector search — index repos, search via MCP."""


@main.command()
@click.argument("path", type=click.Path(exists=True, file_okay=False))
def index(path: str) -> None:
    """Index a repo. Fails if already indexed — use reindex to update."""
    abs_path = str(Path(path).resolve())
    if is_indexed(abs_path):
        console.print("[red]Already indexed. Use `reindex` to update.[/red]")
        raise SystemExit(1)
    with console.status(f"Indexing {abs_path}..."):
        count = indexer.index_repo(abs_path)
    console.print(f"[green]Indexed {count} chunks from {abs_path}[/green]")


@main.command()
@click.argument("path", type=click.Path(exists=True, file_okay=False))
def reindex(path: str) -> None:
    """Re-index a repo, replacing any existing index."""
    abs_path = str(Path(path).resolve())
    with console.status(f"Re-indexing {abs_path}..."):
        count = indexer.index_repo(abs_path)
    console.print(f"[green]Re-indexed {count} chunks from {abs_path}[/green]")


@main.command("list")
def list_repos() -> None:
    """List all indexed repos with stats."""
    repos = get_all_repos()
    if not repos:
        console.print("No repos indexed.")
        return
    table = Table("Path", "Chunks", "Last Indexed")
    for path, meta in repos.items():
        table.add_row(path, str(meta["chunk_count"]), meta["last_indexed"])
    console.print(table)


@main.command()
@click.argument("path")
def remove(path: str) -> None:
    """Remove a repo from the index."""
    abs_path = str(Path(path).resolve())
    config = load_config()
    if abs_path not in config:
        console.print(f"[red]Not indexed: {abs_path}[/red]")
        raise SystemExit(1)
    repo_id = config[abs_path]["repo_id"]
    try:
        get_client().delete_collection(repo_id)
    except Exception:
        pass
    remove_repo(abs_path)
    console.print(f"[green]Removed {abs_path}[/green]")


@main.command()
def serve() -> None:
    """Start the MCP server (used by Claude Code)."""
    from .server import mcp
    mcp.run(transport="stdio")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_cli.py -v
```

Expected: all 7 tests pass

- [ ] **Step 5: Commit**

```bash
git add src/codebase_mcp/cli.py tests/test_cli.py
git commit -m "feat: CLI (index, reindex, list, remove, serve)"
```

---

## Task 8: Integration Tests + conftest

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/test_integration.py`

- [ ] **Step 1: Create shared conftest.py**

Create `tests/conftest.py`:

```python
import pytest
from pathlib import Path


@pytest.fixture
def fixture_repo(tmp_path) -> Path:
    """Minimal repo with known content for integration tests."""
    repo = tmp_path / "fixture_repo"
    repo.mkdir()
    (repo / "auth.py").write_text(
        "def authenticate(token: str) -> bool:\n"
        "    # Verify JWT token\n"
        "    return token.startswith('Bearer ')\n"
    )
    (repo / "utils.py").write_text(
        "def format_date(dt) -> str:\n"
        "    return dt.strftime('%Y-%m-%d')\n"
    )
    (repo / "README.md").write_text("# Fixture Repo\n\nUsed for testing.\n")
    return repo
```

- [ ] **Step 2: Write integration tests**

Create `tests/test_integration.py`:

```python
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEBASE_MCP_DATA_DIR", str(tmp_path))
    import importlib
    import codebase_mcp.store as store_mod
    importlib.reload(store_mod)


def _fake_embedding(size: int = 1536) -> list[float]:
    return [0.1] * size


def _mock_openai_for_index(chunk_count: int):
    mock = MagicMock()
    mock.embeddings.create.return_value.data = [
        MagicMock(embedding=_fake_embedding()) for _ in range(chunk_count)
    ]
    return mock


def _mock_openai_for_search():
    mock = MagicMock()
    mock.embeddings.create.return_value.data = [MagicMock(embedding=_fake_embedding())]
    return mock


def test_index_and_search(fixture_repo, tmp_path):
    """Full flow: index a repo, search it, get non-empty results."""
    from codebase_mcp.indexer import index_repo
    from codebase_mcp.searcher import search

    with patch("codebase_mcp.indexer.OpenAI") as MockIdx:
        MockIdx.return_value = _mock_openai_for_index(100)
        count = index_repo(str(fixture_repo))

    assert count > 0

    with patch("codebase_mcp.searcher.OpenAI") as MockSearch:
        MockSearch.return_value = _mock_openai_for_search()
        result = search("authentication token")

    assert "auth.py" in result
    assert "lines" in result


def test_reindex_clears_old(fixture_repo, tmp_path):
    """Reindex replaces old chunks; chunk count stays consistent."""
    from codebase_mcp.indexer import index_repo
    from codebase_mcp.store import load_config

    with patch("codebase_mcp.indexer.OpenAI") as MockIdx:
        MockIdx.return_value = _mock_openai_for_index(100)
        first_count = index_repo(str(fixture_repo))
        second_count = index_repo(str(fixture_repo))

    assert first_count == second_count
    config = load_config()
    abs_path = str(fixture_repo.resolve())
    assert config[abs_path]["chunk_count"] == second_count


def test_search_no_index(tmp_path):
    """Searching an unindexed repo returns a helpful error message."""
    from codebase_mcp.searcher import search

    with patch("codebase_mcp.searcher.OpenAI") as MockSearch:
        MockSearch.return_value = _mock_openai_for_search()
        result = search("anything", repo_path=str(tmp_path / "nonexistent"))

    assert "not indexed" in result.lower()
    assert "codebase-mcp index" in result
```

- [ ] **Step 3: Run all tests**

```bash
uv run pytest -v
```

Expected: all tests pass (24+ total across all test files)

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py tests/test_integration.py
git commit -m "test: integration tests for index, search, reindex, and error cases"
```

---

## Task 9: Install + Claude Config

**Files:**
- Create: `README.md`

- [ ] **Step 1: Install as uv tool**

```bash
uv tool install /Users/gzamboni/Code/ai/codebase-mcp
```

- [ ] **Step 2: Verify CLI works**

```bash
codebase-mcp --help
```

Expected: shows `index`, `reindex`, `list`, `remove`, `serve` commands

- [ ] **Step 3: Add to Claude Code MCP config**

Add to `~/.claude/settings.json` under `mcpServers`:

```json
"codebase": {
  "command": "codebase-mcp",
  "args": ["serve"],
  "env": {
    "OPENAI_API_KEY": "sk-..."
  }
}
```

- [ ] **Step 4: Test MCP server starts**

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | OPENAI_API_KEY=test codebase-mcp serve
```

Expected: JSON response listing `search_codebase` and `list_indexed_repos`

- [ ] **Step 5: Create README.md**

```markdown
# codebase-mcp

Vector search MCP server for codebases. Index repos locally, let Claude search them.

## Install

```bash
uv tool install /path/to/codebase-mcp
```

## Usage

```bash
# Index a repo
codebase-mcp index ~/Code/myproject

# Re-index after changes
codebase-mcp reindex ~/Code/myproject

# List indexed repos
codebase-mcp list

# Remove from index
codebase-mcp remove ~/Code/myproject
```

## Claude Code config

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "codebase": {
      "command": "codebase-mcp",
      "args": ["serve"],
      "env": { "OPENAI_API_KEY": "sk-..." }
    }
  }
}
```

## Data

Index stored at `~/.codebase-mcp/` (Qdrant in-process + config.json).
```

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "docs: README with install and usage instructions"
```

---

## Self-Review

**Spec coverage:**
- ✅ CLI commands: `index`, `reindex`, `list`, `remove`, `serve`
- ✅ MCP tools: `search_codebase`, `list_indexed_repos`
- ✅ Qdrant in-process at `~/.codebase-mcp/qdrant/`
- ✅ OpenAI `text-embedding-3-small`
- ✅ Sliding window chunking 100L / 20L overlap
- ✅ Skip dirs: `.git`, `node_modules`, `__pycache__`, etc.
- ✅ Error: already indexed → fail, use reindex
- ✅ Error: binary/undecodable → skip silently
- ✅ Error: rate limit → retry 3x exponential backoff
- ✅ Error: repo not indexed in search → helpful message
- ✅ `repo_path` absent in search → search all repos
- ✅ 3 integration tests from spec
- ✅ uv tool install + Claude config
- ✅ Tests use `CODEBASE_MCP_DATA_DIR` env var to avoid touching `~/.codebase-mcp`

**Type consistency check:**
- `get_repo_id(repo_path: str) -> str` — used consistently in store, indexer, cli ✅
- `index_repo(repo_path: str) -> int` — returns chunk count, used in cli (index + reindex) ✅
- `search(query: str, repo_path: str | None) -> str` — matches server.py tool signature ✅
- `ensure_collection(client, repo_id)` — called in indexer only ✅
