# yacodebase-mcp — dev guide

## Setup

```bash
uv sync
```

All commands use `.venv/bin/` prefix or `uv run`.

## Run tests

```bash
.venv/bin/pytest
.venv/bin/pytest -v
.venv/bin/pytest tests/test_ast_chunker.py -v   # specific module
```

## Lint / format

```bash
.venv/bin/ruff check src tests
.venv/bin/ruff format src tests
```

Line length: 100. Rules: E, F, I (pycodestyle, pyflakes, isort).

## Project structure

```
src/codebase_mcp/
  cli.py          # Click CLI: index, reindex, list, remove, serve, config *
  server.py       # FastMCP server — exposes search_codebase + list_indexed_repos
  indexer.py      # File walking, chunking, embedding, Qdrant upsert
  ast_chunker.py  # tree-sitter AST chunking (function/method boundaries)
  searcher.py     # Query embedding + Qdrant search + result formatting
  store.py        # Qdrant client, config.json r/w, repo metadata
  settings.py     # settings.json r/w (embedding model, vector size, api_key, api_base)
tests/
  test_ast_chunker.py
  test_cli.py
  test_indexer.py
  test_integration.py
  test_searcher.py
  test_settings.py
  test_store.py
```

## Key design decisions

**AST chunking → line fallback**: `indexer.chunk_file` tries `ast_chunker.chunk_file_ast` first. If the language is unsupported or tree-sitter fails, falls back to 100-line sliding window (20-line overlap). This ensures semantic boundaries for supported languages without breaking on unknown file types.

**In-process Qdrant**: No external service needed. `store.get_client()` returns a `QdrantClient` pointing at `~/.codebase-mcp/qdrant/`. One collection per repo, named by `repo_id` (hash of abs path).

**OpenAI-compatible embeddings**: `indexer` and `searcher` both instantiate `openai.OpenAI(api_key=..., base_url=...)` from settings. Any OpenAI-compatible provider works by setting `api_base`.

**Vector size mismatch detection**: `searcher.search` reads actual vector dim from Qdrant and skips repos where it doesn't match current model's `vector_size`. Emits a warning prompting reindex.

**MAX_CHUNK_CHARS = 16000**: Both `ast_chunker` and `indexer` truncate chunk text at 16k chars (~8192 tokens at 2 chars/token for dense code). Applied post-collection in `indexer.index_repo` as final safety.

**Config precedence**: `settings.json` fields override env vars. `api_key=None` in settings → falls back to `OPENAI_API_KEY` env var (handled by OpenAI SDK). `api_base=None` → uses OpenAI default.

## Adding a new language

1. Add tree-sitter grammar to `pyproject.toml` dependencies.
2. Add entry to `ast_chunker.EXT_TO_LANG` mapping extension → language name.
3. Add entry to `ast_chunker.SEMANTIC_NODES` mapping language → relevant node type set.
4. Add parser instantiation branch in `ast_chunker._get_parser`.
5. Add extension to `indexer.INDEXED_EXTENSIONS`.

## Data dir

`~/.codebase-mcp/` — created on first use by `store._data_dir()`.

- `config.json` — repo registry: `{abs_path: {repo_id, chunk_count, last_indexed}}`
- `settings.json` — persistent settings (only non-null, non-default values written)
- `qdrant/` — Qdrant on-disk storage
