# codebase-mcp

Vector search MCP server for codebases. Index repos locally with AST-aware chunking; let Claude (or any MCP client) search them via semantic similarity.

## How it works

1. **Index** — walks repo files, chunks them using tree-sitter AST (function/method boundaries) with line-based fallback, embeds via OpenAI-compatible API, stores in in-process Qdrant.
2. **Serve** — exposes two MCP tools (`search_codebase`, `list_indexed_repos`) over stdio.
3. **Search** — embeds the query, retrieves top-8 chunks across all indexed repos (or a specific one), returns ranked results with file path and line numbers.

## Install

```bash
uv tool install /path/to/codebase-mcp
```

Or for development:

```bash
uv sync
```

## CLI

```bash
# Index a repo (fails if already indexed)
codebase-mcp index ~/Code/myproject

# Re-index after changes (replaces existing index)
codebase-mcp reindex ~/Code/myproject

# List indexed repos with chunk counts
codebase-mcp list

# Remove a repo from the index
codebase-mcp remove ~/Code/myproject

# Start MCP server (stdio, used by Claude Code)
codebase-mcp serve
```

### Config commands

```bash
# Show current settings
codebase-mcp config list

# Set embedding model (known models auto-resolve vector size)
codebase-mcp config set embedding-model text-embedding-3-large
codebase-mcp config set embedding-model my-custom-model --vector-size 768

# Set API credentials
codebase-mcp config set api-key sk-...
codebase-mcp config set api-base https://my-provider.com/v1

# Revert a setting to default / env var fallback
codebase-mcp config unset embedding-model
codebase-mcp config unset api-key
codebase-mcp config unset api-base
```

**Known models** (vector size auto-detected):

| Model | Vector size |
|---|---|
| `text-embedding-3-small` | 1536 |
| `text-embedding-3-large` | 3072 |
| `text-embedding-ada-002` | 1536 |

Default: `text-embedding-3-small`.

## Claude Code config

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "codebase-search": {
      "command": "codebase-mcp",
      "args": ["serve"],
      "env": { "OPENAI_API_KEY": "sk-..." }
    }
  }
}
```

API key can also be set via `codebase-mcp config set api-key sk-...` (persisted in `~/.codebase-mcp/settings.json`), which takes precedence over the env var.

## MCP tools

### `search_codebase`

Search indexed repos for relevant code and docs.

| Parameter | Type | Description |
|---|---|---|
| `query` | string | Natural language description of what to find |
| `repo_path` | string (optional) | Absolute path to a specific repo; omit to search all |

Returns top-8 results ranked by similarity, each with file path, line range, score, and code block.

### `list_indexed_repos`

List all indexed repos with chunk count and last indexed timestamp. No parameters.

## Supported languages (AST chunking)

| Language | Extensions | Chunk boundary |
|---|---|---|
| Python | `.py` | `function_definition`, `decorated_definition` |
| TypeScript | `.ts` | `function_declaration`, `method_definition`, `arrow_function` |
| TSX | `.tsx` | same as TypeScript |
| JavaScript | `.js`, `.jsx` | `function_declaration`, `method_definition`, `arrow_function` |
| Go | `.go` | `function_declaration`, `method_declaration` |
| Rust | `.rs` | `function_item` |
| Java | `.java` | `method_declaration`, `constructor_declaration` |
| HCL/Terraform | `.tf` | `block` |

Files without AST support (`.md`, `.yaml`, `.toml`, `.json`, `.rb`, `.cpp`, `.c`, `.h`) fall back to 100-line sliding window with 20-line overlap.

## Data storage

All data lives in `~/.codebase-mcp/`:

```
~/.codebase-mcp/
  config.json      # indexed repo metadata (paths, repo_ids, chunk counts, timestamps)
  settings.json    # embedding model, vector size, api_key, api_base
  qdrant/          # Qdrant in-process storage (one collection per repo)
```

Each repo gets a stable `repo_id` derived from its absolute path (used as Qdrant collection name). Reindexing replaces the collection in-place.

## OpenAI-compatible providers

Set `api-base` to use any OpenAI-compatible embedding API (e.g. Ollama, vLLM, Azure):

```bash
codebase-mcp config set api-base http://localhost:11434/v1
codebase-mcp config set api-key ollama
codebase-mcp config set embedding-model nomic-embed-text --vector-size 768
```

After changing the model, reindex all repos (vector dimensions must match).
