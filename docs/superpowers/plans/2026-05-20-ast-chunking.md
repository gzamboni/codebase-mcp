# AST-Based Chunking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace sliding-window line chunking with tree-sitter AST-based chunking for Python, TypeScript, JavaScript, Go, Rust, Java, and Terraform, with automatic fallback to line chunking for unsupported files.

**Architecture:** New `ast_chunker.py` module exposes `chunk_file_ast()` which parses files with tree-sitter and extracts semantic nodes (functions, methods, HCL blocks). `indexer.py` is modified minimally: existing `chunk_file` is renamed to `_chunk_file_lines`, and a new `chunk_file` dispatcher tries AST first, falls back to lines. All other code is unchanged.

**Tech Stack:** Python 3.11+, tree-sitter 0.23, tree-sitter-python/typescript/javascript/go/rust/java/hcl

---

## File Map

| File | Change |
|---|---|
| `pyproject.toml` | Add 8 tree-sitter dependencies |
| `src/codebase_mcp/ast_chunker.py` | New: AST parsing for all 7 languages |
| `src/codebase_mcp/indexer.py` | Rename `chunk_file`→`_chunk_file_lines`, add dispatcher, add `.tf` to extensions |
| `tests/test_ast_chunker.py` | New: 11 unit tests |
| `tests/test_indexer.py` | Update: rename references `chunk_file`→`_chunk_file_lines`, add 1 test |

---

## Task 1: Add tree-sitter dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add dependencies**

Edit `pyproject.toml` — add to `dependencies` list:

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
    "tree-sitter>=0.23",
    "tree-sitter-python>=0.23",
    "tree-sitter-typescript>=0.23",
    "tree-sitter-javascript>=0.23",
    "tree-sitter-go>=0.23",
    "tree-sitter-rust>=0.23",
    "tree-sitter-java>=0.23",
    "tree-sitter-hcl>=0.23",
]
```

- [ ] **Step 2: Sync and verify**

```bash
uv sync --dev
uv run python -c "
import tree_sitter_python
import tree_sitter_typescript
import tree_sitter_javascript
import tree_sitter_go
import tree_sitter_rust
import tree_sitter_java
import tree_sitter_hcl
print('OK')
"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add tree-sitter dependencies"
```

---

## Task 2: Create ast_chunker.py

**Files:**
- Create: `src/codebase_mcp/ast_chunker.py`
- Create: `tests/test_ast_chunker.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_ast_chunker.py`:

```python
import pytest
from pathlib import Path

# ── Fixtures ──────────────────────────────────────────────────────────────────

PYTHON_FUNCTIONS = """\
def greet(name: str) -> str:
    return f"Hello, {name}"


def farewell(name: str) -> str:
    return f"Goodbye, {name}"
"""

PYTHON_DECORATED = """\
def decorator(fn):
    return fn


@decorator
def annotated() -> None:
    pass
"""

PYTHON_ONLY_IMPORTS = """\
import os
import sys
from pathlib import Path
"""

TYPESCRIPT_CLASS = """\
class UserService {
    getUser(id: number): string {
        return `user-${id}`;
    }

    createUser(name: string): string {
        return `created-${name}`;
    }
}
"""

GO_FUNCTIONS = """\
package math

func Add(a, b int) int {
    return a + b
}

func Subtract(a, b int) int {
    return a - b
}
"""

RUST_FUNCTIONS = """\
fn add(a: i32, b: i32) -> i32 {
    a + b
}

struct Calculator;

impl Calculator {
    fn multiply(&self, a: i32, b: i32) -> i32 {
        a * b
    }
}
"""

JAVA_CLASS = """\
public class MathUtils {
    public MathUtils() {}

    public int add(int a, int b) {
        return a + b;
    }
}
"""

TERRAFORM_RESOURCES = """\
resource "aws_s3_bucket" "main" {
  bucket = "my-bucket"
}

resource "aws_instance" "web" {
  ami           = "ami-123456"
  instance_type = "t2.micro"
}
"""

# ── Tests ──────────────────────────────────────────────────────────────────────

def test_python_extracts_functions():
    from codebase_mcp.ast_chunker import chunk_file_ast
    chunks = chunk_file_ast(PYTHON_FUNCTIONS, "foo.py", "/repo")
    assert chunks is not None
    assert len(chunks) == 2
    names = [c["text"].split("(")[0].split()[-1] for c in chunks]
    assert "greet" in names
    assert "farewell" in names


def test_python_extracts_decorated():
    from codebase_mcp.ast_chunker import chunk_file_ast
    chunks = chunk_file_ast(PYTHON_DECORATED, "foo.py", "/repo")
    assert chunks is not None
    # decorator def + decorated_definition = 2 nodes
    # only the decorated_definition wraps @decorator+def annotated
    func_chunks = [c for c in chunks if "annotated" in c["text"]]
    assert len(func_chunks) == 1
    assert func_chunks[0]["node_type"] == "decorated_definition"


def test_typescript_extracts_methods():
    from codebase_mcp.ast_chunker import chunk_file_ast
    chunks = chunk_file_ast(TYPESCRIPT_CLASS, "svc.ts", "/repo")
    assert chunks is not None
    assert len(chunks) == 2
    texts = " ".join(c["text"] for c in chunks)
    assert "getUser" in texts
    assert "createUser" in texts
    # class itself must NOT be a chunk
    assert not any(c["text"].strip().startswith("class") for c in chunks)


def test_go_extracts_functions():
    from codebase_mcp.ast_chunker import chunk_file_ast
    chunks = chunk_file_ast(GO_FUNCTIONS, "math.go", "/repo")
    assert chunks is not None
    assert len(chunks) == 2
    texts = " ".join(c["text"] for c in chunks)
    assert "Add" in texts
    assert "Subtract" in texts


def test_rust_extracts_functions():
    from codebase_mcp.ast_chunker import chunk_file_ast
    chunks = chunk_file_ast(RUST_FUNCTIONS, "lib.rs", "/repo")
    assert chunks is not None
    assert len(chunks) == 2
    texts = " ".join(c["text"] for c in chunks)
    assert "fn add" in texts
    assert "fn multiply" in texts


def test_java_extracts_methods():
    from codebase_mcp.ast_chunker import chunk_file_ast
    chunks = chunk_file_ast(JAVA_CLASS, "MathUtils.java", "/repo")
    assert chunks is not None
    assert len(chunks) == 2
    texts = " ".join(c["text"] for c in chunks)
    assert "MathUtils()" in texts   # constructor
    assert "add" in texts


def test_terraform_extracts_blocks():
    from codebase_mcp.ast_chunker import chunk_file_ast
    chunks = chunk_file_ast(TERRAFORM_RESOURCES, "main.tf", "/repo")
    assert chunks is not None
    assert len(chunks) == 2
    texts = " ".join(c["text"] for c in chunks)
    assert "aws_s3_bucket" in texts
    assert "aws_instance" in texts


def test_unsupported_ext_returns_none():
    from codebase_mcp.ast_chunker import chunk_file_ast
    result = chunk_file_ast("key: value\n", "config.yaml", "/repo")
    assert result is None


def test_no_semantic_nodes_returns_none():
    from codebase_mcp.ast_chunker import chunk_file_ast
    result = chunk_file_ast(PYTHON_ONLY_IMPORTS, "imports.py", "/repo")
    assert result is None


def test_chunk_metadata():
    from codebase_mcp.ast_chunker import chunk_file_ast
    chunks = chunk_file_ast(PYTHON_FUNCTIONS, "src/foo.py", "/myrepo")
    assert chunks is not None
    c = chunks[0]
    assert c["file"] == "src/foo.py"
    assert c["repo_path"] == "/myrepo"
    assert isinstance(c["start_line"], int)
    assert isinstance(c["end_line"], int)
    assert c["start_line"] >= 1
    assert c["end_line"] >= c["start_line"]
    assert "node_type" in c


def test_chunk_file_falls_back_to_lines():
    from codebase_mcp.indexer import chunk_file
    content = "key: value\nanother: line\n"
    chunks = chunk_file(content, "config.yaml", "/repo")
    assert len(chunks) >= 1
    assert "node_type" not in chunks[0]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_ast_chunker.py -v
```

Expected: `ImportError` — `ast_chunker.py` doesn't exist yet

- [ ] **Step 3: Implement ast_chunker.py**

Create `src/codebase_mcp/ast_chunker.py`:

```python
from __future__ import annotations

from pathlib import Path

MAX_CHUNK_CHARS = 32_000  # mirrors indexer.py constant

SEMANTIC_NODES: dict[str, set[str]] = {
    "python": {"function_definition", "decorated_definition"},
    "typescript": {"function_declaration", "method_definition", "arrow_function"},
    "typescript_tsx": {"function_declaration", "method_definition", "arrow_function"},
    "javascript": {"function_declaration", "method_definition", "arrow_function"},
    "go": {"function_declaration", "method_declaration"},
    "rust": {"function_item"},
    "java": {"method_declaration", "constructor_declaration"},
    "hcl": {"block"},
}

EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript_tsx",
    ".js": "javascript",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".tf": "hcl",
}

_parsers: dict[str, object] = {}


def _get_parser(lang_name: str):
    if lang_name in _parsers:
        return _parsers[lang_name]

    try:
        from tree_sitter import Language, Parser

        if lang_name == "python":
            import tree_sitter_python as m
            lang = Language(m.language())
        elif lang_name == "typescript":
            import tree_sitter_typescript as m
            lang = Language(m.language_typescript())
        elif lang_name == "typescript_tsx":
            import tree_sitter_typescript as m
            lang = Language(m.language_tsx())
        elif lang_name == "javascript":
            import tree_sitter_javascript as m
            lang = Language(m.language())
        elif lang_name == "go":
            import tree_sitter_go as m
            lang = Language(m.language())
        elif lang_name == "rust":
            import tree_sitter_rust as m
            lang = Language(m.language())
        elif lang_name == "java":
            import tree_sitter_java as m
            lang = Language(m.language())
        elif lang_name == "hcl":
            import tree_sitter_hcl as m
            lang = Language(m.language())
        else:
            _parsers[lang_name] = None
            return None

        parser = Parser(lang)
        _parsers[lang_name] = parser
        return parser

    except Exception:
        _parsers[lang_name] = None
        return None


def chunk_file_ast(content: str, filepath: str, repo_path: str) -> list[dict] | None:
    """Parse file with tree-sitter and extract semantic nodes as chunks.

    Returns None if the extension is unsupported, the parser is unavailable,
    or the file contains no semantic nodes (caller should fall back to line chunking).
    """
    lang_name = EXT_TO_LANG.get(Path(filepath).suffix)
    if not lang_name:
        return None

    parser = _get_parser(lang_name)
    if parser is None:
        return None

    node_types = SEMANTIC_NODES[lang_name]
    tree = parser.parse(content.encode("utf-8", errors="replace"))

    chunks: list[dict] = []

    def walk(node) -> None:
        if node.type == "ERROR":
            return
        if node.type in node_types:
            text = content[node.start_byte:node.end_byte][:MAX_CHUNK_CHARS]
            chunks.append({
                "text": text,
                "file": filepath,
                "start_line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
                "repo_path": repo_path,
                "node_type": node.type,
            })
            return  # don't descend — avoids nested method-inside-class duplication
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return chunks if chunks else None
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_ast_chunker.py -v
```

Expected: all 11 tests pass

- [ ] **Step 5: Commit**

```bash
git add src/codebase_mcp/ast_chunker.py tests/test_ast_chunker.py
git commit -m "feat: AST chunker with tree-sitter for Python/TS/JS/Go/Rust/Java/HCL"
```

---

## Task 3: Update indexer.py

**Files:**
- Modify: `src/codebase_mcp/indexer.py`

- [ ] **Step 1: Apply changes**

In `src/codebase_mcp/indexer.py`, make these exact changes:

**a) Add `.tf` to `INDEXED_EXTENSIONS`:**
```python
INDEXED_EXTENSIONS = {
    ".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".rb",
    ".java", ".cpp", ".c", ".h", ".md", ".yaml", ".yml", ".toml", ".json",
    ".tf",
}
```

**b) Add import at top of file (after existing imports):**
```python
from .ast_chunker import chunk_file_ast
```

**c) Rename existing `chunk_file` to `_chunk_file_lines`** — change the `def` signature only, no logic changes:
```python
def _chunk_file_lines(content: str, filepath: str, repo_path: str) -> list[dict]:
```

**d) Add new `chunk_file` dispatcher after `_chunk_file_lines`:**
```python
def chunk_file(content: str, filepath: str, repo_path: str) -> list[dict]:
    """Chunk file using AST if supported, falling back to sliding-window lines."""
    try:
        chunks = chunk_file_ast(content, filepath, repo_path)
    except Exception:
        chunks = None
    return chunks if chunks else _chunk_file_lines(content, filepath, repo_path)
```

The internal call in `index_repo` already calls `chunk_file(...)` — no changes needed there.

- [ ] **Step 2: Verify full test suite still passes**

```bash
uv run pytest -v
```

Expected: all 28 existing tests + 11 new = 39 tests pass

- [ ] **Step 3: Commit**

```bash
git add src/codebase_mcp/indexer.py
git commit -m "feat: chunk_file dispatcher — AST with line fallback, add .tf extension"
```

---

## Task 4: Update test_indexer.py

**Files:**
- Modify: `tests/test_indexer.py`

- [ ] **Step 1: Fix renamed references**

In `tests/test_indexer.py`, update ALL imports and calls of `chunk_file` that test line-chunking behaviour to use `_chunk_file_lines`.

Find these tests and update the import inside each one:
- `test_chunk_file_short_file_single_chunk` → import `_chunk_file_lines` instead of `chunk_file`
- `test_chunk_file_long_file_multiple_chunks` → same
- `test_chunk_file_overlap` → imports `_chunk_file_lines`, `CHUNK_LINES`, `OVERLAP_LINES`

Updated tests:

```python
def test_chunk_file_short_file_single_chunk():
    from codebase_mcp.indexer import _chunk_file_lines
    content = "\n".join(f"line {i}" for i in range(10))
    chunks = _chunk_file_lines(content, "short.py", "/repo")
    assert len(chunks) == 1
    assert chunks[0]["start_line"] == 1
    assert chunks[0]["file"] == "short.py"
    assert chunks[0]["repo_path"] == "/repo"


def test_chunk_file_long_file_multiple_chunks():
    from codebase_mcp.indexer import _chunk_file_lines
    content = "\n".join(f"line {i}" for i in range(200))
    chunks = _chunk_file_lines(content, "long.py", "/repo")
    assert len(chunks) > 1
    assert chunks[1]["start_line"] < chunks[0]["end_line"]


def test_chunk_file_overlap():
    from codebase_mcp.indexer import _chunk_file_lines, CHUNK_LINES, OVERLAP_LINES
    content = "\n".join(f"line {i}" for i in range(CHUNK_LINES * 2))
    chunks = _chunk_file_lines(content, "f.py", "/r")
    step = CHUNK_LINES - OVERLAP_LINES
    assert chunks[1]["start_line"] == step + 1
```

- [ ] **Step 2: Add AST dispatcher test**

Append to `tests/test_indexer.py`:

```python
def test_chunk_file_uses_ast_for_python():
    from codebase_mcp.indexer import chunk_file
    content = "def hello():\n    return 'hi'\n"
    chunks = chunk_file(content, "hello.py", "/repo")
    assert len(chunks) == 1
    assert "node_type" in chunks[0]
    assert chunks[0]["node_type"] == "function_definition"
```

- [ ] **Step 3: Run full test suite**

```bash
uv run pytest -v
```

Expected: all 40 tests pass

- [ ] **Step 4: Commit**

```bash
git add tests/test_indexer.py
git commit -m "test: update indexer tests for _chunk_file_lines rename + AST dispatcher"
```

---

## Task 5: Reinstall uv tool

- [ ] **Step 1: Reinstall to pick up new code**

```bash
uv tool install --force /Users/gzamboni/Code/ai/codebase-mcp
```

- [ ] **Step 2: Smoke test**

```bash
codebase-mcp --help
```

Expected: shows all 5 commands without error.

- [ ] **Step 3: Commit** (nothing to commit — install-only step)

No commit needed.

---

## Self-Review

**Spec coverage:**
- ✅ New `ast_chunker.py` with `chunk_file_ast()` returning `None` on failure
- ✅ `EXT_TO_LANG` maps all 7 language families (py, ts, tsx, js, jsx, go, rs, java, tf)
- ✅ `SEMANTIC_NODES` per language matches spec table exactly
- ✅ Lazy parser init — `ImportError` caught, returns `None`
- ✅ Walk stops at semantic nodes (no nested duplication)
- ✅ `ERROR` nodes ignored
- ✅ `node_type` added to chunk payload
- ✅ `MAX_CHUNK_CHARS` truncation applied
- ✅ `chunk_file` dispatcher in `indexer.py` — try AST, fallback on `None` or exception
- ✅ `_chunk_file_lines` rename — no logic change
- ✅ `.tf` added to `INDEXED_EXTENSIONS`
- ✅ All 11 required tests in `test_ast_chunker.py`
- ✅ `test_chunk_file_uses_ast_for_python` in `test_indexer.py`
- ✅ `chunk_file` references updated to `_chunk_file_lines` in existing tests

**Type consistency:**
- `chunk_file_ast(content: str, filepath: str, repo_path: str) -> list[dict] | None` — used in dispatcher with `None` check ✅
- `_chunk_file_lines` has same signature as old `chunk_file` — no callers break ✅
- Chunk dict schema: `text, file, start_line, end_line, repo_path, node_type` — `node_type` only present in AST chunks, absent in line chunks — `test_chunk_file_falls_back_to_lines` asserts this ✅
