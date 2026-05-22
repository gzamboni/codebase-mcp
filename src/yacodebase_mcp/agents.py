from __future__ import annotations

import copy
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

MCP_SERVER_NAME = "codebase-search"
MCP_SERVER_CMD = "yacodebase-mcp"
MCP_SERVER_ARGS: tuple[str, ...] = ("serve",)


@dataclass
class Agent:
    name: str
    label: str
    _get_path: Callable[[], Path] = field(repr=False)
    _merge_fn: Callable[[dict], dict] = field(repr=False)
    _check_fn: Callable[[dict], bool] = field(repr=False)
    _format: str = "json"

    def config_path(self) -> Path:
        return self._get_path()

    def read_config(self) -> dict:
        p = self.config_path()
        if not p.exists():
            return {}
        if self._format == "toml":
            import tomllib

            return tomllib.loads(p.read_text(encoding="utf-8"))
        return json.loads(p.read_text(encoding="utf-8"))

    def write_config(self, data: dict) -> None:
        p = self.config_path()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            if self._format == "toml":
                import tomli_w

                p.write_text(tomli_w.dumps(data), encoding="utf-8")
            else:
                p.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        except OSError as e:
            raise OSError(f"Cannot write {self.name} config to {p}: {e}") from e

    def is_installed(self, data: dict | None = None) -> bool:
        if data is None:
            data = self.read_config()
        return self._check_fn(data)

    def merge(self, data: dict) -> dict:
        return self._merge_fn(data)


def _appdata() -> Path:
    """Windows %APPDATA% or fallback."""
    return Path(os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming"))


# ── JSON agent helpers ────────────────────────────────────────────────────────

def _std_entry() -> dict:
    return {"command": MCP_SERVER_CMD, "args": list(MCP_SERVER_ARGS)}


def _merge_mcpservers(data: dict) -> dict:
    result = copy.deepcopy(data)
    result.setdefault("mcpServers", {})[MCP_SERVER_NAME] = _std_entry()
    return result


def _check_mcpservers(data: dict) -> bool:
    return MCP_SERVER_NAME in data.get("mcpServers", {})


def _merge_vscode(data: dict) -> dict:
    result = copy.deepcopy(data)
    entry = {"type": "stdio", "command": MCP_SERVER_CMD, "args": list(MCP_SERVER_ARGS)}
    result.setdefault("mcp", {}).setdefault("servers", {})[MCP_SERVER_NAME] = entry
    return result


def _check_vscode(data: dict) -> bool:
    return MCP_SERVER_NAME in data.get("mcp", {}).get("servers", {})


def _merge_zed(data: dict) -> dict:
    result = copy.deepcopy(data)
    result.setdefault("context_servers", {})[MCP_SERVER_NAME] = {
        "command": {"path": MCP_SERVER_CMD, "args": list(MCP_SERVER_ARGS)}
    }
    return result


def _check_zed(data: dict) -> bool:
    return MCP_SERVER_NAME in data.get("context_servers", {})


# ── TOML agent helpers (Codex CLI) ────────────────────────────────────────────

def _merge_codex(data: dict) -> dict:
    result = copy.deepcopy(data)
    result.setdefault("mcp_servers", {})[MCP_SERVER_NAME] = {
        "command": MCP_SERVER_CMD,
        "args": list(MCP_SERVER_ARGS),
    }
    return result


def _check_codex(data: dict) -> bool:
    return MCP_SERVER_NAME in data.get("mcp_servers", {})


# ── OS-aware path functions ───────────────────────────────────────────────────

def _copilot_path() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Code" / "User" / "settings.json"
    if sys.platform == "win32":
        return _appdata() / "Code" / "User" / "settings.json"
    return Path.home() / ".config" / "Code" / "User" / "settings.json"


def _zed_path() -> Path:
    if sys.platform == "win32":
        return _appdata() / "Zed" / "settings.json"
    return Path.home() / ".config" / "zed" / "settings.json"


def _opencode_path() -> Path:
    if sys.platform == "win32":
        return _appdata() / "opencode" / "config.json"
    return Path.home() / ".config" / "opencode" / "config.json"


# ── Agent registry ────────────────────────────────────────────────────────────

AGENTS: dict[str, Agent] = {
    "claude-code": Agent(
        name="claude-code",
        label="Claude Code",
        _get_path=lambda: Path.home() / ".claude" / "settings.json",
        _merge_fn=_merge_mcpservers,
        _check_fn=_check_mcpservers,
    ),
    "cursor": Agent(
        name="cursor",
        label="Cursor",
        _get_path=lambda: Path.home() / ".cursor" / "mcp.json",
        _merge_fn=_merge_mcpservers,
        _check_fn=_check_mcpservers,
    ),
    "windsurf": Agent(
        name="windsurf",
        label="Windsurf",
        _get_path=lambda: Path.home() / ".codeium" / "windsurf" / "mcp_config.json",
        _merge_fn=_merge_mcpservers,
        _check_fn=_check_mcpservers,
    ),
    "copilot": Agent(
        name="copilot",
        label="GitHub Copilot (VS Code)",
        _get_path=_copilot_path,
        _merge_fn=_merge_vscode,
        _check_fn=_check_vscode,
    ),
    "zed": Agent(
        name="zed",
        label="Zed",
        _get_path=_zed_path,
        _merge_fn=_merge_zed,
        _check_fn=_check_zed,
    ),
    "opencode": Agent(
        name="opencode",
        label="OpenCode",
        _get_path=_opencode_path,
        _merge_fn=_merge_mcpservers,
        _check_fn=_check_mcpservers,
    ),
    "codex": Agent(
        name="codex",
        label="OpenAI Codex CLI",
        _get_path=lambda: Path.home() / ".codex" / "config.toml",
        _merge_fn=_merge_codex,
        _check_fn=_check_codex,
        _format="toml",
    ),
}


def install_agent(agent: Agent, dry_run: bool = False) -> str:
    """Install MCP server entry into agent config. Returns 'already', 'dry_run', or 'installed'."""
    if agent.is_installed():
        return "already"
    data = agent.read_config()
    new_data = agent.merge(data)
    if dry_run:
        return "dry_run"
    agent.write_config(new_data)
    return "installed"
