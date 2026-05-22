import json
from pathlib import Path

import pytest


@pytest.fixture
def agent_config_dir(tmp_path, monkeypatch):
    """Patch Path.home so all agent config paths point into tmp_path."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


def test_claude_code_config_path(agent_config_dir):
    from yacodebase_mcp.agents import AGENTS

    expected = agent_config_dir / ".claude" / "settings.json"
    assert AGENTS["claude-code"].config_path() == expected


def test_install_creates_config(agent_config_dir):
    from yacodebase_mcp.agents import AGENTS, MCP_SERVER_NAME, install_agent

    agent = AGENTS["claude-code"]
    result = install_agent(agent)
    assert result == "installed"
    data = json.loads(agent.config_path().read_text())
    assert MCP_SERVER_NAME in data["mcpServers"]
    assert data["mcpServers"][MCP_SERVER_NAME]["command"].endswith("yacodebase-mcp")
    assert data["mcpServers"][MCP_SERVER_NAME]["args"] == ["serve"]


def test_install_idempotent(agent_config_dir):
    from yacodebase_mcp.agents import AGENTS, install_agent

    agent = AGENTS["claude-code"]
    install_agent(agent)
    result = install_agent(agent)
    assert result == "already"


def test_install_dry_run_no_write(agent_config_dir):
    from yacodebase_mcp.agents import AGENTS, install_agent

    agent = AGENTS["claude-code"]
    result = install_agent(agent, dry_run=True)
    assert result == "dry_run"
    assert not agent.config_path().exists()


def test_install_merges_existing_config(agent_config_dir):
    from yacodebase_mcp.agents import AGENTS, MCP_SERVER_NAME, install_agent

    agent = AGENTS["claude-code"]
    agent.config_path().parent.mkdir(parents=True, exist_ok=True)
    agent.config_path().write_text(json.dumps({"other_setting": True}))

    install_agent(agent)
    data = json.loads(agent.config_path().read_text())
    assert data["other_setting"] is True
    assert MCP_SERVER_NAME in data["mcpServers"]


def test_vscode_format(agent_config_dir):
    from yacodebase_mcp.agents import AGENTS, MCP_SERVER_NAME, install_agent

    agent = AGENTS["copilot"]
    install_agent(agent)
    data = json.loads(agent.config_path().read_text())
    entry = data["mcp"]["servers"][MCP_SERVER_NAME]
    assert entry["type"] == "stdio"
    assert entry["command"].endswith("yacodebase-mcp")


def test_zed_format(agent_config_dir):
    from yacodebase_mcp.agents import AGENTS, MCP_SERVER_NAME, install_agent

    agent = AGENTS["zed"]
    install_agent(agent)
    data = json.loads(agent.config_path().read_text())
    entry = data["context_servers"][MCP_SERVER_NAME]
    assert entry["command"]["path"].endswith("yacodebase-mcp")
    assert entry["command"]["args"] == ["serve"]


def test_is_installed_false_when_absent(agent_config_dir):
    from yacodebase_mcp.agents import AGENTS

    assert AGENTS["claude-code"].is_installed() is False


def test_is_installed_true_after_install(agent_config_dir):
    from yacodebase_mcp.agents import AGENTS, install_agent

    agent = AGENTS["cursor"]
    install_agent(agent)
    assert agent.is_installed() is True


def test_codex_format(agent_config_dir):
    from yacodebase_mcp.agents import AGENTS, MCP_SERVER_NAME, install_agent

    agent = AGENTS["codex"]
    install_agent(agent)
    import tomllib

    data = tomllib.loads(agent.config_path().read_bytes().decode("utf-8"))
    entry = data["mcp_servers"][MCP_SERVER_NAME]
    assert entry["command"].endswith("yacodebase-mcp")
    assert entry["args"] == ["serve"]


def test_all_agents_present():
    from yacodebase_mcp.agents import AGENTS

    expected = {"claude-code", "cursor", "windsurf", "copilot", "zed", "opencode", "codex"}
    assert set(AGENTS.keys()) == expected


# ── inject tests ──────────────────────────────────────────────────────────────


def test_inject_creates_file(tmp_path):
    from yacodebase_mcp.agents import INJECT_TARGETS

    target = INJECT_TARGETS["claude-code"]
    result = target.inject(tmp_path)
    assert result == "injected"
    assert target.instructions_path(tmp_path).exists()
    assert "search_codebase" in target.instructions_path(tmp_path).read_text()


def test_inject_idempotent(tmp_path):
    from yacodebase_mcp.agents import INJECT_TARGETS

    target = INJECT_TARGETS["claude-code"]
    target.inject(tmp_path)
    result = target.inject(tmp_path)
    assert result == "already"


def test_inject_appends_to_existing(tmp_path):
    from yacodebase_mcp.agents import INJECT_TARGETS

    target = INJECT_TARGETS["claude-code"]
    p = target.instructions_path(tmp_path)
    p.write_text("# My project\n\nExisting content.\n")
    target.inject(tmp_path)
    text = p.read_text()
    assert "Existing content." in text
    assert "search_codebase" in text


def test_inject_dry_run_no_write(tmp_path):
    from yacodebase_mcp.agents import INJECT_TARGETS

    target = INJECT_TARGETS["claude-code"]
    result = target.inject(tmp_path, dry_run=True)
    assert result == "dry_run"
    assert not target.instructions_path(tmp_path).exists()


def test_eject_removes_block(tmp_path):
    from yacodebase_mcp.agents import INJECT_TARGETS

    target = INJECT_TARGETS["claude-code"]
    target.inject(tmp_path)
    result = target.eject(tmp_path)
    assert result == "ejected"
    assert not target.is_injected(tmp_path)


def test_eject_preserves_surrounding_content(tmp_path):
    from yacodebase_mcp.agents import INJECT_TARGETS

    target = INJECT_TARGETS["claude-code"]
    p = target.instructions_path(tmp_path)
    p.write_text("# My project\n\nExisting content.\n")
    target.inject(tmp_path)
    target.eject(tmp_path)
    text = p.read_text()
    assert "Existing content." in text
    assert "search_codebase" not in text


def test_eject_not_found(tmp_path):
    from yacodebase_mcp.agents import INJECT_TARGETS

    target = INJECT_TARGETS["claude-code"]
    result = target.eject(tmp_path)
    assert result == "not_found"


def test_all_inject_targets_present():
    from yacodebase_mcp.agents import INJECT_TARGETS

    expected = {"claude-code", "cursor", "copilot", "codex"}
    assert set(INJECT_TARGETS.keys()) == expected
