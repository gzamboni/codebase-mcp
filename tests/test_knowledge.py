import pytest

from yacodebase_mcp.knowledge import (
    add_decision,
    add_note,
    get_notes,
    search_decisions,
    update_decision,
)


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("YACODEBASE_MCP_DATA_DIR", str(tmp_path))


def test_add_and_search_decision():
    add_decision(
        "Use Qdrant", "Qdrant chosen for vector storage because in-process", "architecture"
    )
    results = search_decisions("Qdrant")
    assert len(results) == 1
    assert results[0]["title"] == "Use Qdrant"
    assert results[0]["status"] == "active"


def test_search_decision_by_category():
    add_decision("Auth via JWT", "JWT for stateless auth", "security")
    add_decision("DB is Postgres", "Postgres for relational data", "architecture")
    results = search_decisions(category="security")
    assert len(results) == 1
    assert results[0]["title"] == "Auth via JWT"


def test_update_decision_status():
    add_decision("Old approach", "We used to do X", "architecture")
    results = search_decisions("Old approach")
    decision_id = results[0]["id"]
    update_decision(decision_id, status="superseded")
    updated = search_decisions("Old approach")
    assert updated[0]["status"] == "superseded"


def test_add_and_get_notes():
    add_note("Remember to update tests after refactor", scope="project")
    notes = get_notes()
    assert len(notes) == 1
    assert "tests" in notes[0]["content"]


def test_get_notes_by_scope():
    add_note("Note for auth module", scope="file", reference="src/auth.py")
    add_note("Project-wide note", scope="project")
    notes = get_notes(scope="file")
    assert len(notes) == 1
    assert notes[0]["reference"] == "src/auth.py"
