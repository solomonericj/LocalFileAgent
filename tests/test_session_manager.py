import pytest
from session_manager import SessionManager


def _s(**kwargs) -> dict:
    base = {"model": "mistral", "files": [], "messages": [], "summaries": {}}
    base.update(kwargs)
    return base


def test_save_creates_json_file(tmp_path):
    sm = SessionManager(session_dir=tmp_path)
    path = sm.save(_s())
    assert path.exists() and path.suffix == ".json"


def test_save_adds_created_timestamp(tmp_path):
    sm = SessionManager(session_dir=tmp_path)
    session = _s()
    sm.save(session)
    assert "created" in session


def test_save_preserves_existing_created(tmp_path):
    sm = SessionManager(session_dir=tmp_path)
    session = _s(created="2026-01-01T10:00:00")
    sm.save(session)
    assert session["created"] == "2026-01-01T10:00:00"


def test_list_newest_first(tmp_path):
    sm = SessionManager(session_dir=tmp_path)
    sm.save(_s(created="2026-01-01T10:00:00"))
    sm.save(_s(created="2026-01-02T10:00:00"))
    listed = sm.list()
    assert len(listed) == 2
    assert listed[0]["created"] == "2026-01-02T10:00:00"


def test_list_includes_path_key(tmp_path):
    sm = SessionManager(session_dir=tmp_path)
    path = sm.save(_s())
    assert sm.list()[0]["_path"] == str(path)


def test_list_skips_corrupt_files(tmp_path):
    (tmp_path / "bad.json").write_text("not json", encoding="utf-8")
    sm = SessionManager(session_dir=tmp_path)
    assert sm.list() == []


def test_load_returns_data(tmp_path):
    sm = SessionManager(session_dir=tmp_path)
    path = sm.save(_s(model="gemma3"))
    assert sm.load(str(path))["model"] == "gemma3"


def test_delete_removes_file(tmp_path):
    sm = SessionManager(session_dir=tmp_path)
    path = sm.save(_s())
    sm.delete(str(path))
    assert not path.exists()


def test_delete_noop_if_missing(tmp_path):
    sm = SessionManager(session_dir=tmp_path)
    sm.delete(str(tmp_path / "ghost.json"))  # must not raise
