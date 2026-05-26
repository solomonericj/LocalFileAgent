import pytest

from session_manager import SessionManager
from gui import SessionDialog, MainWindow


def test_session_list_survives_message_without_role(qtbot, tmp_path):
    """A hand-edited/corrupt session (message missing 'role') must not crash the
    Sessions dialog when it builds the preview list."""
    sm = SessionManager(session_dir=tmp_path)
    sm.save({"model": "m", "embed_model": "e", "files": [],
             "messages": [{"text": "no role key here"}], "summaries": {}})

    dlg = SessionDialog(sm)          # _load_list runs in __init__
    qtbot.addWidget(dlg)
    assert dlg._list.count() == 1    # listed, didn't raise


def test_load_session_survives_message_without_role(qtbot, monkeypatch, tmp_path):
    """Replaying a corrupt session's history must skip malformed messages
    rather than raising KeyError."""
    monkeypatch.setattr(MainWindow, "_fetch_models", lambda self: None)
    w = MainWindow()
    qtbot.addWidget(w)

    session = {
        "model": "m", "embed_model": "e", "files": [],
        "messages": [{"role": "user", "content": "hi"}, {"text": "no role"}],
        "summaries": {}, "created": "2026-01-01T00:00:00",
    }
    w._load_session(session)   # must not raise
