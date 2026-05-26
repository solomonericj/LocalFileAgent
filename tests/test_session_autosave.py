from datetime import datetime as real_dt

import pytest

from gui import MainWindow
from session_manager import SessionManager


@pytest.fixture
def window(qtbot, monkeypatch):
    # Avoid spawning the Ollama model-fetch thread during construction.
    monkeypatch.setattr(MainWindow, "_fetch_models", lambda self: None)
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def test_auto_save_reuses_single_file(window, monkeypatch, tmp_path):
    """Auto-save after every reply must update ONE session file, not spawn a
    new file per turn (regression: a fresh `created` stamp each save)."""
    import session_manager as sm_mod

    # Force a distinct, increasing timestamp on each save so a regression would
    # produce distinct filenames (proving the fix reuses one identity).
    stamps = iter([
        real_dt(2026, 5, 26, 10, 0, 0),
        real_dt(2026, 5, 26, 10, 0, 1),
        real_dt(2026, 5, 26, 10, 0, 2),
    ])

    class FakeDateTime:
        @staticmethod
        def now():
            return next(stamps)

    monkeypatch.setattr(sm_mod, "datetime", FakeDateTime)

    window._session_manager = SessionManager(session_dir=tmp_path)
    window._session_created = None

    window._chat_messages = [{"role": "user", "content": "q1"}]
    window._auto_save()
    window._chat_messages.append({"role": "assistant", "content": "a1"})
    window._auto_save()
    window._chat_messages.append({"role": "user", "content": "q2"})
    window._auto_save()

    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1


def test_auto_save_persists_latest_messages(window, monkeypatch, tmp_path):
    """The single retained file must hold the most recent conversation state."""
    import session_manager as sm_mod

    stamps = iter([
        real_dt(2026, 5, 26, 11, 0, 0),
        real_dt(2026, 5, 26, 11, 0, 1),
    ])

    class FakeDateTime:
        @staticmethod
        def now():
            return next(stamps)

    monkeypatch.setattr(sm_mod, "datetime", FakeDateTime)

    window._session_manager = SessionManager(session_dir=tmp_path)
    window._session_created = None

    window._chat_messages = [{"role": "user", "content": "q1"}]
    window._auto_save()
    window._chat_messages.append({"role": "assistant", "content": "a1"})
    window._auto_save()

    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    saved = SessionManager(session_dir=tmp_path).load(str(files[0]))
    assert saved["messages"][-1] == {"role": "assistant", "content": "a1"}
