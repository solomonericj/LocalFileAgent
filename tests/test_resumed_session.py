from pathlib import Path

import pytest

import gui
from gui import MainWindow


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr(MainWindow, "_fetch_models", lambda self: None)
    w = MainWindow()
    qtbot.addWidget(w)
    return w


class _Sig:
    def connect(self, *a, **k):
        pass


class _FakeWorker:
    last_kwargs = None

    def __init__(self, *args, **kwargs):
        _FakeWorker.last_kwargs = kwargs
        for s in ("token_ready", "timing", "finished", "error",
                  "context_info", "index_ready", "file_status"):
            setattr(self, s, _Sig())

    def start(self):
        pass


def test_resumed_session_dispatches_index_rebuild(window, monkeypatch, tmp_path):
    """Continuing a freshly-loaded session (history present, no cached index)
    must rebuild the index on the next turn instead of sending a contextless
    RAG prompt (#8)."""
    monkeypatch.setattr(gui, "StreamingChatWorker", _FakeWorker)

    f = tmp_path / "a.txt"
    f.write_text("hi")
    window._sidebar.add_path(str(f))

    # Mimic a just-loaded session: files marked loaded, index not yet built.
    window._chat_files_loaded = True
    window._rag_index = None
    window._chat_messages = [
        {"role": "system", "content": gui.RAG_SYSTEM},
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "ans"},
    ]
    window._chat_input.setText("new question")
    window._send_chat()

    assert _FakeWorker.last_kwargs.get("preserve_history") is True
    assert _FakeWorker.last_kwargs.get("files_to_load")   # files passed for rebuild


def test_normal_followup_turn_does_not_rebuild(window, monkeypatch, tmp_path):
    """With a cached index already present, a follow-up turn must NOT re-index —
    it just sends the cached index through for retrieval."""
    monkeypatch.setattr(gui, "StreamingChatWorker", _FakeWorker)

    window._chat_files_loaded = True
    window._rag_index = object()   # cached index present
    window._chat_messages = [
        {"role": "system", "content": gui.RAG_SYSTEM},
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "ans"},
    ]
    window._chat_input.setText("another question")
    window._send_chat()

    assert _FakeWorker.last_kwargs.get("preserve_history") in (None, False)
    assert _FakeWorker.last_kwargs.get("files_to_load") is None
    assert _FakeWorker.last_kwargs.get("rag_index") is window._rag_index
