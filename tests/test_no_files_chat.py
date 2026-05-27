"""GUI: sending a message with zero files starts general chat instead of warning."""
import pytest

import gui
from gui import MainWindow


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr(MainWindow, "_fetch_models", lambda self: None)
    w = MainWindow()
    qtbot.addWidget(w)
    return w


class _RecordingWorker:
    """Inert worker stand-in that records its construction kwargs."""
    instances = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        for s in ("token_ready", "timing", "finished", "error", "context_info",
                  "index_ready", "file_status"):
            setattr(self, s, _Sig())
        _RecordingWorker.instances.append(self)

    def start(self):
        pass


class _Sig:
    def connect(self, *a, **k):
        pass


def test_send_with_no_files_dispatches_general_chat_worker(window, monkeypatch):
    _RecordingWorker.instances = []
    monkeypatch.setattr(gui, "StreamingChatWorker", _RecordingWorker)
    warned = []
    monkeypatch.setattr(gui.QMessageBox, "warning",
                        lambda *a, **k: warned.append(a))

    # No files in the sidebar, first turn.
    assert window._sidebar.get_valid_paths() == []
    window._chat_input.setText("who are you?")
    window._send_chat()

    # A worker was dispatched (general chat), and no "No Files" warning shown.
    assert _RecordingWorker.instances, "expected a chat worker to be dispatched"
    assert warned == []
    worker = _RecordingWorker.instances[-1]
    assert worker.kwargs.get("user_text") == "who are you?"
    # No files passed to load.
    assert worker.kwargs.get("files_to_load") is None
