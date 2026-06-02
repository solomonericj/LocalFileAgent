import pytest

import gui
from gui import MainWindow


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr(MainWindow, "_fetch_models", lambda self: None)
    w = MainWindow()
    qtbot.addWidget(w)
    return w


class _Running:
    """A worker stand-in that reports as running (and supports the shutdown
    calls the window makes on it at teardown via closeEvent)."""

    def isRunning(self):
        return True

    def requestInterruption(self):
        pass

    def wait(self, ms=None):
        return True


class _Sig:
    def connect(self, *a, **k):
        pass


class _NeverStarts:
    """Inert stand-in worker — its construction means the guard let a second
    worker through (which the guard tests assert must NOT happen)."""

    def __init__(self, *a, **k):
        for s in ("token_ready", "timing", "finished", "error", "context_info",
                  "index_ready", "file_status", "file_done", "progress"):
            setattr(self, s, _Sig())

    def start(self):
        pass


# ── #6: a single in-flight guard prevents overlapping workers ────────────────

def test_no_operation_in_progress_initially(window):
    assert window._operation_in_progress() is False


def test_summarize_refused_while_another_worker_runs(window, monkeypatch):
    monkeypatch.setattr(gui, "SummarizeWorker", _NeverStarts)
    window._chat_worker = _Running()
    window._run_single_summarize("anything.py")
    # Guard must bail out before spawning a second worker.
    assert not hasattr(window, "_summarize_worker")


def test_send_chat_refused_while_summarize_runs(window, monkeypatch):
    monkeypatch.setattr(gui, "StreamingChatWorker", _NeverStarts)
    window._summarize_worker = _Running()
    window._chat_files_loaded = True   # skip the file-loading branch (no dialog)
    window._chat_input.setText("hello")
    window._send_chat()
    assert not hasattr(window, "_chat_worker")
    assert window._chat_input.text() == "hello"   # input preserved on refusal


# ── #7: closing the window waits on running threads ──────────────────────────

def test_shutdown_waits_for_running_workers(window):
    events = []

    class FakeWorker:
        def isRunning(self):
            return True

        def requestInterruption(self):
            events.append("interrupt")

        def wait(self, ms=None):
            events.append(("wait", ms))
            return True

    window._chat_worker = FakeWorker()
    window._shutdown_workers()
    assert "interrupt" in events
    assert any(isinstance(e, tuple) and e[0] == "wait" for e in events)


def test_shutdown_ignores_finished_workers(window):
    events = []

    class Finished:
        def isRunning(self):
            return False

        def requestInterruption(self):
            events.append("interrupt")

        def wait(self, ms=None):
            events.append("wait")

    window._summarize_worker = Finished()
    window._shutdown_workers()
    assert events == []
