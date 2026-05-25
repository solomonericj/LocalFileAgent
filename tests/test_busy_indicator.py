import pytest

from gui import MainWindow


@pytest.fixture
def window(qtbot, monkeypatch):
    # Avoid spawning the Ollama model-fetch thread during construction.
    monkeypatch.setattr(MainWindow, "_fetch_models", lambda self: None)
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def test_busy_bar_hidden_initially(window):
    assert window._busy_bar.isVisibleTo(window) is False
    assert window._status_bar.currentMessage() == "Ready"


def test_set_busy_shows_bar_and_message(window):
    window._set_busy("Thinking…")
    assert window._busy_bar.isVisibleTo(window) is True
    assert window._status_bar.currentMessage() == "Thinking…"


def test_busy_bar_is_indeterminate(window):
    # range (0, 0) renders as a continuous animation, not a percentage.
    assert window._busy_bar.minimum() == 0
    assert window._busy_bar.maximum() == 0


def test_clear_busy_hides_bar_and_resets(window):
    window._set_busy("Thinking…")
    window._clear_busy()
    assert window._busy_bar.isVisibleTo(window) is False
    assert window._status_bar.currentMessage() == "Ready"


def test_clear_busy_accepts_custom_message(window):
    window._set_busy("Summarizing report.md…")
    window._clear_busy("Done")
    assert window._busy_bar.isVisibleTo(window) is False
    assert window._status_bar.currentMessage() == "Done"
