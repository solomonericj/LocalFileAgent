import pytest

from gui import MainWindow

CURSOR = "▌"   # ▌ blinking-cursor marker used during streaming


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr(MainWindow, "_fetch_models", lambda self: None)
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def test_streaming_preserves_multiline_text(window):
    """Tokens containing newlines (every real LLM reply) must render intact —
    not get reordered or lose characters (regression in _on_token_ready)."""
    window._start_stream_bubble("mistral")
    for tok in ["Here are steps:\n", "1. first\n", "2. second"]:
        window._on_token_ready(tok)
    window._finish_stream()

    assert "Here are steps:\n1. first\n2. second" in window._chat_history.toPlainText()


def test_streaming_removes_cursor_marker(window):
    """After a multi-line stream finishes, no ▌ marker may remain in the text."""
    window._start_stream_bubble("mistral")
    window._on_token_ready("line one\nline two")
    window._finish_stream()

    assert CURSOR not in window._chat_history.toPlainText()
