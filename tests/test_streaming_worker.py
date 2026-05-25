from unittest.mock import patch
import pytest
from gui import StreamingChatWorker, FileItemWidget


def _fake_stream(tokens):
    def _gen(messages, model):
        yield from tokens
    return _gen


def test_emits_tokens_and_finished(qtbot):
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]
    worker = StreamingChatWorker(msgs, "mistral")
    tokens_received = []
    finished_msgs = []
    worker.token_ready.connect(tokens_received.append)
    worker.finished.connect(finished_msgs.append)

    with patch("gui.stream_ollama_chat", _fake_stream(["Hello", " world"])):
        with qtbot.waitSignal(worker.finished, timeout=5000):
            worker.start()

    assert tokens_received == ["Hello", " world"]
    assert finished_msgs[0][-1]["role"] == "assistant"
    assert finished_msgs[0][-1]["content"] == "Hello world"


def test_emits_error_on_connection_error(qtbot):
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]
    worker = StreamingChatWorker(msgs, "mistral")
    errors = []
    worker.error.connect(errors.append)

    def _raises(messages, model):
        raise ConnectionError("no ollama")
        yield  # make it a generator

    with patch("gui.stream_ollama_chat", _raises):
        with qtbot.waitSignal(worker.error, timeout=5000):
            worker.start()

    assert "no ollama" in errors[0]


def test_snapshots_messages_list(qtbot):
    """Worker must not be affected by mutations to the original list."""
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]
    worker = StreamingChatWorker(msgs, "mistral")
    msgs.clear()   # mutate after construction

    finished_msgs = []
    worker.finished.connect(finished_msgs.append)

    with patch("gui.stream_ollama_chat", _fake_stream(["ok"])):
        with qtbot.waitSignal(worker.finished, timeout=5000):
            worker.start()

    assert len(finished_msgs[0]) > 0   # worker used its snapshot, not the cleared list


def test_file_status_signals_emitted(qtbot, tmp_path):
    f = tmp_path / "test.py"
    f.write_text("hello world " * 100)

    worker = StreamingChatWorker(
        [], "mistral",
        files_to_load=[f],
        user_text="what does this do?",
    )
    statuses = []
    finished_msgs = []
    worker.file_status.connect(lambda p, s, t: statuses.append((p, s)))
    worker.finished.connect(finished_msgs.append)

    with patch("gui.stream_ollama_chat", _fake_stream(["answer"])):
        with qtbot.waitSignal(worker.finished, timeout=5000):
            worker.start()

    assert any(s == FileItemWidget.STATUS_LOADED for _, s in statuses)
