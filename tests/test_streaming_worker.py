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
        use_rag=False,
    )
    statuses = []
    finished_msgs = []
    worker.file_status.connect(lambda p, s, t: statuses.append((p, s)))
    worker.finished.connect(finished_msgs.append)

    with patch("gui.stream_ollama_chat", _fake_stream(["answer"])):
        with qtbot.waitSignal(worker.finished, timeout=5000):
            worker.start()

    assert any(s == FileItemWidget.STATUS_LOADED for _, s in statuses)


def test_stops_streaming_on_interruption(qtbot, monkeypatch):
    """When close requests interruption, the token loop must stop promptly so
    the thread can finish and not be destroyed mid-run (#7)."""
    worker = StreamingChatWorker([{"role": "user", "content": "hi"}], "mistral")
    monkeypatch.setattr(worker, "isInterruptionRequested", lambda: True)
    tokens = []
    worker.token_ready.connect(tokens.append)

    with patch("gui.stream_ollama_chat", _fake_stream(["a", "b", "c"])):
        with qtbot.waitSignal(worker.finished, timeout=5000):
            worker.start()

    assert tokens == []   # interrupted before emitting any token


def test_preserve_history_rebuilds_index_keeping_history(qtbot, monkeypatch, tmp_path):
    """Resumed-session path (#8): rebuild the index on this turn without wiping
    the replayed history, and still inject retrieved context into the request."""
    import gui
    from rag import Chunk

    class _FakeIndex:
        def __len__(self):
            return 3

    monkeypatch.setattr(gui, "build_index", lambda files, embed_model: _FakeIndex())
    monkeypatch.setattr(gui, "retrieve", lambda *a, **k: [Chunk("a.txt", "a.txt", "CTX", 0)])

    captured = {}

    def fake_stream(messages, model):
        captured["messages"] = messages
        yield "ok"

    monkeypatch.setattr(gui, "stream_ollama_chat", fake_stream)

    f = tmp_path / "a.txt"
    f.write_text("hello")
    history = [
        {"role": "system", "content": gui.RAG_SYSTEM},
        {"role": "user", "content": "old q"},
        {"role": "assistant", "content": "old a"},
        {"role": "user", "content": "new q"},
    ]
    worker = gui.StreamingChatWorker(
        history, "mistral",
        files_to_load=[f], user_text="new q",
        preserve_history=True, embed_model="e",
    )
    index_emitted = []
    finished = []
    worker.index_ready.connect(index_emitted.append)
    worker.finished.connect(finished.append)
    with qtbot.waitSignal(worker.finished, timeout=5000):
        worker.start()

    # Retrieval fired into the outgoing last turn...
    assert "CTX" in captured["messages"][-1]["content"]
    # ...the rebuilt index was published for later turns...
    assert index_emitted
    # ...and the replayed history survived (old turns intact, plain new turn).
    msgs = finished[0]
    assert msgs[1]["content"] == "old q"
    assert msgs[-2] == {"role": "user", "content": "new q"}
    assert msgs[-1] == {"role": "assistant", "content": "ok"}


def test_rag_turn_injects_context_but_keeps_plain_history(qtbot, monkeypatch):
    import gui
    from rag import Chunk

    monkeypatch.setattr(gui, "retrieve", lambda *a, **k: [Chunk("a.txt", "a.txt", "ground truth", 0)])

    captured = {}
    def fake_stream(messages, model):
        captured["messages"] = messages
        yield "ok"
    monkeypatch.setattr(gui, "stream_ollama_chat", fake_stream)

    history = [
        {"role": "system", "content": gui.RAG_SYSTEM},
        {"role": "user", "content": "what is the truth?"},
    ]
    worker = gui.StreamingChatWorker(
        history, "mistral",
        rag_index=object(), embed_model="nomic-embed-text",
        top_k=5, user_text="what is the truth?",
    )
    finished = []
    worker.finished.connect(finished.append)
    with qtbot.waitSignal(worker.finished, timeout=5000):
        worker.start()

    # Outgoing request carried the retrieved context...
    assert "ground truth" in captured["messages"][-1]["content"]
    # ...but persisted history kept the plain user text and appended the reply.
    assert finished[0][-2]["content"] == "what is the truth?"
    assert finished[0][-1] == {"role": "assistant", "content": "ok"}
