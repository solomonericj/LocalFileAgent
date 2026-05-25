from pathlib import Path

import LocalfileAgent as lfa
import rag
from rag import Chunk


def test_run_chat_rag_keeps_plain_history(monkeypatch):
    """CLI must persist plain user text and inject retrieved context only into
    the outgoing request — never accumulate composed prompts in history."""
    # Non-empty fake index (len > 0); real build is bypassed.
    monkeypatch.setattr(rag, "build_index", lambda files, embed_model: [1, 2, 3])
    monkeypatch.setattr(
        rag, "retrieve",
        lambda index, query, embed_model, k: [Chunk("a.txt", "a.txt", "CTX", 0)],
    )

    captured = []

    def fake_chat(messages, model):
        captured.append([dict(m) for m in messages])
        return "answer", messages + [{"role": "assistant", "content": "answer"}]

    monkeypatch.setattr(lfa, "query_ollama_chat", fake_chat)

    inputs = iter(["first question", "second question", "/quit"])
    monkeypatch.setattr("builtins.input", lambda _="": next(inputs))

    lfa.run_chat([Path("a.txt")], "mistral", embed_model="e", top_k=3, use_rag=True)

    # Both outgoing requests carry the injected context in the latest turn.
    assert "CTX" in captured[0][-1]["content"]
    assert "CTX" in captured[1][-1]["content"]

    # The second turn's history (everything before the latest turn) must hold
    # only plain text — no leaked "Context excerpts" from the first turn.
    earlier = captured[1][:-1]
    assert all("Context excerpts" not in m["content"] for m in earlier)
    user_contents = [m["content"] for m in earlier if m["role"] == "user"]
    assert "first question" in user_contents
