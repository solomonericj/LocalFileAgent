"""General-assistant chat mode: answer any question, work with zero files."""
from pathlib import Path

import LocalfileAgent as lfa
import rag


# ── Prompts must permit outside knowledge ───────────────────────────────────

def test_rag_system_permits_general_knowledge():
    """RAG_SYSTEM must no longer fence answers to the excerpts only."""
    sysmsg = lfa.RAG_SYSTEM.lower()
    # The old refusal language is gone...
    assert "using only" not in sysmsg
    assert "say so plainly" not in sysmsg
    # ...and it still asks the model to cite sources when it uses excerpts.
    assert "cite" in sysmsg


def test_general_system_constant_has_no_file_language():
    """GENERAL_SYSTEM is the no-files prompt — a plain assistant, no excerpts."""
    sysmsg = lfa.GENERAL_SYSTEM.lower()
    assert "assistant" in sysmsg
    assert "excerpt" not in sysmsg


def test_general_system_is_exported_to_gui():
    """The GUI imports GENERAL_SYSTEM from the core module."""
    import gui
    assert gui.GENERAL_SYSTEM == lfa.GENERAL_SYSTEM


# ── CLI chat with zero files ─────────────────────────────────────────────────

def test_run_chat_no_files_uses_general_prompt(monkeypatch):
    """run_chat([]) seeds GENERAL_SYSTEM, never indexes, and sends the plain
    question with the general system prompt."""
    def _boom(*a, **k):
        raise AssertionError("build_index must not run with no files")

    monkeypatch.setattr(rag, "build_index", _boom)

    captured = []

    def fake_chat(messages, model):
        captured.append([dict(m) for m in messages])
        return "answer", messages + [{"role": "assistant", "content": "answer"}]

    monkeypatch.setattr(lfa, "query_ollama_chat", fake_chat)

    inputs = iter(["who are you?", "/quit"])
    monkeypatch.setattr("builtins.input", lambda _="": next(inputs))

    lfa.run_chat([], "mistral", embed_model="e", top_k=3, use_rag=True)

    assert captured, "the model was never called"
    sent = captured[0]
    assert sent[0] == {"role": "system", "content": lfa.GENERAL_SYSTEM}
    assert sent[-1] == {"role": "user", "content": "who are you?"}
