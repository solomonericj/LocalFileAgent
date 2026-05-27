# Design: General-Assistant Chat (answer any question, not just file content)

**Date:** 2026-05-26
**Status:** Approved for planning
**Scope:** Chat mode only (CLI REPL + PySide6 GUI). Summarise mode unchanged.

## Goal

Today chat is fenced to imported files: the system prompt instructs the model to
answer *"using only the context excerpts"* and to refuse (*"say so plainly"*) when
the files don't cover the question. We want chat to behave as a **full general
assistant** that:

- Answers any question from its own knowledge.
- Still uses and **cites** loaded files when they're relevant (RAG keeps running).
- Works with **zero files loaded** — usable as a plain chatbot.
- Applies to **both** the CLI REPL and the GUI workspace.

Non-goals: changing summarise mode, the `--no-rag` full-context fallback's plumbing,
session persistence, the RAG index/caching, or any network helper.

## Why this is small

The files-only behavior is enforced by **three prompt strings** and **two entry
guards**, not by architecture. Retrieval, streaming, sessions, and history are
behavior-agnostic. The work is: (A) reword the prompts so outside knowledge is
allowed, and (B) relax the two guards that require files to exist.

## Changes

### A. Prompt rewording (the "answer anything" part)

1. **`RAG_SYSTEM`** (`LocalfileAgent.py:78`) — replace the "only the context / say so
   plainly" instruction with a general-assistant instruction:

   > You are a helpful assistant. The user's message may include context excerpts
   > pulled from their local files, each labelled with its source file in square
   > brackets. When an excerpt is relevant, use it and cite the source file. When the
   > excerpts don't cover the question, answer normally from your own knowledge —
   > do not refuse or say the answer isn't in the files.

2. **`CHAT_SYSTEM_TEMPLATE`** (`LocalfileAgent.py:70`) — same softening for the
   full-context fallback path: keep the file block, but tell the model it may also
   answer from general knowledge and should cite files when it draws on them.

3. **`build_rag_prompt`** (`rag.py:183`) — change the per-turn framing so excerpts
   read as optional context rather than the sole basis for the answer. Current:
   `"Context excerpts:\n{ctx}\n\nQuestion: {user_text}"`. New framing (when chunks
   exist): label the block as context that *may* be relevant, then the question.
   The empty-chunks branch (returns `user_text` unchanged) stays as-is.

4. **New constant `GENERAL_SYSTEM`** (`LocalfileAgent.py`) — the system prompt used
   when chat starts with **no files**: a plain, friendly general-assistant prompt
   with no file references. Exported alongside the others so the GUI can import it.

### B. Zero-files support (the structural wrinkle)

#### CLI (`LocalfileAgent.py`)

- `main()` (`:698`) currently calls `parser.error(...)` when `paths` is empty. Relax
  so that **`--chat` with no paths** is allowed and dispatches `run_chat([], ...)`.
  Summarise with no paths still errors (summarise needs files).
- `main()` (`:711`) currently `sys.exit(0)`s when `collect_files` returns nothing.
  For chat mode, treat "no files" as a valid general-chat session rather than exiting.
  (Summarise keeps the early exit.)
- `run_chat(files, ...)` (`:508`) — add an empty-`files` branch **before** the
  indexing block: skip `build_index`, seed `messages = [{"role":"system",
  "content": GENERAL_SYSTEM}]`, print the no-files hint (below), and enter the
  existing REPL loop. With `index is None` and no full-context block, the loop's
  `else: api_messages = messages` path (`:590-591`) already does the right thing.

#### GUI (`gui.py`)

- `_send_chat` (`:1056-1061`) — when `not self._chat_files_loaded` **and** there are
  no valid paths, instead of warning and bailing, dispatch a no-files first turn:
  a `StreamingChatWorker([], model, user_text=user_text, embed_model=...)` with no
  `files_to_load` and no `rag_index`.
- `StreamingChatWorker.run` (`gui.py:122`) — today `files_to_load is None` skips
  straight to `_compose_api_messages()`, which returns `list(self.messages)` — and
  on a first turn `self.messages` is empty, so no system prompt is sent. Add a
  guard: when there's no index, no files, and `messages` is empty, seed
  `self.messages = [{"role":"system","content": GENERAL_SYSTEM}, {"role":"user",
  "content": self.user_text}]` before composing. Mark the session as "loaded" so
  subsequent turns append normally (emit `context_info` with the no-files hint, which
  already sets `_chat_files_loaded = True` at `gui.py:1123`).
- Token-bar / sidebar: with zero files the token bar simply reflects an empty file
  set (already RAG-aware per recent work) — no special handling needed.

## Resolved open questions

1. **Empty-file UX:** Show a brief one-line hint when chat starts with no files —
   CLI prints it, GUI emits it as a system line. Wording:
   *"No files loaded — general chat mode. Add files anytime for grounded answers."*
2. **CLI invocation:** Use `localfileagent --chat` with no path (no new flag).

## Data flow (after change)

```
question
  ├─ files loaded?  yes → retrieve top-k chunks → build_rag_prompt (excerpts + Q)
  │                        → model uses+cites excerpts, or answers generally if irrelevant
  └─ files loaded?  no  → GENERAL_SYSTEM + plain question → model answers generally
```

History/persistence unchanged: persisted history keeps plain user text; only the
outgoing copy gets excerpts injected (when an index exists).

## Error handling

- No new failure modes. Zero-files chat makes no network call until the user sends a
  message; embeddings/index code is simply not invoked.
- `--no-rag` and the numpy-missing / nothing-indexable fallbacks keep working: they
  route through `CHAT_SYSTEM_TEMPLATE`, which now also permits general answers.
- Existing connection/timeout handling in `run_chat` and the worker is untouched.

## Testing

- **Prompt constants:** assert `RAG_SYSTEM` / `CHAT_SYSTEM_TEMPLATE` no longer contain
  the "only"/"say so plainly" refusal language and do mention citing sources.
- **`build_rag_prompt`:** with chunks → excerpts framed as optional context + question;
  with no chunks → returns `user_text` unchanged (regression guard).
- **`run_chat([], ...)`:** with a stubbed chat function and simulated input, verify it
  seeds `GENERAL_SYSTEM`, never calls `build_index`, prints the hint, and the
  outgoing messages contain the system prompt + user turn (no file block).
- **GUI worker (pytest-qt):** `StreamingChatWorker` with no files/index on an empty
  history seeds `GENERAL_SYSTEM` + user turn; with an index present, retrieval still
  fires (existing behavior intact).
- **GUI `_send_chat`:** sending with zero valid paths dispatches a no-files worker
  rather than showing the "No Files" warning.

## Files touched

- `LocalfileAgent.py` — prompts (`RAG_SYSTEM`, `CHAT_SYSTEM_TEMPLATE`), new
  `GENERAL_SYSTEM`, `main()` guards, `run_chat` empty-files branch + hint.
- `rag.py` — `build_rag_prompt` framing.
- `gui.py` — import `GENERAL_SYSTEM`, `_send_chat` no-files dispatch, worker seed.
- `tests/` — additions per above.

Estimated ~40–70 lines of source plus tests. No new dependencies.
