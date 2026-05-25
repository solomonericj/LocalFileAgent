# LocalFileAgent — Unified Workspace Redesign

**Date:** 2026-05-24
**Status:** Approved for implementation

---

## Overview

Replace the current two-tab (Summarize / Chat) GUI with a single unified workspace where context management, chat, and on-demand summarization coexist in one view. The redesign addresses the two primary pain points: no feedback during loading, and no persistent visibility of what files are in context.

**Goals:**
- Always show which files are loaded and whether they're usable
- Stream model responses so the UI never feels frozen
- Remove the mental overhead of switching tabs
- Preserve all existing CLI behaviour (no changes to `LocalfileAgent.py` except streaming support)

**Out of scope:** Dark mode, CLI changes, remote/non-Ollama models.

---

## Layout

```
┌─────────────────────────────────────────────┐
│ ⬡ LocalFileAgent              [🕐 Sessions] │
├──────────────┬──────────────────────────────┤
│ Model        │                              │
│ [mistral  ▾] │   You: What does _post do?   │
│              │                              │
│ Context      │   mistral: It's a low-level  │
│ ✓ agent.py   │   HTTP helper… ▌             │
│   ~4.2k tok  │                              │
│ ✓ gui.py     │                              │
│   ~6.1k tok  ├──────────────────────────────┤
│ ⚠ report.pdf │ Summarize: [agent.py▶][gui▶] [📋]
│   too large  ├──────────────────────────────┤
│              │ Ask a question…       [Send] │
│ [Drop here]  │                              │
│ [+Files][+Folder]                           │
│ ──────────── │                              │
│ ~10.3k / 32k │                              │
│ [████░░░░░░] │                              │
└──────────────┴──────────────────────────────┘
```

---

## Components

### 1. Top Bar

A single slim bar spanning the full window width.

- **Left:** App title "⬡ LocalFileAgent" (non-interactive label)
- **Right:** "🕐 Sessions" button — opens the session history panel (see §Sessions)
- No model selector here; no mode toggles

### 2. Context Sidebar (left column, fixed width ~230 px)

**Model selector** (top of sidebar):
- `QComboBox`, editable, auto-populated from Ollama on startup
- Changing the model invalidates the current loaded context (sets `_chat_files_loaded = False`, shows a status message)

**File list** (scrollable):
Each entry shows:
- Status badge: `✓` green (loaded), `⚠` amber (skipped — too large / unreadable), `↻` grey (pending load)
- Filename (truncated with tooltip showing full path)
- Token estimate (approx `len(content) // 4`, shown in muted text below filename)
- `✕` remove button (right-aligned, visible on hover)

File entries support drag-and-drop reordering within the list (cosmetic only — load order doesn't affect context). This is internal list reordering (`QListWidget` with `InternalMove`), distinct from the drop-zone below which accepts files dragged from the OS file manager.

**Add controls:**
- Drag-and-drop target covering the lower portion of the sidebar (dashed border, `dragEnterEvent` / `dropEvent` on the sidebar widget)
- `[+ Files]` and `[+ Folder]` buttons (existing logic, unchanged)
- Existing recursive checkbox and extension filter field move here from the top options bar (the options bar is removed)

**Token bar** (pinned to sidebar bottom):
- Label: "~{n}k / {max}k tokens" — `n` is sum of per-file estimates, `max` is a configurable constant (default 32k)
- `QProgressBar` showing percentage; turns amber > 75%, red > 95%
- Clicking the bar does nothing (display only)

### 3. Chat Panel (right column, fills remaining width)

**Chat history** (`QTextEdit`, read-only):
- Existing HTML rendering kept (user messages blue, model messages green)
- Streaming: responses render word-by-word as chunks arrive (see §Streaming)
- A blinking block cursor `▌` appended while streaming; removed when done

**Summarize strip** (between chat history and input):
- One `QPushButton` per loaded file: label = filename, click triggers per-file summarization
- A "📋 Copy All" button that copies all summaries in the current session to the clipboard as Markdown
- Buttons are disabled while a chat response or summarization is in progress
- Strip is hidden if no files are loaded

**Chat input row:**
- `QLineEdit` + `[Send]` button (existing layout, unchanged)
- Send is disabled while a response streams

### 4. Streaming Responses

Ollama's `/api/chat` endpoint supports `"stream": true`, returning NDJSON with one token per line. Replace the current single-response `_post` call with a streaming variant.

**New function in `LocalfileAgent.py`:**
```python
def stream_ollama_chat(messages: list[dict], model: str) -> Iterator[str]:
    """Yields token strings one at a time. Raises StopIteration when done."""
```

Uses `urllib.request.urlopen` in streaming mode, reading the response line-by-line. Each line is a JSON object; the token string is at `obj["message"]["content"]`. The generator stops when `obj["done"]` is `True`. The caller accumulates tokens to reconstruct the full assistant message for `updated_messages`.

**New worker in `gui.py`:** `StreamingChatWorker(QThread)`:
- Signals: `token_ready = Signal(str)`, `finished = Signal(list)`, `error = Signal(str)`
- `token_ready` fires for each chunk → main thread appends to the current assistant bubble
- `finished` fires with the full `updated_messages` list when done

`ChatWorker` is replaced by `StreamingChatWorker`. The file-loading path in the current `ChatWorker` (added in the bug-fix session) is preserved in `StreamingChatWorker`.

### 5. Session History

Sessions persist chat history and loaded-file paths to disk so they survive restarts.

**Storage:** `~/.localfileagent/sessions/` — one JSON file per session named `YYYY-MM-DD_HH-MM-SS.json`.

**Schema:**
```json
{
  "created": "2026-05-24T14:30:00",
  "model": "mistral",
  "files": ["/abs/path/to/file.py"],
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "summaries": {
    "/abs/path/to/file.py": "Summary text..."
  }
}
```

**Session panel** (opened by "🕐 Sessions" button):
- `QDialog` listing past sessions sorted newest-first
- Each row: date/time, model name, file count, first user message (truncated)
- "Load" button restores the session: repopulates the file list, restores messages, re-validates files still exist
- "Delete" button removes the JSON file

**Auto-save:** The current session is written to disk after every assistant reply and after every summarization. No explicit save button needed.

**New session:** Clicking "🕐 Sessions → New Session" clears everything (equivalent to current "Clear History" but also clears files).

---

## Data Flow

### First chat message (files not yet loaded)

1. User types message, presses Send
2. `_send_chat()` checks `_chat_files_loaded`; if False, passes `files_to_load` to `StreamingChatWorker`
3. Worker thread: `build_file_block()` runs off-thread (existing fix preserved)
4. Worker emits `context_info` → sidebar updates file status badges to `✓`/`⚠`
5. Worker calls `stream_ollama_chat()`, emitting `token_ready` for each chunk
6. Main thread appends each chunk to the assistant bubble in real time
7. Worker emits `finished(updated_messages)` → `_chat_messages` updated, session auto-saved

### Subsequent messages

Steps 3–4 skipped; worker uses existing `_chat_messages` snapshot.

### Per-file summarization

1. User clicks a file's `[▶]` button in the summarize strip
2. `SummarizeWorker` runs for that single file (existing worker, scoped to one file)
3. Result appended to the chat history as a system-style message and stored in `session.summaries`
4. "📋 Copy All" copies all entries in `session.summaries` as `## filename\n\nsummary\n\n`

---

## File / Module Changes

| File | Change |
|------|--------|
| `LocalfileAgent.py` | Add `stream_ollama_chat()` generator function |
| `gui.py` | Full rewrite of `MainWindow`; replace `ChatWorker` with `StreamingChatWorker`; import and use `SessionManager` |
| `session_manager.py` *(new)* | `SessionManager` class: load, save, list, delete sessions; owns the `~/.localfileagent/sessions/` directory |

Keeping `session_manager.py` separate from `gui.py` makes it testable without a Qt environment and keeps `gui.py` from growing further.

---

## Error Handling

| Scenario | Behaviour |
|----------|-----------|
| File removed after being added to sidebar | Badge changes to `⚠ deleted`; excluded from next context load |
| Stream interrupted mid-response | Partial text shown + "(response interrupted)" appended; Send re-enabled |
| Session file corrupt on load | Skipped with a warning in the sessions panel; other sessions unaffected |
| Token bar exceeds 95% | Amber warning under bar: "Context nearly full — remove files or start a new session" |
| Model changed while context loaded | Banner: "Model changed — context will reload on next message" |

---

## Out of Scope (explicitly deferred)

- Dark mode
- CLI changes beyond `stream_ollama_chat()`
- Prompt template customisation
- Watch mode / auto-reload
- Semantic search across summaries
- CLI shell completions
