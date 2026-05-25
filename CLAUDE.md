# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

LocalFileAgent scans local files/directories and either summarises them or opens an interactive chat about them, using a locally-running Ollama model (default `mistral`). All I/O stays on `localhost:11434` — no external network calls.

## Running

```bash
# CLI
python LocalfileAgent.py /path/to/dir              # summarise mode
python LocalfileAgent.py /path/to/dir --chat       # REPL chat mode
python LocalfileAgent.py /path/to/dir --ext .py .md --recursive
python LocalfileAgent.py /path/to/dir -o out.md    # .md → Markdown output

# GUI (PySide6)
python gui.py
```

**Tests** (pytest + pytest-qt): `pip install -r requirements-dev.txt` then `pytest tests/ -v`. No linter config or build step. Requires Python 3.10+ and a running `ollama serve` with the target model pulled (`ollama pull <model>`).

## Architecture

Two entry points share one core module:

- **`LocalfileAgent.py`** — single-file CLI. Sections (delimited by `# ──` banners):
  - Ollama HTTP helpers (`_post`, `query_ollama_generate`, `query_ollama_chat`, `check_ollama_available`) hit `/api/generate`, `/api/chat`, `/api/tags` using `urllib` (no `requests` dependency).
  - File collection (`collect_files`, `read_file_safe`) deduplicates paths and dispatches by extension.
  - Binary extractors (`_extract_pdf/_docx/_xlsx/_pptx/_xls/_via_word_powerpoint`) lazy-import their backing package and print a friendly `_missing()` hint when absent. `.doc`/`.ppt` go through `pywin32` COM (Windows + Office only).
  - `run_summarise` and `run_chat` are the two mode entry points called from `main()`.

- **`gui.py`** — PySide6 unified workspace. Widget classes (in order): `FileItemWidget` (single file row with status badge + token count + remove button), `ContextSidebar` (model combo + scrollable file list + drag-drop + token bar), `StreamingChatWorker` (QThread that calls `stream_ollama_chat()` and emits `token_ready` per chunk), `SessionDialog` (QDialog for browsing/loading/deleting sessions), `MainWindow`. All Ollama calls run in QThread workers — never call network helpers directly from a Qt slot.

- **`session_manager.py`** — no-Qt module. `SessionManager` persists sessions as JSON to `~/.localfileagent/sessions/`. Methods: `save(session) -> Path`, `list() -> list[dict]`, `load(path) -> dict`, `delete(path)`.

- **`LocalfileAgent.py`** also exports `stream_ollama_chat(messages, model) -> Iterator[str]` — yields NDJSON token strings from Ollama's streaming `/api/chat` endpoint.

Key invariants worth preserving when editing:
- Text files >`MAX_FILE_BYTES` (200 KB) are skipped; binary-extracted text is capped at `MAX_EXTRACT_CHARS` (400 KB); chat mode loads at most `CONTEXT_FILE_CAP` (20) files.
- `_chat_generation` (int) is incremented on `_new_session`; streaming workers capture it at dispatch time and discard replies if it no longer matches — guards against stale responses after session clear.
- `ContextSidebar._items` is a `dict[str, FileItemWidget]`; use `sidebar.get_valid_paths()` (public) to filter out `STATUS_DELETED` entries — do not access `_items` directly from `MainWindow`.
- Adding a new file format means updating `TEXT_EXTENSIONS` or `BINARY_EXTENSIONS` *and* (for binary) adding an `_extract_*` function plus a branch in `_extract_binary`. The GUI picks up new extensions automatically via `SUPPORTED_EXTENSIONS`.
