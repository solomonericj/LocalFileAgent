# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

LocalFileAgent scans local files/directories and either summarises them or opens an interactive chat about them, using a locally-running Ollama model (default `mistral`). Chat uses **RAG** (retrieval-augmented generation) by default: files are chunked and embedded once via Ollama's embeddings endpoint (default `nomic-embed-text`), then only the most relevant chunks are retrieved per question. All I/O stays on `localhost:11434` — no external network calls.

## Running

```bash
# CLI
python LocalfileAgent.py /path/to/dir              # summarise mode
python LocalfileAgent.py /path/to/dir --chat       # REPL chat mode (RAG by default)
python LocalfileAgent.py /path/to/dir --chat --embed-model nomic-embed-text --top-k 5
python LocalfileAgent.py /path/to/dir --chat --no-rag   # disable RAG, stuff full files
python LocalfileAgent.py /path/to/dir --ext .py .md --recursive
python LocalfileAgent.py /path/to/dir -o out.md    # .md → Markdown output

# GUI (PySide6)
python gui.py
```

**Runtime deps**: `pip install -r requirements.txt` (numpy, used by the RAG vector index — lazily imported, with `--no-rag` as a fallback). **Tests** (pytest + pytest-qt): `pip install -r requirements-dev.txt` then `pytest tests/ -v`. No linter config or build step. Requires Python 3.10+ and a running `ollama serve` with the target chat model pulled, plus an embeddings model for RAG (`ollama pull nomic-embed-text`).

## Architecture

Two entry points share one core module:

- **`LocalfileAgent.py`** — single-file CLI. Sections (delimited by `# ──` banners):
  - Ollama HTTP helpers (`_post`, `query_ollama_generate`, `query_ollama_chat`, `embed_ollama`, `check_ollama_available`) hit `/api/generate`, `/api/chat`, `/api/embed`, `/api/tags` using `urllib` (no `requests` dependency). `embed_ollama(texts, model)` returns one vector per input and raises `ValueError` if the model returns none. `check_ollama_available(model, embed_model=None)` verifies both models are pulled.
  - File collection (`collect_files`, `read_file_safe`) deduplicates paths and dispatches by extension.
  - Binary extractors (`_extract_pdf/_docx/_xlsx/_pptx/_xls/_via_word_powerpoint`) lazy-import their backing package and print a friendly `_missing()` hint when absent. `.doc`/`.ppt` go through `pywin32` COM (Windows + Office only).
  - `run_summarise` and `run_chat` are the two mode entry points called from `main()`. `run_chat` builds a RAG index up front and retrieves per turn unless `--no-rag` is passed (or numpy/embeddings are unavailable, in which case it falls back to full-context stuffing).

- **`rag.py`** — no-Qt retrieval engine (mirrors `session_manager.py`'s plain-module style). `chunk_text(text, path)` splits files into ~900-char overlapping `Chunk`s; `VectorIndex` stores L2-normalised vectors (numpy) with cosine `search`; per-file disk cache under `~/.localfileagent/index/` keyed by `path+size+mtime+embed-model` (`_cache_key`, `save_cache`, `load_cached`). `build_index(files, embed_model, *, embed_fn=None)` reads→chunks→embeds (cache-aware)→indexes; `retrieve(index, query, embed_model, k)` returns the top-k chunks; `build_rag_prompt(chunks, user_text)` composes the per-turn user message. `embed_fn` is injectable so tests run offline; the real `embed_ollama` is imported lazily inside the functions to avoid a circular import.

- **`gui.py`** — PySide6 unified workspace. Widget classes (in order): `FileItemWidget` (single file row with status badge + token count + remove button), `ContextSidebar` (model combo + **embed-model combo** + scrollable file list + drag-drop + token bar), `StreamingChatWorker` (QThread that builds the RAG index on the file-load turn, emits `index_ready`, and injects retrieved chunks into the outgoing message per turn while keeping plain history; emits `token_ready` per chunk), `SessionDialog` (QDialog for browsing/loading/deleting sessions), `MainWindow`. All Ollama calls run in QThread workers — never call network helpers directly from a Qt slot. An indeterminate `QProgressBar` busy indicator with a live elapsed-time readout (`_set_busy`/`_clear_busy`/`_tick_busy`, driven by a `QTimer` + `QElapsedTimer`; `_format_elapsed` renders `3.4s` / `m:ss`) animates in the status bar whenever a worker runs (Indexing… / Thinking… / Responding… / Summarizing…).

- **`session_manager.py`** — no-Qt module. `SessionManager` persists sessions as JSON to `~/.localfileagent/sessions/`. Methods: `save(session) -> Path`, `list() -> list[dict]`, `load(path) -> dict`, `delete(path)`. Sessions store `model`, `embed_model`, `files`, `messages`, and `summaries`.

- **`LocalfileAgent.py`** also exports `stream_ollama_chat(messages, model) -> Iterator[str]` — yields NDJSON token strings from Ollama's streaming `/api/chat` endpoint.

Key invariants worth preserving when editing:
- Text files >`MAX_FILE_BYTES` (200 KB) are skipped; binary-extracted text is capped at `MAX_EXTRACT_CHARS` (400 KB); chat mode loads at most `CONTEXT_FILE_CAP` (20) files.
- RAG keeps **plain user text in persisted history** — only the *outgoing* copy sent to the model gets retrieved context injected (`StreamingChatWorker._compose_api_messages`; CLI `run_chat` appends the composed prompt). The embedding index is built once, cached to disk, and reused per turn.
- In the GUI, `MainWindow._rag_index` caches the `VectorIndex`; it is set via the worker's `index_ready` signal and invalidated on file change, embed-model change, and new session. The worker must receive `rag_index=` on turns 2+ or retrieval won't fire. Loading a saved session leaves the index `None` (don't rebuild on load — it would wipe replayed history); instead the *next* chat turn dispatches a `StreamingChatWorker(..., preserve_history=True)` that rebuilds the index without resetting `messages`, then caches it via `index_ready` for subsequent turns.
- Only one worker runs at a time: `_send_chat`/`_run_single_summarize` bail early if `_operation_in_progress()` (any chat/summarize `QThread` still running), so a live thread is never dropped/GC'd mid-run. `closeEvent` calls `_shutdown_workers()` which `requestInterruption()`s and `wait()`s each running worker; `StreamingChatWorker`'s token loop checks `isInterruptionRequested()` so a close interrupts streaming promptly.
- `_chat_generation` (int) is incremented on `_new_session`; streaming workers capture it at dispatch time and discard replies if it no longer matches — guards against stale responses after session clear.
- `ContextSidebar._items` is a `dict[str, FileItemWidget]`; use `sidebar.get_valid_paths()` (public) to filter out `STATUS_DELETED` entries — do not access `_items` directly from `MainWindow`.
- Adding a new file format means updating `TEXT_EXTENSIONS` or `BINARY_EXTENSIONS` *and* (for binary) adding an `_extract_*` function plus a branch in `_extract_binary`. The GUI picks up new extensions automatically via `SUPPORTED_EXTENSIONS`.
