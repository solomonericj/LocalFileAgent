# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

LocalFileAgent scans local files/directories and either summarises them or opens an interactive chat about them, using a locally-running Ollama model (default `mistral`). Chat is a **general assistant**: it answers from the model's own knowledge and works with **zero files loaded** (`localfileagent --chat` with no paths), only citing file excerpts when relevant. When files are loaded, chat uses **RAG** (retrieval-augmented generation) by default: files are chunked and embedded once via Ollama's embeddings endpoint (default `nomic-embed-text`), then only the most relevant chunks are retrieved per question. Before each answer the assistant runs a non-destructive **clarification check** (`run_clarification_check`): it rates its own confidence (0–100) and, if below 95, asks a focused clarifying question — up to `MAX_CLARIFY_ROUNDS` (3) per turn. All I/O stays on `localhost:11434` — no external network calls.

## Running

```bash
# Install once → installs numpy + PySide6 and puts a `localfileagent` command on PATH
pip install -e .

# CLI
localfileagent /path/to/dir              # summarise mode
localfileagent /path/to/dir --chat       # REPL chat mode (RAG by default)
localfileagent /path/to/dir --chat --embed-model nomic-embed-text --top-k 5
localfileagent /path/to/dir --chat --no-rag   # disable RAG, stuff full files
localfileagent /path/to/dir --ext .py .md --recursive
localfileagent /path/to/dir -o out.md    # .md → Markdown output
localfileagent --gui                     # GUI (PySide6)

# Or run the scripts directly without installing:
python LocalfileAgent.py /path/to/dir    # python LocalfileAgent.py --gui  also works
python gui.py
```

**Packaging**: `pyproject.toml` declares the `localfileagent = "LocalfileAgent:main"` console script and the four top-level `py-modules`; binary parsers are optional extras (`.[parsers]` for all cross-platform, or `.[pdf]`/`.[docx]`/`.[xlsx]`/`.[xls]`/`.[pptx]`/`.[odf]`/`.[7z]`, plus `.[legacy]` = pywin32 on Windows). Notebooks (`.ipynb`), email (`.eml`), and `.zip`/`.tar`/`.tgz` archives parse with the stdlib (no extras). **Runtime deps** (numpy + PySide6) install via `pip install -e .` (or `pip install -r requirements.txt`); numpy is lazily imported, with `--no-rag` as a fallback. **Tests** (pytest + pytest-qt): `pip install -e ".[dev]"` (or `pip install -r requirements-dev.txt`) then `pytest tests/ -v`. No linter config. Requires Python 3.10+ and a running `ollama serve` with the target chat model pulled, plus an embeddings model for RAG (`ollama pull nomic-embed-text`).

## Architecture

Two entry points share one core module:

- **`LocalfileAgent.py`** — single-file CLI. Sections (delimited by `# ──` banners):
  - Ollama HTTP helpers (`_post`, `query_ollama_generate`, `query_ollama_chat`, `embed_ollama`, `check_ollama_available`) hit `/api/generate`, `/api/chat`, `/api/embed`, `/api/tags` using `urllib` (no `requests` dependency), all with a `REQUEST_TIMEOUT` (600 s) ceiling. `embed_ollama(texts, model)` returns one vector per input and raises `ValueError` if the model returns none. `check_ollama_available(model, embed_model=None)` verifies both models are pulled (a bare name matches any pulled tag; a `name:tag` must match exactly) and exits cleanly on an unreachable or malformed `/api/tags`.
  - File collection (`collect_files`, `read_file_safe`) deduplicates paths and dispatches by extension.
  - Binary extractors (`_extract_pdf/_docx/_xlsx/_pptx/_xls/_via_word_powerpoint/_ipynb/_eml/_odf/_archive`) lazy-import their backing package and print a friendly `_missing()` hint when absent. `.doc`/`.ppt` go through `pywin32` COM (Windows + Office only); `.odt`/`.ods`/`.odp` use `odfpy` and `.7z` uses `py7zr`, while `.ipynb`/`.eml`/`.zip`/`.tar`/`.tgz` parse with the stdlib. `_extract_archive` reads only text-extension entries in-memory (no extraction to disk), capped at `MAX_EXTRACT_CHARS`.
  - `run_summarise` and `run_chat` are the two mode entry points called from `main()`; `main()` first calls `_force_utf8_output()` (so emoji output survives a non-UTF-8 console, e.g. a redirected/piped Windows `cp1252` stream) and dispatches to the GUI (`import gui; gui.main()`) when `--gui` is passed. `paths` is required for summarise but optional for `--chat`/`--gui`: `run_chat([])` skips indexing and seeds `GENERAL_SYSTEM` for plain general-assistant chat. With files, `run_chat` builds a RAG index up front (seeding `RAG_SYSTEM`) and retrieves per turn unless `--no-rag` is passed (or numpy/embeddings are unavailable, in which case it falls back to full-context stuffing with `CHAT_SYSTEM_TEMPLATE`). All three modes share one REPL, `_chat_repl(messages, model, *, index, embed_model, top_k)`, where `index=None` means general/full-context (plain history sent as-is) and a `VectorIndex` means RAG.
  - `run_clarification_check(messages, model) -> (confidence, question)` appends `_CLARIFY_PROBE` as a throwaway user turn and parses the reply; it never raises (returns `(95, None)` on any error) so a failed probe never blocks chat. `_chat_repl` runs the clarification loop before answering: the Q&A exchange is appended to the persisted `messages`, but RAG context is injected into the **original** question (tracked by `orig_idx`), not the latest message, so clarification answers stay intact in the API call.

- **`rag.py`** — no-Qt retrieval engine (mirrors `session_manager.py`'s plain-module style). `chunk_text(text, path)` splits files into ~900-char overlapping `Chunk`s; `VectorIndex` stores L2-normalised vectors (numpy) with cosine `search`; per-file disk cache under `~/.localfileagent/index/` keyed by `path+size+mtime+embed-model` (`_cache_key`, `save_cache`, `load_cached`). `build_index(files, embed_model, *, embed_fn=None)` reads→chunks→embeds in batches of `EMBED_BATCH` (64) (cache-aware)→indexes; `retrieve(index, query, embed_model, k)` returns the top-k chunks; `build_rag_prompt(chunks, user_text)` composes the per-turn user message. `embed_fn` is injectable so tests run offline; the real `embed_ollama` is imported lazily inside the functions to avoid a circular import. The disk cache stores chunk metadata as a JSON **string** array (not a pickled object array) and loads with `allow_pickle=False`, so a planted `.npz` cannot execute code.

- **`gui.py`** — PySide6 unified workspace. Widget classes (in order): `FileItemWidget` (single file row with status badge + token count + remove button), `ContextSidebar` (model combo + **embed-model combo** + scrollable file list + drag-drop + token bar), `ClarificationWorker` (QThread wrapping `run_clarification_check` off the UI thread), `StreamingChatWorker` (QThread that builds the RAG index on the file-load turn, emits `index_ready`, and injects retrieved chunks into the outgoing message per turn while keeping plain history; emits `token_ready` per chunk; seeds `GENERAL_SYSTEM` on a fresh no-files turn and takes an `initial_exchange` param so a clarification Q&A precedes the answer), `SessionDialog` (QDialog for browsing/loading/deleting sessions), `MainWindow`. `_send_chat` dispatches a no-files worker (general chat) instead of warning when no files are loaded. Clarification is a `MainWindow` state machine (`_clarifying`, `_clarify_round`, `_clarify_first_turn`, `_clarify_exchange`): clarifying Q&A renders as inline chat bubbles, the input placeholder updates to guide the user, and any probe error falls through silently to the real answer. All Ollama calls run in QThread workers — never call network helpers directly from a Qt slot. An indeterminate `QProgressBar` busy indicator with a live elapsed-time readout (`_set_busy`/`_clear_busy`/`_tick_busy`, driven by a `QTimer` + `QElapsedTimer`; `_format_elapsed` renders `3.4s` / `m:ss`) animates in the status bar whenever a worker runs (Indexing… / Thinking… / Responding… / Summarizing…).

- **`session_manager.py`** — no-Qt module. `SessionManager` persists sessions as JSON to `~/.localfileagent/sessions/`. Methods: `save(session) -> Path`, `list() -> list[dict]`, `load(path) -> dict`, `delete(path)`. Sessions store `model`, `embed_model`, `files`, `messages`, `summaries`, and `created` (the filename derives from `created`). `MainWindow` holds the active session's `created` in `_session_created` (set on first save, on `_load_session`, reset in `_new_session`) and reuses it, so auto-save after each reply **updates one file** rather than creating a new file per turn.

- **`LocalfileAgent.py`** also exports `stream_ollama_chat(messages, model) -> Iterator[str]` — yields NDJSON token strings from Ollama's streaming `/api/chat` endpoint.

Key invariants worth preserving when editing:
- Text files >`MAX_FILE_BYTES` (200 KB) are skipped; binary-extracted text is capped at `MAX_EXTRACT_CHARS` (400 KB); chat mode loads at most `CONTEXT_FILE_CAP` (20) files.
- RAG keeps **plain user text in persisted history** — only the *outgoing* copy sent to the model gets retrieved context injected (`StreamingChatWorker._compose_api_messages`; CLI `run_chat` appends the composed prompt). The embedding index is built once, cached to disk, and reused per turn.
- In the GUI, `MainWindow._rag_index` caches the `VectorIndex`; it is set via the worker's `index_ready` signal and invalidated on file change, embed-model change, and new session. The worker must receive `rag_index=` on turns 2+ or retrieval won't fire. Loading a saved session leaves the index `None` (don't rebuild on load — it would wipe replayed history); instead the *next* chat turn dispatches a `StreamingChatWorker(..., preserve_history=True)` that rebuilds the index without resetting `messages`, then caches it via `index_ready` for subsequent turns.
- Only one worker runs at a time: `_send_chat`/`_run_single_summarize` bail early if `_operation_in_progress()` (any chat/summarize `QThread` still running), so a live thread is never dropped/GC'd mid-run. `closeEvent` calls `_shutdown_workers()` which `requestInterruption()`s and `wait()`s each running worker; `StreamingChatWorker`'s token loop checks `isInterruptionRequested()` so a close interrupts streaming promptly.
- `_chat_generation` (int) is incremented on `_new_session`; streaming workers capture it at dispatch time and discard replies if it no longer matches — guards against stale responses after session clear.
- `ContextSidebar._items` is a `dict[str, FileItemWidget]`; use `sidebar.get_valid_paths()` (public) to filter out `STATUS_DELETED` entries — do not access `_items` directly from `MainWindow`.
- Adding a new file format means updating `TEXT_EXTENSIONS` or `BINARY_EXTENSIONS` *and* (for binary) adding an `_extract_*` function plus a branch in `_extract_binary`. The GUI picks up new extensions automatically via `SUPPORTED_EXTENSIONS`.
