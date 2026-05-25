# RAG Retrieval for LocalFileAgent — Design

**Date:** 2026-05-25
**Status:** Approved (pending spec review)

## Goal

Replace the current context-stuffing chat flow with embedding-based retrieval
(RAG) to improve **speed** and **reliability** of answers.

Today both the CLI (`run_chat`) and GUI (`StreamingChatWorker`) bake the full
text of every loaded file into the system prompt (`build_file_block` →
`CHAT_SYSTEM_TEMPLATE`), capped at `CONTEXT_FILE_CAP` (20) files and
`MAX_FILE_BYTES` (200 KB) per text file. This means:

- The model re-processes all file tokens on every turn → slow.
- Large dumps cause "lost in the middle" → unreliable answers.

RAG embeds file chunks once, caches them, and feeds only the top-k relevant
chunks per question.

## Scope

- Shared retrieval engine in the core, wired into **both** CLI and GUI.
- 100% local: embeddings via Ollama on `localhost:11434`, no external calls.

## Architecture

### New module: `rag.py` (no-Qt, like `session_manager.py`)

- **`chunk_text(text, path) -> list[Chunk]`** — split file text into ~900-char
  chunks with ~150-char overlap. Each `Chunk` carries its source filename and
  the text. Char-based (not token-based) to stay dependency-free.
- **`VectorIndex`** — holds chunk vectors as a numpy 2-D array plus parallel
  lists of chunk text/metadata.
  - `add(chunks, vectors)`
  - `search(query_vector, k) -> list[(chunk, score)]` via cosine similarity.
- **Disk cache** under `~/.localfileagent/index/` — one `.npz` per source file,
  keyed by `path + size + mtime`. Unchanged files load from cache; only
  new/changed files are re-embedded. This is the primary speed win on repeat use.
- **`build_index(files, embed_model) -> VectorIndex`** — orchestrates: read
  (reusing `read_file_safe`), chunk, check cache, embed cache-misses, persist.
- **`retrieve(index, query, embed_model, k) -> list[Chunk]`**.

### New Ollama helper in `LocalfileAgent.py`

- **`embed_ollama(texts, model) -> list[list[float]]`** — POST to `/api/embed`
  via urllib, same style as `_post`. Default embed model `nomic-embed-text`.
- Extend `check_ollama_available` to also warn if the embed model isn't pulled.

## Retrieval flow (behavioral change)

The chat flips from "bake all files into the system prompt once" to **per-query
retrieval**:

1. **At load time:** build the `VectorIndex` (embed once, cache to disk). The
   system prompt becomes a short instruction *without* file contents — e.g.
   "Answer using the provided context excerpts; cite the source file."
2. **On each user turn:** embed the question, retrieve top-k (default 5) chunks,
   and inject them into the *current* user message:
   `"Context:\n{chunks}\n\nQuestion: {user_text}"`. This keeps conversation
   history clean and gives each turn context relevant to *that* question.
3. Retrieved chunks carry their source filename, so the existing "mention which
   file it came from" behavior still works and answers stay grounded.

Result: the model processes a few hundred relevant tokens per turn instead of
re-chewing every file (speed), and isn't lost in a giant dump (reliability).

## Wiring

### CLI (`run_chat`)

- Replace `build_file_block` stuffing with `build_index` + per-turn `retrieve`.
- New flags:
  - `--embed-model` (default `nomic-embed-text`)
  - `--top-k` (default 5)
  - `--no-rag` (escape hatch → old context-stuffing behavior)

### GUI (`StreamingChatWorker` + sidebar)

- Build the index in a worker thread (embedding is slow — never on the Qt main
  thread). Show an indexing status, then retrieve per message.
- Embed-model picker next to the existing model combo. `top-k` uses the default
  for now.

## Dependency / fallback

- `numpy` is a new runtime dependency. Add a `requirements.txt` and lazy-import
  numpy with the same friendly `_missing()`-style hint the binary extractors use.
- If numpy **or** the embed model is unavailable, fall back to the existing
  context-stuffing path so nothing breaks.

## Testing

`rag.py` is pure-Python/no-Qt, so unit tests run offline:

- **Chunking:** boundaries, overlap, tiny files, empty input.
- **Cosine ranking:** with a fake/stub embedder so tests need no Ollama.
- **Cache:** reuse on unchanged file; invalidation when mtime/size changes.
- Monkeypatch `embed_ollama` to keep the suite offline, like existing tests.

## Invariants preserved

- All I/O stays on `localhost:11434`.
- `read_file_safe`, `collect_files`, size/extract caps unchanged.
- `--no-rag` and the numpy/embed-model fallback guarantee the old behavior
  remains reachable.
