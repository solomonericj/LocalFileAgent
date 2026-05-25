# RAG Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace chat context-stuffing with embedding-based retrieval (RAG) so the model receives only the relevant chunks per question — faster and more reliable answers.

**Architecture:** A new no-Qt `rag.py` module handles chunking, a numpy `VectorIndex`, a per-file disk cache under `~/.localfileagent/index/`, and `build_index`/`retrieve`. Embeddings come from a new `embed_ollama` helper hitting Ollama's local `/api/embed`. Both the CLI (`run_chat`) and the GUI (`StreamingChatWorker`) build the index once, then retrieve top-k chunks per turn and inject them into the current user message. A `--no-rag` flag and numpy/embed-model fallback preserve the old behavior.

**Tech Stack:** Python 3.10+, numpy, urllib (no `requests`), PySide6, pytest / pytest-qt, Ollama (`nomic-embed-text` for embeddings).

---

## File Structure

- **Create `rag.py`** — `Chunk` dataclass, `chunk_text`, `VectorIndex`, disk cache (`_cache_dir`, `_cache_key`, `save_cache`, `load_cached`), `build_index`, `retrieve`, `build_rag_prompt`. No Qt, no direct network (embedding injected as `embed_fn`).
- **Modify `LocalfileAgent.py`** — add `DEFAULT_EMBED_MODEL`, `OLLAMA_EMBED`, `RAG_SYSTEM`, `embed_ollama`; extend `check_ollama_available`; rewrite `run_chat` to use RAG with `--no-rag` fallback; add `--embed-model` / `--top-k` flags.
- **Modify `gui.py`** — `StreamingChatWorker` gains RAG support (build index on load, retrieve per turn, emit `index_ready`); `ContextSidebar` gains an embed-model combo; `MainWindow` caches the index and resets it on new/load session.
- **Create `requirements.txt`** — runtime deps (numpy).
- **Create `tests/test_rag.py`** — chunking, cosine ranking, cache reuse/invalidation, build_index/retrieve with a fake embedder.
- **Create `tests/test_embed.py`** — `embed_ollama` over a mocked urllib response.
- **Modify `tests/test_streaming_worker.py`** — add a RAG-path worker test.

---

## Task 1: Runtime requirements file

**Files:**
- Create: `requirements.txt`

- [ ] **Step 1: Create `requirements.txt`**

```
# Runtime dependencies for LocalFileAgent.
# Binary-format extractors (pypdf, python-docx, openpyxl, python-pptx, xlrd,
# pywin32) are optional and lazy-imported — install only what you need.
numpy>=1.24
```

- [ ] **Step 2: Install it**

Run: `pip install -r requirements.txt`
Expected: numpy installed (or "already satisfied").

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "build: add runtime requirements.txt with numpy for RAG"
```

---

## Task 2: `embed_ollama` helper

**Files:**
- Modify: `LocalfileAgent.py` (Configuration block ~line 29-37; add helper after `stream_ollama_chat` ~line 154)
- Test: `tests/test_embed.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_embed.py`:

```python
import io
import json
import socket
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from LocalfileAgent import embed_ollama


def _fake_json_response(payload: dict):
    buf = io.BytesIO(json.dumps(payload).encode())
    buf.__enter__ = lambda s: s
    buf.__exit__ = MagicMock(return_value=False)
    return buf


def test_returns_embeddings_list():
    resp = _fake_json_response({"embeddings": [[0.1, 0.2], [0.3, 0.4]]})
    with patch("urllib.request.urlopen", return_value=resp):
        vecs = embed_ollama(["a", "b"], "nomic-embed-text")
    assert vecs == [[0.1, 0.2], [0.3, 0.4]]


def test_sends_input_field():
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data)
        return _fake_json_response({"embeddings": [[1.0]]})

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        embed_ollama(["hello"], "nomic-embed-text")
    assert captured["body"]["input"] == ["hello"]
    assert captured["body"]["model"] == "nomic-embed-text"


def test_raises_connection_error():
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
        with pytest.raises(ConnectionError):
            embed_ollama(["x"], "nomic-embed-text")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_embed.py -v`
Expected: FAIL with `ImportError: cannot import name 'embed_ollama'`.

- [ ] **Step 3: Add config constants**

In `LocalfileAgent.py`, in the Configuration block (after `DEFAULT_MODEL = "mistral"` and the URL constants), add:

```python
DEFAULT_EMBED_MODEL = "nomic-embed-text"
OLLAMA_EMBED        = "http://localhost:11434/api/embed"
```

Then add a new system prompt constant after `CHAT_SYSTEM_TEMPLATE`:

```python
RAG_SYSTEM = (
    "You are a helpful assistant. Answer the user's question using only the "
    "context excerpts provided in their message. Each excerpt is labelled with "
    "its source file in square brackets. Cite the source file when you reference "
    "information. If the excerpts do not contain the answer, say so plainly."
)
```

- [ ] **Step 4: Implement `embed_ollama`**

After `stream_ollama_chat` (around line 154), add:

```python
def embed_ollama(texts: list[str], model: str) -> list[list[float]]:
    """Embed a batch of strings via Ollama's /api/embed. Returns one vector per input."""
    result = _post(OLLAMA_EMBED, {"model": model, "input": texts})
    return result.get("embeddings", [])
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_embed.py -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Commit**

```bash
git add LocalfileAgent.py tests/test_embed.py
git commit -m "feat: add embed_ollama helper and RAG constants"
```

---

## Task 3: `chunk_text` and `Chunk`

**Files:**
- Create: `rag.py`
- Test: `tests/test_rag.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_rag.py`:

```python
from pathlib import Path

from rag import Chunk, chunk_text


def test_short_text_single_chunk():
    chunks = chunk_text("hello world", Path("a.txt"))
    assert len(chunks) == 1
    assert chunks[0].text == "hello world"
    assert chunks[0].source == "a.txt"
    assert chunks[0].index == 0


def test_empty_text_no_chunks():
    assert chunk_text("", Path("a.txt")) == []
    assert chunk_text("   ", Path("a.txt")) == []


def test_long_text_splits_with_overlap():
    text = "x" * 2000
    chunks = chunk_text(text, Path("big.txt"), size=900, overlap=150)
    assert len(chunks) >= 2
    # Each chunk no larger than `size`
    assert all(len(c.text) <= 900 for c in chunks)
    # Overlap: end of chunk 0 reappears at start of chunk 1
    assert chunks[0].text[-150:] == chunks[1].text[:150]
    # Ordinals increment
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_source_is_filename_path_is_full():
    p = Path("dir/sub/report.md")
    chunks = chunk_text("content", p)
    assert chunks[0].source == "report.md"
    assert chunks[0].path == str(p)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rag.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag'`.

- [ ] **Step 3: Implement `Chunk` and `chunk_text`**

Create `rag.py`:

```python
#!/usr/bin/env python3
"""
rag.py — Embedding-based retrieval for LocalFileAgent.

No Qt, no direct network: embedding is injected as `embed_fn` so the module is
unit-testable offline. Vectors are stored in-memory (numpy) and cached per file
under ~/.localfileagent/index/.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

CHUNK_SIZE    = 900
CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 5
CACHE_DIR     = Path.home() / ".localfileagent" / "index"


@dataclass
class Chunk:
    source: str   # filename, e.g. "report.md"
    path: str     # full path string
    text: str
    index: int    # ordinal within the source file


def chunk_text(text: str, path: Path, *, size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    """Split text into overlapping char windows, each tagged with its source."""
    if not text or not text.strip():
        return []
    chunks: list[Chunk] = []
    step = max(1, size - overlap)
    start = 0
    ordinal = 0
    while start < len(text):
        piece = text[start:start + size]
        if piece.strip():
            chunks.append(Chunk(source=path.name, path=str(path),
                                text=piece, index=ordinal))
            ordinal += 1
        if start + size >= len(text):
            break
        start += step
    return chunks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rag.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add rag.py tests/test_rag.py
git commit -m "feat: add Chunk dataclass and chunk_text"
```

---

## Task 4: `VectorIndex`

**Files:**
- Modify: `rag.py`
- Test: `tests/test_rag.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rag.py`:

```python
from rag import VectorIndex


def _chunk(text, i=0):
    return Chunk(source="f.txt", path="f.txt", text=text, index=i)


def test_index_len_and_search_ranks_by_cosine():
    idx = VectorIndex()
    idx.add([_chunk("apple", 0), _chunk("banana", 1), _chunk("cat", 2)],
            [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]])
    assert len(idx) == 3
    results = idx.search([1.0, 0.0], k=2)
    assert len(results) == 2
    # Closest to [1,0] is "apple" (exact), then "cat"
    assert results[0][0].text == "apple"
    assert results[1][0].text == "cat"
    # Scores are descending floats
    assert results[0][1] >= results[1][1]


def test_search_k_larger_than_corpus():
    idx = VectorIndex()
    idx.add([_chunk("only", 0)], [[1.0, 1.0]])
    results = idx.search([1.0, 1.0], k=5)
    assert len(results) == 1


def test_empty_index_search_returns_empty():
    assert VectorIndex().search([1.0, 0.0], k=3) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rag.py -v`
Expected: FAIL with `ImportError: cannot import name 'VectorIndex'`.

- [ ] **Step 3: Implement `VectorIndex`**

Add to `rag.py` (after `chunk_text`):

```python
def _require_numpy():
    try:
        import numpy as np  # noqa: F401
        return np
    except ImportError as exc:
        raise ImportError(
            "RAG needs numpy — install it:  pip install numpy\n"
            "(or run chat with --no-rag to use plain context-stuffing)"
        ) from exc


class VectorIndex:
    """In-memory store of L2-normalized chunk vectors with cosine search."""

    def __init__(self):
        self.chunks: list[Chunk] = []
        self.vectors = None   # numpy array (N, d) or None

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        np = _require_numpy()
        v = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        v = v / np.maximum(norms, 1e-8)
        self.vectors = v if self.vectors is None else np.vstack([self.vectors, v])
        self.chunks.extend(chunks)

    def search(self, query_vector: list[float], k: int) -> list[tuple[Chunk, float]]:
        if self.vectors is None or len(self.chunks) == 0:
            return []
        np = _require_numpy()
        q = np.asarray(query_vector, dtype=np.float32)
        q = q / max(float(np.linalg.norm(q)), 1e-8)
        sims = self.vectors @ q
        k = min(k, len(self.chunks))
        top = np.argsort(-sims)[:k]
        return [(self.chunks[i], float(sims[i])) for i in top]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rag.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rag.py tests/test_rag.py
git commit -m "feat: add VectorIndex with cosine search"
```

---

## Task 5: Disk cache

**Files:**
- Modify: `rag.py`
- Test: `tests/test_rag.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rag.py`:

```python
from rag import _cache_key, save_cache, load_cached


def test_cache_round_trip(tmp_path, monkeypatch):
    import rag
    monkeypatch.setattr(rag, "CACHE_DIR", tmp_path)
    f = tmp_path / "doc.txt"
    f.write_text("hello")
    chunks = [Chunk(source="doc.txt", path=str(f), text="hello", index=0)]
    save_cache(f, "nomic-embed-text", chunks, [[0.5, 0.5]])

    loaded = load_cached(f, "nomic-embed-text")
    assert loaded is not None
    cached_chunks, cached_vecs = loaded
    assert cached_chunks[0].text == "hello"
    assert cached_chunks[0].source == "doc.txt"
    assert cached_vecs[0] == [0.5, 0.5]


def test_cache_miss_when_file_changes(tmp_path, monkeypatch):
    import rag
    monkeypatch.setattr(rag, "CACHE_DIR", tmp_path)
    f = tmp_path / "doc.txt"
    f.write_text("hello")
    save_cache(f, "nomic-embed-text", [Chunk("doc.txt", str(f), "hello", 0)], [[0.1]])
    # Rewrite with different size -> key changes -> miss
    f.write_text("hello world, now longer")
    assert load_cached(f, "nomic-embed-text") is None


def test_cache_miss_on_different_model(tmp_path, monkeypatch):
    import rag
    monkeypatch.setattr(rag, "CACHE_DIR", tmp_path)
    f = tmp_path / "doc.txt"
    f.write_text("hello")
    save_cache(f, "nomic-embed-text", [Chunk("doc.txt", str(f), "hello", 0)], [[0.1]])
    assert load_cached(f, "other-model") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rag.py -v`
Expected: FAIL with `ImportError: cannot import name '_cache_key'`.

- [ ] **Step 3: Implement the cache**

Add to `rag.py` (after `VectorIndex`):

```python
def _cache_key(path: Path, embed_model: str) -> str:
    """Stable key from resolved path + size + mtime + embed model."""
    try:
        st = path.stat()
        sig = f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}|{embed_model}"
    except OSError:
        sig = f"{path}|missing|{embed_model}"
    return hashlib.sha1(sig.encode()).hexdigest()


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.npz"


def save_cache(path: Path, embed_model: str, chunks: list[Chunk],
               vectors: list[list[float]]) -> None:
    np = _require_numpy()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    meta = [{"source": c.source, "path": c.path, "text": c.text, "index": c.index}
            for c in chunks]
    np.savez(
        _cache_path(_cache_key(path, embed_model)),
        vectors=np.asarray(vectors, dtype=np.float32),
        meta=np.array(json.dumps(meta), dtype=object),
    )


def load_cached(path: Path, embed_model: str) -> Optional[tuple[list[Chunk], list[list[float]]]]:
    np = _require_numpy()
    cache_file = _cache_path(_cache_key(path, embed_model))
    if not cache_file.exists():
        return None
    try:
        data = np.load(cache_file, allow_pickle=True)
        meta = json.loads(str(data["meta"]))
        chunks = [Chunk(m["source"], m["path"], m["text"], m["index"]) for m in meta]
        vectors = data["vectors"].tolist()
        return chunks, vectors
    except (OSError, ValueError, KeyError):
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rag.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rag.py tests/test_rag.py
git commit -m "feat: add per-file embedding disk cache"
```

---

## Task 6: `build_index`, `retrieve`, `build_rag_prompt`

**Files:**
- Modify: `rag.py`
- Test: `tests/test_rag.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rag.py`:

```python
from rag import build_index, retrieve, build_rag_prompt


def _fake_embedder(call_log=None):
    """Deterministic embedder: vector = [len(text), count('a')]."""
    def embed(texts, model):
        if call_log is not None:
            call_log.extend(texts)
        return [[float(len(t)), float(t.count("a"))] for t in texts]
    return embed


def test_build_index_embeds_and_caches(tmp_path, monkeypatch):
    import rag
    monkeypatch.setattr(rag, "CACHE_DIR", tmp_path)
    f = tmp_path / "doc.txt"
    f.write_text("banana apple")
    log = []
    idx = build_index([f], "nomic-embed-text", embed_fn=_fake_embedder(log))
    assert len(idx) >= 1
    assert log  # embedder was called on first build

    # Second build hits cache — embedder NOT called again
    log2 = []
    idx2 = build_index([f], "nomic-embed-text", embed_fn=_fake_embedder(log2))
    assert len(idx2) == len(idx)
    assert log2 == []


def test_retrieve_returns_relevant_chunks(tmp_path, monkeypatch):
    import rag
    monkeypatch.setattr(rag, "CACHE_DIR", tmp_path)
    f = tmp_path / "doc.txt"
    f.write_text("aaaa")
    idx = build_index([f], "m", embed_fn=_fake_embedder())
    chunks = retrieve(idx, "aaaa", "m", k=1, embed_fn=_fake_embedder())
    assert len(chunks) == 1
    assert isinstance(chunks[0], Chunk)


def test_build_rag_prompt_formats_context():
    chunks = [Chunk("a.txt", "a.txt", "first", 0), Chunk("b.txt", "b.txt", "second", 0)]
    prompt = build_rag_prompt(chunks, "What is X?")
    assert "[a.txt]" in prompt and "first" in prompt
    assert "[b.txt]" in prompt and "second" in prompt
    assert prompt.rstrip().endswith("What is X?")


def test_build_rag_prompt_no_chunks_passthrough():
    assert build_rag_prompt([], "just this") == "just this"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rag.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_index'`.

- [ ] **Step 3: Implement the orchestration**

Add to `rag.py`. The `read_file_safe` import is local (inside the function) to avoid a circular import with `LocalfileAgent`:

```python
def build_index(files, embed_model: str, *,
                embed_fn: Optional[Callable] = None) -> VectorIndex:
    """Read, chunk, embed (cache-aware), and index the given files."""
    from LocalfileAgent import read_file_safe, embed_ollama
    embed_fn = embed_fn or embed_ollama

    index = VectorIndex()
    for path in files:
        path = Path(path)
        cached = load_cached(path, embed_model)
        if cached is not None:
            chunks, vectors = cached
            index.add(chunks, vectors)
            continue

        content = read_file_safe(path)
        if content is None:
            continue
        chunks = chunk_text(content, path)
        if not chunks:
            continue
        vectors = embed_fn([c.text for c in chunks], embed_model)
        save_cache(path, embed_model, chunks, vectors)
        index.add(chunks, vectors)
    return index


def retrieve(index: VectorIndex, query: str, embed_model: str, k: int = DEFAULT_TOP_K, *,
             embed_fn: Optional[Callable] = None) -> list[Chunk]:
    """Embed the query and return the top-k most similar chunks."""
    if len(index) == 0:
        return []
    from LocalfileAgent import embed_ollama
    embed_fn = embed_fn or embed_ollama
    query_vec = embed_fn([query], embed_model)[0]
    return [chunk for chunk, _score in index.search(query_vec, k)]


def build_rag_prompt(chunks: list[Chunk], user_text: str) -> str:
    """Compose the per-turn user message: context excerpts + question."""
    if not chunks:
        return user_text
    ctx = "\n\n".join(f"[{c.source}]\n{c.text}" for c in chunks)
    return f"Context excerpts:\n{ctx}\n\nQuestion: {user_text}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rag.py -v`
Expected: PASS (all rag tests).

- [ ] **Step 5: Commit**

```bash
git add rag.py tests/test_rag.py
git commit -m "feat: add build_index, retrieve, build_rag_prompt"
```

---

## Task 7: Extend `check_ollama_available` for the embed model

**Files:**
- Modify: `LocalfileAgent.py:156-173`
- Test: `tests/test_embed.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_embed.py`:

```python
from LocalfileAgent import check_ollama_available


def _tags_response(names):
    return _fake_json_response({"models": [{"name": n} for n in names]})


def test_check_passes_when_both_models_present():
    with patch("urllib.request.urlopen", return_value=_tags_response(["mistral", "nomic-embed-text"])):
        check_ollama_available("mistral", embed_model="nomic-embed-text")  # no SystemExit


def test_check_exits_when_embed_model_missing():
    with patch("urllib.request.urlopen", return_value=_tags_response(["mistral"])):
        with pytest.raises(SystemExit):
            check_ollama_available("mistral", embed_model="nomic-embed-text")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_embed.py -v`
Expected: FAIL — `check_ollama_available()` got an unexpected keyword `embed_model`.

- [ ] **Step 3: Update `check_ollama_available`**

Replace the function (lines ~156-173) with:

```python
def check_ollama_available(model: str, embed_model: str | None = None) -> None:
    """Verify Ollama is reachable and the requested model(s) are pulled."""
    try:
        req = urllib.request.Request(OLLAMA_TAGS)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        available = [m["name"].split(":")[0] for m in data.get("models", [])]
        wanted = [model] + ([embed_model] if embed_model else [])
        missing = [m for m in wanted if m.split(":")[0] not in available]
        if missing:
            print(
                f"⚠  Model(s) not found locally: {', '.join(missing)}\n"
                f"   Available: {', '.join(available) or 'none'}\n"
                f"   Pull with:  {'; '.join(f'ollama pull {m}' for m in missing)}\n",
                file=sys.stderr,
            )
            sys.exit(1)
    except urllib.error.URLError:
        print("✗  Ollama is not running.  Start it with:  ollama serve", file=sys.stderr)
        sys.exit(1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_embed.py -v`
Expected: PASS.

- [ ] **Step 5: Run full suite to confirm no regressions**

Run: `pytest tests/ -v`
Expected: PASS (existing tests still green; `test_check_ollama` style tests, if any, unaffected since `embed_model` defaults to None).

- [ ] **Step 6: Commit**

```bash
git add LocalfileAgent.py tests/test_embed.py
git commit -m "feat: check embed model availability in check_ollama_available"
```

---

## Task 8: Wire RAG into the CLI `run_chat`

**Files:**
- Modify: `LocalfileAgent.py` — `run_chat` (~434-514), `build_parser` (~519-556), `main` (~559-578)

- [ ] **Step 1: Add CLI flags in `build_parser`**

After the `--model` argument, add:

```python
    p.add_argument(
        "--embed-model",
        default=DEFAULT_EMBED_MODEL,
        help=f"Embedding model for RAG (default: {DEFAULT_EMBED_MODEL}).",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="(chat) Number of context chunks to retrieve per question (default: 5).",
    )
    p.add_argument(
        "--no-rag",
        action="store_true",
        help="(chat) Disable RAG; load full file contents into context instead.",
    )
```

- [ ] **Step 2: Update `main` to pass the new args**

Change the `check_ollama_available` call and the `run_chat` call:

```python
    if not args.no_check:
        embed_model = None if args.no_rag else args.embed_model
        check_ollama_available(args.model, embed_model)

    files = collect_files(args.paths, extensions, args.recursive)
    if not files:
        print("No matching files found.", file=sys.stderr)
        sys.exit(0)

    if args.chat:
        run_chat(files, args.model, embed_model=args.embed_model,
                 top_k=args.top_k, use_rag=not args.no_rag)
    else:
        run_summarise(files, args.model, args.output)
```

- [ ] **Step 3: Rewrite `run_chat` signature and body**

Replace `run_chat` (starting at `def run_chat(files: list[Path], model: str) -> None:`) with the version below. The non-RAG path keeps the old context-stuffing behavior; the RAG path builds an index then retrieves per turn.

```python
def run_chat(files: list[Path], model: str, *, embed_model: str = DEFAULT_EMBED_MODEL,
             top_k: int = 5, use_rag: bool = True) -> None:
    if len(files) > CONTEXT_FILE_CAP:
        print(
            f"⚠  {len(files)} files found — only the first {CONTEXT_FILE_CAP} will be "
            f"used (limit).\n   Use --ext or --recursive to narrow the selection.\n"
        )
        files = files[:CONTEXT_FILE_CAP]

    index = None
    if use_rag:
        try:
            print(f"📂  Indexing {len(files)} file(s) with '{embed_model}'…", end=" ", flush=True)
            from rag import build_index, retrieve, build_rag_prompt
            index = build_index(files, embed_model)
            if len(index) == 0:
                print("\n⚠  No content indexed — falling back to full-context mode.")
                index = None
            else:
                print(f"done  ({len(index)} chunks)")
        except ImportError as exc:
            print(f"\n⚠  {exc}\n   Falling back to full-context mode.")
            index = None
        except ConnectionError as exc:
            print(f"\n✗  {exc}", file=sys.stderr)
            sys.exit(1)

    if index is not None:
        messages: list[dict] = [{"role": "system", "content": RAG_SYSTEM}]
        print(
            f"\n💬  Chat mode (RAG) — ask anything about the indexed files.\n"
            f"    Commands:  /clear  /quit\n{'─'*60}"
        )
    else:
        # Fallback: original full-context behavior.
        print(f"📂  Loading {len(files)} file(s) into context…", end=" ", flush=True)
        file_block, skipped = build_file_block(files)
        if not file_block.strip():
            print("\n✗  No readable content found in the selected files.", file=sys.stderr)
            sys.exit(1)
        loaded = len(files) - len(skipped)
        print(f"done  ({loaded} loaded, {len(skipped)} skipped)")
        if skipped:
            print(f"   Skipped (empty/too large): {', '.join(skipped)}")
        messages = [{"role": "system",
                     "content": CHAT_SYSTEM_TEMPLATE.format(n=loaded, file_block=file_block)}]
        print(
            f"\n💬  Chat mode — ask anything about the loaded files.\n"
            f"    Commands:  /clear  /quit\n{'─'*60}"
        )

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nBye!")
            break
        if not user_input:
            continue
        if user_input.lower() in ("/quit", "/exit", "/q"):
            print("Bye!")
            break
        if user_input.lower() == "/clear":
            messages = [messages[0]]
            print("🗑  Conversation history cleared.")
            continue

        if index is not None:
            from rag import retrieve, build_rag_prompt
            try:
                chunks = retrieve(index, user_input, embed_model, top_k)
            except ConnectionError as exc:
                print(f"\n✗  {exc}", file=sys.stderr)
                sys.exit(1)
            messages.append({"role": "user", "content": build_rag_prompt(chunks, user_input)})
        else:
            messages.append({"role": "user", "content": user_input})

        print(f"\n{model}: ", end="", flush=True)
        try:
            reply, messages = query_ollama_chat(messages, model)
        except ConnectionError as exc:
            print(f"\n✗  {exc}", file=sys.stderr)
            sys.exit(1)
        print(reply)
```

Note: the `/list` and `/help` commands are dropped from the loop above to keep it focused; if you want them, re-add the original blocks — but `/list` referenced `skipped`/`files` which only exist in the fallback path, so guard accordingly. (YAGNI: leave them out unless asked.)

- [ ] **Step 4: Manual smoke test (requires Ollama + models pulled)**

Run: `python LocalfileAgent.py . --ext .py --chat` then ask "what does rag.py do?"
Expected: Indexing line prints a chunk count, then a grounded answer citing `rag.py`.

Run the fallback: `python LocalfileAgent.py . --ext .py --chat --no-rag`
Expected: Old "Loading N file(s)" behavior.

- [ ] **Step 5: Run full suite**

Run: `pytest tests/ -v`
Expected: PASS (no regressions; CLI loop is not unit-tested but imports must resolve).

- [ ] **Step 6: Commit**

```bash
git add LocalfileAgent.py
git commit -m "feat: wire RAG into CLI run_chat with --no-rag fallback"
```

---

## Task 9: RAG support in `StreamingChatWorker`

**Files:**
- Modify: `gui.py` — imports (~28-30), `StreamingChatWorker` (~89-152)
- Test: `tests/test_streaming_worker.py`

- [ ] **Step 1: Write the failing test**

First inspect `tests/test_streaming_worker.py` to match its fixtures/imports, then append a test mirroring its style. The test injects a fake index and monkeypatches `retrieve` + `stream_ollama_chat` so no Ollama/numpy is needed:

```python
def test_worker_rag_path_composes_prompt(qtbot, monkeypatch):
    import gui
    from rag import Chunk

    fake_chunks = [Chunk("a.txt", "a.txt", "ground truth", 0)]
    monkeypatch.setattr(gui, "retrieve", lambda *a, **k: fake_chunks)
    monkeypatch.setattr(gui, "stream_ollama_chat", lambda messages, model: iter(["ok"]))

    sentinel_index = object()
    worker = gui.StreamingChatWorker(
        [{"role": "system", "content": gui.RAG_SYSTEM},
         {"role": "user", "content": "what is the truth?"}],
        "mistral",
        rag_index=sentinel_index, embed_model="nomic-embed-text",
        top_k=5, user_text="what is the truth?",
    )

    captured = {}
    orig = gui.stream_ollama_chat
    def capture(messages, model):
        captured["messages"] = messages
        return iter(["ok"])
    monkeypatch.setattr(gui, "stream_ollama_chat", capture)

    with qtbot.waitSignal(worker.finished, timeout=3000) as blocker:
        worker.run()  # run synchronously in-test
    # The outgoing last user message contains the retrieved context...
    assert "ground truth" in captured["messages"][-1]["content"]
    # ...but the persisted history keeps the plain user text.
    updated = blocker.args[0]
    assert updated[-2]["content"] == "what is the truth?"
    assert updated[-1] == {"role": "assistant", "content": "ok"}
```

(Adjust import style — `from LocalfileAgent import retrieve` vs `gui.retrieve` — to match how Step 3 imports `retrieve` into `gui`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_streaming_worker.py::test_worker_rag_path_composes_prompt -v`
Expected: FAIL — `StreamingChatWorker.__init__` got an unexpected keyword `rag_index`.

- [ ] **Step 3: Update imports and the worker**

In `gui.py`, extend the import block (lines 28-30) to add the RAG names:

```python
from LocalfileAgent import (
    SUMMARISE_SYSTEM, CHAT_SYSTEM_TEMPLATE, CONTEXT_FILE_CAP,
    RAG_SYSTEM, DEFAULT_EMBED_MODEL,
    read_file_safe, collect_files, build_file_block,
    query_ollama_generate, query_ollama_chat, stream_ollama_chat,
)
from rag import build_index, retrieve, build_rag_prompt
```

Add a new signal and extend `StreamingChatWorker.__init__`:

```python
class StreamingChatWorker(QThread):
    token_ready  = Signal(str)
    finished     = Signal(list)    # updated_messages
    context_info = Signal(str)
    file_status  = Signal(str, str, int)
    index_ready  = Signal(object)  # emits the built VectorIndex
    error        = Signal(str)

    def __init__(self, messages: list, model: str, *,
                 files_to_load: list = None, user_text: str = None,
                 rag_index=None, embed_model: str = DEFAULT_EMBED_MODEL,
                 top_k: int = 5, use_rag: bool = True):
        super().__init__()
        self.messages = list(messages)
        self.model = model
        self.files_to_load = files_to_load
        self.user_text = user_text
        self.rag_index = rag_index
        self.embed_model = embed_model
        self.top_k = top_k
        self.use_rag = use_rag
```

Replace `run()` with the version below. Key idea: persisted history holds the **plain** user text; only the outgoing copy gets the retrieved context injected.

```python
    def run(self):
        try:
            if self.files_to_load is not None:
                if self.use_rag:
                    index = self._build_rag_index()
                    if index is not None:
                        self.rag_index = index
                        self.index_ready.emit(index)
                        self.messages = [
                            {"role": "system", "content": RAG_SYSTEM},
                            {"role": "user", "content": self.user_text},
                        ]
                    else:
                        self._load_full_context()   # fallback sets self.messages
                else:
                    self._load_full_context()

            api_messages = self._compose_api_messages()

            accumulated = ""
            for token in stream_ollama_chat(api_messages, self.model):
                accumulated += token
                self.token_ready.emit(token)

            updated = self.messages + [{"role": "assistant", "content": accumulated}]
            self.finished.emit(updated)

        except (ConnectionError, TimeoutError) as exc:
            self.error.emit(str(exc))
        except Exception as exc:
            self.error.emit(f"Unexpected error: {exc}")

    def _build_rag_index(self):
        """Build the vector index; return None to signal fallback."""
        try:
            index = build_index(self.files_to_load, self.embed_model)
        except ImportError:
            return None
        if len(index) == 0:
            return None
        self.context_info.emit(f"Indexed {len(index)} chunk(s) from {len(self.files_to_load)} file(s)")
        for path in self.files_to_load:
            self.file_status.emit(str(path), FileItemWidget.STATUS_LOADED, 0)
        return index

    def _load_full_context(self):
        """Original context-stuffing path; sets self.messages in place."""
        parts, skipped_names = [], []
        for path in self.files_to_load:
            content = read_file_safe(path)
            if content is None:
                skipped_names.append(path.name)
                self.file_status.emit(str(path), FileItemWidget.STATUS_SKIPPED, 0)
            else:
                parts.append(f"### {path.name}\nPath: {path}\n\n{content}")
                self.file_status.emit(str(path), FileItemWidget.STATUS_LOADED, len(content) // 4)
        file_block = "\n\n---\n\n".join(parts)
        if not file_block.strip():
            raise RuntimeError("No readable content found in the selected files.")
        loaded = len(self.files_to_load) - len(skipped_names)
        info = f"Context ready: {loaded} file(s) loaded"
        if skipped_names:
            info += f", {len(skipped_names)} skipped ({', '.join(skipped_names)})"
        self.context_info.emit(info)
        self.messages = [
            {"role": "system", "content": CHAT_SYSTEM_TEMPLATE.format(n=loaded, file_block=file_block)},
            {"role": "user", "content": self.user_text},
        ]

    def _compose_api_messages(self):
        """Return messages for the API call; inject RAG context into the last user turn."""
        if self.rag_index is not None and self.use_rag and self.user_text is not None:
            chunks = retrieve(self.rag_index, self.user_text, self.embed_model, self.top_k)
            composed = build_rag_prompt(chunks, self.user_text)
            return self.messages[:-1] + [{"role": "user", "content": composed}]
        return list(self.messages)
```

Note: `_load_full_context` now raises `RuntimeError` instead of emitting `error` directly; the existing `except Exception` in `run()` converts it to an `error` signal with the message text. Verify the message still reads sensibly in the UI.

- [ ] **Step 4: Run the worker test**

Run: `pytest tests/test_streaming_worker.py -v`
Expected: PASS (new test + existing worker tests). Fix import-style mismatches if the existing tests patch `gui.stream_ollama_chat` differently.

- [ ] **Step 5: Commit**

```bash
git add gui.py tests/test_streaming_worker.py
git commit -m "feat: RAG support in StreamingChatWorker with full-context fallback"
```

---

## Task 10: Wire the index + embed-model picker into `MainWindow`

**Files:**
- Modify: `gui.py` — `ContextSidebar._build_ui` (~287-291) + new `embed_model()` method; `MainWindow.__init__` (~612 area), `_send_message` (~830-872), `_on_chat_reply`/new/load session (~878-958)

- [ ] **Step 1: Add an embed-model combo to `ContextSidebar`**

After the model combo block (line ~291), add:

```python
        embed_lbl = QLabel("EMBED MODEL")
        embed_lbl.setStyleSheet(
            "font-size: 9px; color: #64748b; letter-spacing: 1px; font-weight: 600;"
        )
        layout.addWidget(embed_lbl)

        self._embed_combo = QComboBox()
        self._embed_combo.setEditable(True)
        self._embed_combo.addItem(DEFAULT_EMBED_MODEL)
        layout.addWidget(self._embed_combo)
```

Add `DEFAULT_EMBED_MODEL` to the `from LocalfileAgent import (...)` block if not already there (it was added in Task 9). Add a public accessor near `model()` (line ~373):

```python
    def embed_model(self) -> str:
        return self._embed_combo.currentText().strip()
```

- [ ] **Step 2: Initialize index state in `MainWindow.__init__`**

Where `self._chat_messages: list = []` is set (~612), add:

```python
        self._rag_index = None
```

- [ ] **Step 3: Update `_send_message` to build/reuse the index**

Replace the `if not self._chat_files_loaded:` / `else:` dispatch block (~839-864) with logic that triggers a build whenever there is no index yet (covers fresh and loaded sessions), and otherwise reuses the cached index:

```python
        embed_model = self._sidebar.embed_model() or DEFAULT_EMBED_MODEL
        needs_build = self._rag_index is None and not self._chat_files_loaded

        if needs_build:
            valid_paths = self._sidebar.get_valid_paths()
            if not valid_paths:
                QMessageBox.warning(self, "No Files", "No accessible files. Check for deleted files in the sidebar.")
                self._set_chat_input_enabled(True)
                return
            files = [Path(p) for p in valid_paths[:CONTEXT_FILE_CAP]]
            if len(valid_paths) > CONTEXT_FILE_CAP:
                self._append_system(
                    f"⚠  Only first {CONTEXT_FILE_CAP} of {len(valid_paths)} files used."
                )
            self._append_system(f"Indexing {len(files)} file(s)…")
            self._append_chat("You", user_text, "#3b82f6")
            self._chat_worker = StreamingChatWorker(
                [], model, files_to_load=files, user_text=user_text,
                embed_model=embed_model,
            )
            self._chat_worker.context_info.connect(self._on_context_info)
            self._chat_worker.index_ready.connect(self._on_index_ready)
            self._chat_worker.file_status.connect(
                lambda p, s, t: self._sidebar.set_file_status(p, s, t)
            )
        else:
            self._chat_messages.append({"role": "user", "content": user_text})
            self._append_chat("You", user_text, "#3b82f6")
            self._chat_worker = StreamingChatWorker(
                list(self._chat_messages), model,
                rag_index=self._rag_index, embed_model=embed_model,
                user_text=user_text,
            )
```

- [ ] **Step 4: Add the `_on_index_ready` slot**

Next to `_on_context_info` (~874), add:

```python
    def _on_index_ready(self, index):
        self._rag_index = index
```

- [ ] **Step 5: Reset index on new/load session**

In `_new_session` (~918), where `self._chat_messages = []` is set, add:

```python
        self._rag_index = None
```

In `_load_session` (~930), after `self._new_session()` already cleared it, the index stays `None` — so the next message rebuilds it from the restored sidebar files. Confirm `_chat_files_loaded` is **not** forced True for RAG: change the tail of `_load_session` (~953) so a restored session with files but no index will rebuild:

```python
        # RAG index is rebuilt lazily on the next message; only mark files
        # loaded when there is conversation history to continue.
        if self._chat_messages:
            self._chat_files_loaded = True
```

(Leave this line as-is if already present; the key point is `_rag_index` is None after load, so `needs_build` in Step 3 — gated on `not self._chat_files_loaded` — will be False for a session with history. To force a rebuild for RAG, change `needs_build` in Step 3 to also trigger when `self._rag_index is None and self.use_rag`. Since the GUI always uses RAG, simplest is: `needs_build = self._rag_index is None`. Update Step 3 accordingly and drop the `_chat_files_loaded` term.)

  Final `needs_build` line for Step 3:

```python
        needs_build = self._rag_index is None
```

- [ ] **Step 6: Persist embed model in sessions (optional, low-risk)**

In `_auto_save` (~895), add `"embed_model": self._sidebar.embed_model(),` to the session dict, and in `_load_session` restore it:

```python
        embed_name = session.get("embed_model", DEFAULT_EMBED_MODEL)
        eidx = self._sidebar._embed_combo.findText(embed_name)
        if eidx >= 0:
            self._sidebar._embed_combo.setCurrentIndex(eidx)
        else:
            self._sidebar._embed_combo.setEditText(embed_name)
```

- [ ] **Step 7: Manual smoke test (requires Ollama + PySide6)**

Run: `python gui.py`
- Add a few files, pick `nomic-embed-text` as embed model, send a question.
- Expected: "Indexing N file(s)…" then a grounded answer; second question is fast (no re-index).
- Start a new session, load it back, ask a question → index rebuilds and answers.

- [ ] **Step 8: Run full suite**

Run: `pytest tests/ -v`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add gui.py
git commit -m "feat: cache RAG index in MainWindow and add embed-model picker"
```

---

## Task 11: Documentation + final verification

**Files:**
- Modify: `CLAUDE.md`, `README.md` (if present)

- [ ] **Step 1: Update `CLAUDE.md`**

- Add `rag.py` to the Architecture module list with a one-line description (chunking, `VectorIndex`, disk cache, `build_index`/`retrieve`/`build_rag_prompt`).
- Note the new `embed_ollama` export and `/api/embed` usage in the `LocalfileAgent.py` section.
- Add to the Running section: `--embed-model`, `--top-k`, `--no-rag` flags.
- Add an invariant: "Chat uses RAG by default — index built once per file (cached under `~/.localfileagent/index/` keyed by path+size+mtime+embed-model), top-k chunks retrieved per turn and injected into the user message; falls back to full-context stuffing if numpy or the embed model is unavailable, or with `--no-rag`."

- [ ] **Step 2: Update `README.md`** (if it exists)

Mirror the CLAUDE.md changes: document RAG mode, the embed model requirement (`ollama pull nomic-embed-text`), and the new flags.

- [ ] **Step 3: Run the full suite one last time**

Run: `pytest tests/ -v`
Expected: PASS (all tests).

- [ ] **Step 4: Verify imports resolve cleanly**

Run: `python -c "import rag, LocalfileAgent, gui; print('imports ok')"`
Expected: `imports ok` (no circular-import error).

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: document RAG mode, embed model, and new chat flags"
```

---

## Self-Review Notes

- **Spec coverage:** rag.py module (Tasks 3-6) ✓; embed_ollama + /api/embed (Task 2) ✓; check embed model (Task 7) ✓; per-query retrieval injected into user message (Tasks 6, 8, 9) ✓; disk cache keyed by path+size+mtime+model (Task 5) ✓; CLI flags + fallback (Task 8) ✓; GUI worker + index caching + embed picker (Tasks 9, 10) ✓; numpy lazy-import fallback (Tasks 4, 8, 9) ✓; offline tests via injected/monkeypatched embedder (Tasks 2-7, 9) ✓; docs (Task 11) ✓.
- **Type consistency:** `Chunk(source, path, text, index)` used identically across tasks; `build_index(files, embed_model, *, embed_fn)`, `retrieve(index, query, embed_model, k, *, embed_fn)`, `build_rag_prompt(chunks, user_text)` signatures consistent between rag.py, CLI, and GUI.
- **Known follow-up:** the GUI always uses RAG (no `--no-rag` UI toggle) — acceptable per spec ("top-k uses the default for now"); a toggle can be added later if needed.
