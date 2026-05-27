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
EMBED_BATCH   = 64            # chunks embedded per /api/embed request
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
        # Store metadata as a plain unicode-string array (not an object array),
        # so the cache loads with allow_pickle=False — a planted .npz can never
        # execute code via unpickling.
        meta=np.array(json.dumps(meta)),
    )


def load_cached(path: Path, embed_model: str) -> Optional[tuple[list[Chunk], list[list[float]]]]:
    np = _require_numpy()
    cache_file = _cache_path(_cache_key(path, embed_model))
    if not cache_file.exists():
        return None
    try:
        data = np.load(cache_file, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        chunks = [Chunk(m["source"], m["path"], m["text"], m["index"]) for m in meta]
        vectors = data["vectors"].tolist()
        return chunks, vectors
    except (OSError, ValueError, KeyError):
        return None


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
        texts = [c.text for c in chunks]
        vectors: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH):
            vectors.extend(embed_fn(texts[i:i + EMBED_BATCH], embed_model))
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
    return (
        f"Context excerpts from my files that may be relevant:\n{ctx}\n\n"
        f"Question: {user_text}"
    )
