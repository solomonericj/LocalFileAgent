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
