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
