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


def test_cache_loads_without_pickle(tmp_path, monkeypatch):
    """The on-disk cache must not rely on Python pickle — loading a planted
    .npz must never be able to execute arbitrary code (allow_pickle=False)."""
    import rag
    import numpy as np
    monkeypatch.setattr(rag, "CACHE_DIR", tmp_path)
    f = tmp_path / "doc.txt"
    f.write_text("hello")
    save_cache(f, "m", [Chunk("doc.txt", str(f), "hello", 0)], [[0.1, 0.2]])

    cache_file = rag._cache_path(rag._cache_key(f, "m"))
    data = np.load(cache_file, allow_pickle=False)   # must not raise
    assert data["vectors"].shape == (1, 2)
    assert str(data["meta"])                          # readable without unpickling


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
