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
