import LocalfileAgent as lfa


def test_skipped_first_file_does_not_crash(tmp_path, capsys):
    """A skipped first file (empty/too large) must not raise — `elapsed` was
    referenced before assignment when content was None (regression)."""
    empty = tmp_path / "empty.txt"
    empty.write_text("")          # size 0 -> read_file_safe returns None -> skipped

    lfa.run_summarise([empty], "mistral", None)   # must not raise

    out = capsys.readouterr().out
    assert "(empty file)" in out


def test_skipped_file_after_success_has_no_stale_timing(tmp_path, monkeypatch, capsys):
    """A skipped file following a summarised one must not reuse the previous
    file's elapsed time on its progress line."""
    good = tmp_path / "good.txt"
    good.write_text("real content")
    empty = tmp_path / "empty.txt"
    empty.write_text("")

    monkeypatch.setattr(lfa, "query_ollama_generate", lambda *a, **k: "a summary")
    # Deterministic clock: the good file's summarisation takes exactly 5.0s.
    times = iter([100.0, 105.0])
    monkeypatch.setattr(lfa.time, "monotonic", lambda: next(times))

    lfa.run_summarise([good, empty], "mistral", None)

    out = capsys.readouterr().out
    # "(5.0s)" belongs only to the good file's progress line — never the skip.
    assert out.count("(5.0s)") == 1
