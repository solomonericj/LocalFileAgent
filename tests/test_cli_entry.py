import pytest

import LocalfileAgent as lfa


def test_gui_flag_launches_gui(monkeypatch):
    """`localfileagent --gui` dispatches to the GUI and skips CLI work."""
    import gui
    called = []
    monkeypatch.setattr(gui, "main", lambda: called.append(True))
    monkeypatch.setattr(lfa.sys, "argv", ["localfileagent", "--gui"])
    monkeypatch.setattr(lfa, "check_ollama_available",
                        lambda *a, **k: pytest.fail("must not check Ollama in GUI mode"))
    monkeypatch.setattr(lfa, "collect_files",
                        lambda *a, **k: pytest.fail("must not collect files in GUI mode"))

    lfa.main()
    assert called == [True]


def test_requires_paths_without_gui(monkeypatch):
    """The CLI still requires at least one path when not launching the GUI."""
    monkeypatch.setattr(lfa.sys, "argv", ["localfileagent"])
    with pytest.raises(SystemExit):
        lfa.main()
