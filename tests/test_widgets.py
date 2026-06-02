import pytest
from gui import FileItemWidget


def test_initial_status_is_pending(qtbot, tmp_path):
    f = tmp_path / "example.py"
    f.write_text("x" * 4000)
    w = FileItemWidget(str(f))
    qtbot.addWidget(w)
    assert w.status() == FileItemWidget.STATUS_PENDING
    assert w.token_estimate() > 0


def test_set_status_loaded_updates_token_count(qtbot, tmp_path):
    f = tmp_path / "example.py"
    f.write_text("hello")
    w = FileItemWidget(str(f))
    qtbot.addWidget(w)
    w.set_status(FileItemWidget.STATUS_LOADED, token_count=2500)
    assert w.status() == FileItemWidget.STATUS_LOADED
    assert w.token_estimate() == 2500


def test_remove_button_emits_signal(qtbot, tmp_path):
    f = tmp_path / "example.py"
    f.write_text("x")
    w = FileItemWidget(str(f))
    qtbot.addWidget(w)
    received = []
    w.remove_requested.connect(received.append)
    w._remove_btn.click()
    assert received == [str(f)]


def test_missing_file_token_estimate_is_zero(qtbot):
    w = FileItemWidget("/nonexistent/path/file.py")
    qtbot.addWidget(w)
    assert w.token_estimate() == 0


def test_sidebar_embed_model_default(qtbot):
    from gui import ContextSidebar
    from LocalfileAgent import DEFAULT_EMBED_MODEL
    sb = ContextSidebar()
    qtbot.addWidget(sb)
    assert sb.embed_model() == DEFAULT_EMBED_MODEL


def test_token_warning_is_rag_aware_when_large(qtbot, tmp_path):
    """A large corpus must not be framed as 'context nearly full — remove files'
    (false under RAG, which retrieves only the most relevant chunks)."""
    from gui import ContextSidebar, FileItemWidget
    sb = ContextSidebar()
    qtbot.addWidget(sb)
    f = tmp_path / "big.txt"
    f.write_text("x")
    sb.add_path(str(f))
    sb.set_file_status(str(f), FileItemWidget.STATUS_LOADED, token_count=40_000)  # > 32k

    warning = sb._token_warning.text().lower()
    assert "rag" in warning
    assert "remove files" not in warning


def test_file_dialog_skips_unsupported_extensions(qtbot, tmp_path, monkeypatch):
    """Files picked via the dialog's 'All Files' filter that the app can't read
    must be skipped, matching drag-and-drop behaviour."""
    import gui
    from gui import ContextSidebar
    sb = ContextSidebar()
    qtbot.addWidget(sb)
    good = tmp_path / "a.py"
    good.write_text("x")
    bad = tmp_path / "b.bin"
    bad.write_text("x")
    monkeypatch.setattr(gui.QFileDialog, "getOpenFileNames",
                        lambda *a, **k: ([str(good), str(bad)], ""))

    sb._add_files_dialog()

    paths = sb.get_paths()
    assert str(good) in paths
    assert str(bad) not in paths
