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
