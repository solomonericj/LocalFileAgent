#!/usr/bin/env python3
"""
gui.py â€” PySide6 graphical interface for LocalfileAgent.

Run with:  python gui.py
"""

import html
import json
import sys
import urllib.request
import urllib.error
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QListWidget, QListWidgetItem, QPushButton, QLabel,
    QComboBox, QCheckBox, QLineEdit, QTextEdit, QProgressBar,
    QFileDialog, QSplitter, QGroupBox, QStatusBar, QMessageBox,
    QFrame, QScrollArea, QDialog,
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QTextCursor

sys.path.insert(0, str(Path(__file__).parent))
from LocalfileAgent import (
    SUPPORTED_EXTENSIONS, DEFAULT_MODEL, OLLAMA_TAGS,
    SUMMARISE_SYSTEM, CHAT_SYSTEM_TEMPLATE, CONTEXT_FILE_CAP,
    read_file_safe, collect_files, build_file_block,
    query_ollama_generate, query_ollama_chat, stream_ollama_chat,
)
from session_manager import SessionManager

# â”€â”€ Workers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class ModelFetchWorker(QThread):
    models_ready = Signal(list)
    error = Signal(str)

    def run(self):
        try:
            req = urllib.request.Request(OLLAMA_TAGS)
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]
            self.models_ready.emit(models)
        except Exception as exc:
            self.error.emit(str(exc))


class SummarizeWorker(QThread):
    progress = Signal(int, int, str)   # current, total, filename
    file_done = Signal(str, str)       # path_str, summary
    finished = Signal()
    error = Signal(str)

    def __init__(self, files: list, model: str):
        super().__init__()
        self.files = files
        self.model = model

    def run(self):
        for i, path in enumerate(self.files, 1):
            self.progress.emit(i, len(self.files), path.name)
            content = read_file_safe(path)
            if content is None:
                try:
                    size = path.stat().st_size
                except OSError:
                    summary = "(skipped â€” file no longer accessible)"
                else:
                    summary = "(empty file)" if size == 0 else f"(skipped â€” too large: {size:,} bytes)"
            else:
                try:
                    summary = query_ollama_generate(
                        f"File: {path.name}\n\n{content}",
                        SUMMARISE_SYSTEM,
                        self.model,
                    )
                except TimeoutError as exc:
                    summary = f"(skipped â€” timeout: {exc})"
                except ConnectionError as exc:
                    self.error.emit(str(exc))
                    return
            self.file_done.emit(str(path), summary)
        self.finished.emit()


class StreamingChatWorker(QThread):
    token_ready  = Signal(str)
    finished     = Signal(list)    # updated_messages
    context_info = Signal(str)     # human-readable summary after file load
    file_status  = Signal(str, str, int)  # path_str, status constant, token_count
    error        = Signal(str)

    def __init__(self, messages: list, model: str, *,
                 files_to_load: list = None, user_text: str = None):
        super().__init__()
        self.messages = list(messages)         # snapshot â€” never share the live list
        self.model = model
        self.files_to_load = files_to_load     # list[Path] or None
        self.user_text = user_text

    def run(self):
        try:
            if self.files_to_load is not None:
                parts: list[str] = []
                skipped_names: list[str] = []

                for path in self.files_to_load:
                    content = read_file_safe(path)
                    if content is None:
                        skipped_names.append(path.name)
                        self.file_status.emit(
                            str(path), FileItemWidget.STATUS_SKIPPED, 0
                        )
                    else:
                        token_count = len(content) // 4
                        parts.append(f"### {path.name}\nPath: {path}\n\n{content}")
                        self.file_status.emit(
                            str(path), FileItemWidget.STATUS_LOADED, token_count
                        )

                file_block = "\n\n---\n\n".join(parts)
                if not file_block.strip():
                    self.error.emit("No readable content found in the selected files.")
                    return

                loaded = len(self.files_to_load) - len(skipped_names)
                info = f"Context ready: {loaded} file(s) loaded"
                if skipped_names:
                    info += f", {len(skipped_names)} skipped ({', '.join(skipped_names)})"
                self.context_info.emit(info)

                system_prompt = CHAT_SYSTEM_TEMPLATE.format(n=loaded, file_block=file_block)
                self.messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": self.user_text},
                ]

            accumulated = ""
            for token in stream_ollama_chat(self.messages, self.model):
                accumulated += token
                self.token_ready.emit(token)

            updated = self.messages + [{"role": "assistant", "content": accumulated}]
            self.finished.emit(updated)

        except (ConnectionError, TimeoutError) as exc:
            self.error.emit(str(exc))
        except Exception as exc:
            self.error.emit(f"Unexpected error: {exc}")


# â”€â”€ FileItemWidget â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class FileItemWidget(QWidget):
    remove_requested = Signal(str)   # emits path_str

    STATUS_PENDING = "pending"
    STATUS_LOADED  = "loaded"
    STATUS_SKIPPED = "skipped"
    STATUS_DELETED = "deleted"

    _STATUS_STYLES = {
        STATUS_PENDING: ("â†»", "#64748b"),
        STATUS_LOADED:  ("âœ“", "#22c55e"),
        STATUS_SKIPPED: ("âš ", "#f59e0b"),
        STATUS_DELETED: ("âœ•", "#ef4444"),
    }

    def __init__(self, path_str: str, parent=None):
        super().__init__(parent)
        self._path_str = path_str
        self._path = Path(path_str)
        self._status = self.STATUS_PENDING
        self._token_estimate = self._compute_estimate()
        self._build_ui()

    def _compute_estimate(self) -> int:
        try:
            return self._path.stat().st_size // 4
        except OSError:
            return 0

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(1)

        row = QHBoxLayout()
        row.setSpacing(6)

        self._badge = QLabel("â†»")
        self._badge.setFixedWidth(14)
        row.addWidget(self._badge)

        self._name_label = QLabel()
        self._name_label.setToolTip(self._path_str)
        self._name_label.setStyleSheet("font-size: 11px; color: #1e293b;")
        row.addWidget(self._name_label, 1)

        self._remove_btn = QPushButton("âœ•")
        self._remove_btn.setFixedSize(16, 16)
        self._remove_btn.setFlat(True)
        self._remove_btn.setStyleSheet("color: #94a3b8; font-size: 9px; border: none;")
        self._remove_btn.clicked.connect(lambda: self.remove_requested.emit(self._path_str))
        row.addWidget(self._remove_btn)
        layout.addLayout(row)

        self._token_label = QLabel()
        layout.addWidget(self._token_label)

        self._refresh_display()

    def _refresh_display(self):
        icon, color = self._STATUS_STYLES[self._status]
        self._badge.setText(icon)
        self._badge.setStyleSheet(f"color: {color}; font-size: 11px;")

        fm = self._name_label.fontMetrics()
        self._name_label.setText(
            fm.elidedText(self._path.name, Qt.TextElideMode.ElideMiddle, 140)
        )

        if self._status == self.STATUS_SKIPPED:
            self._token_label.setText("skipped â€” too large")
            self._token_label.setStyleSheet(
                "font-size: 9px; color: #f59e0b; padding-left: 20px;"
            )
        elif self._status == self.STATUS_DELETED:
            self._token_label.setText("file deleted")
            self._token_label.setStyleSheet(
                "font-size: 9px; color: #ef4444; padding-left: 20px;"
            )
        else:
            k = self._token_estimate // 1000
            self._token_label.setText(f"~{k or '<1'}k tokens")
            self._token_label.setStyleSheet(
                "font-size: 9px; color: #94a3b8; padding-left: 20px;"
            )

    def set_status(self, status: str, token_count: int = None):
        self._status = status
        if token_count is not None:
            self._token_estimate = token_count
        self._refresh_display()

    def status(self) -> str:
        return self._status

    def token_estimate(self) -> int:
        return self._token_estimate

    def path_str(self) -> str:
        return self._path_str


# â”€â”€ ContextSidebar â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class ContextSidebar(QWidget):
    files_changed = Signal()        # emitted on any add/remove
    model_changed = Signal(str)     # emitted when model combo changes

    MAX_CONTEXT_TOKENS = 32_000

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items: dict[str, FileItemWidget] = {}   # path_str -> widget
        self.setAcceptDrops(True)
        self._build_ui()

    # â”€â”€ construction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 10, 8, 10)
        layout.setSpacing(8)

        # Model selector
        model_lbl = QLabel("MODEL")
        model_lbl.setStyleSheet(
            "font-size: 9px; color: #64748b; letter-spacing: 1px; font-weight: 600;"
        )
        layout.addWidget(model_lbl)

        self._model_combo = QComboBox()
        self._model_combo.setEditable(True)
        self._model_combo.addItem(DEFAULT_MODEL)
        self._model_combo.currentTextChanged.connect(self.model_changed.emit)
        layout.addWidget(self._model_combo)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #e2e8f0;")
        layout.addWidget(sep)

        self._ctx_label = QLabel("CONTEXT â€” 0 FILES")
        self._ctx_label.setStyleSheet(
            "font-size: 9px; color: #64748b; letter-spacing: 1px; font-weight: 600;"
        )
        layout.addWidget(self._ctx_label)

        # Scrollable file list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._file_container = QWidget()
        self._file_layout = QVBoxLayout(self._file_container)
        self._file_layout.setContentsMargins(0, 0, 0, 0)
        self._file_layout.setSpacing(4)
        self._file_layout.addStretch()
        scroll.setWidget(self._file_container)
        layout.addWidget(scroll, 1)

        # Drop zone
        self._drop_label = QLabel("Drop files here")
        self._drop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._drop_label.setStyleSheet(
            "border: 2px dashed #cbd5e1; border-radius: 4px;"
            "padding: 8px; color: #94a3b8; font-size: 10px;"
        )
        layout.addWidget(self._drop_label)

        # + Files / + Folder buttons
        btn_row = QHBoxLayout()
        add_files_btn = QPushButton("+ Files")
        add_files_btn.clicked.connect(self._add_files_dialog)
        btn_row.addWidget(add_files_btn)
        add_folder_btn = QPushButton("+ Folder")
        add_folder_btn.clicked.connect(self._add_folder_dialog)
        btn_row.addWidget(add_folder_btn)
        layout.addLayout(btn_row)

        # Recursive + extension filter
        opt_row = QHBoxLayout()
        self._recursive_check = QCheckBox("Recursive")
        opt_row.addWidget(self._recursive_check)
        self._ext_input = QLineEdit()
        self._ext_input.setPlaceholderText(".py .md  (blank=all)")
        self._ext_input.setStyleSheet("font-size: 10px;")
        opt_row.addWidget(self._ext_input)
        layout.addLayout(opt_row)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("color: #e2e8f0;")
        layout.addWidget(sep2)

        # Token bar
        token_row = QHBoxLayout()
        self._token_label = QLabel("0 / 32k tokens")
        self._token_label.setStyleSheet("font-size: 9px; color: #64748b;")
        token_row.addWidget(self._token_label)
        token_row.addStretch()
        layout.addLayout(token_row)

        self._token_bar = QProgressBar()
        self._token_bar.setRange(0, 100)
        self._token_bar.setValue(0)
        self._token_bar.setTextVisible(False)
        self._token_bar.setFixedHeight(5)
        layout.addWidget(self._token_bar)

        self._token_warning = QLabel("")
        self._token_warning.setStyleSheet("font-size: 9px; color: #f59e0b;")
        self._token_warning.setWordWrap(True)
        self._token_warning.setVisible(False)
        layout.addWidget(self._token_warning)

    # â”€â”€ public API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def model(self) -> str:
        return self._model_combo.currentText().strip()

    def set_model_list(self, models: list[str]):
        current = self.model()
        self._model_combo.clear()
        self._model_combo.addItems(models if models else [DEFAULT_MODEL])
        idx = self._model_combo.findText(current)
        if idx >= 0:
            self._model_combo.setCurrentIndex(idx)

    def add_path(self, path_str: str) -> bool:
        """Add a file. Returns False if already present."""
        if path_str in self._items:
            return False
        item = FileItemWidget(path_str)
        item.remove_requested.connect(self.remove_path)
        self._file_layout.insertWidget(self._file_layout.count() - 1, item)
        self._items[path_str] = item
        self._refresh_counts()
        self.files_changed.emit()
        return True

    def remove_path(self, path_str: str):
        if path_str not in self._items:
            return
        item = self._items.pop(path_str)
        self._file_layout.removeWidget(item)
        item.deleteLater()
        self._refresh_counts()
        self.files_changed.emit()

    def clear_files(self):
        for p in list(self._items):
            self.remove_path(p)

    def get_paths(self) -> list[str]:
        return list(self._items.keys())

    def set_file_status(self, path_str: str, status: str, token_count: int = None):
        if path_str in self._items:
            self._items[path_str].set_status(status, token_count)
            self._refresh_counts()

    def populate_from_paths(self, paths: list[str]):
        """Restore file list (e.g., from a loaded session)."""
        self.clear_files()
        for p in paths:
            self.add_path(p)
            if not Path(p).exists():
                self.set_file_status(p, FileItemWidget.STATUS_DELETED)

    # â”€â”€ private helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _refresh_counts(self):
        n = len(self._items)
        self._ctx_label.setText(f"CONTEXT â€” {n} FILE{'S' if n != 1 else ''}")

        total = sum(w.token_estimate() for w in self._items.values())
        k = total // 1000
        pct = min(100, total * 100 // self.MAX_CONTEXT_TOKENS)
        self._token_label.setText(f"~{k}k / {self.MAX_CONTEXT_TOKENS // 1000}k tokens")
        self._token_bar.setValue(pct)

        if pct >= 95:
            self._token_bar.setStyleSheet("QProgressBar::chunk { background: #ef4444; }")
            self._token_warning.setText(
                "Context nearly full â€” remove files or start a new session"
            )
            self._token_warning.setVisible(True)
        elif pct >= 75:
            self._token_bar.setStyleSheet("QProgressBar::chunk { background: #f59e0b; }")
            self._token_warning.setVisible(False)
        else:
            self._token_bar.setStyleSheet("QProgressBar::chunk { background: #22c55e; }")
            self._token_warning.setVisible(False)

    def _get_extensions(self) -> set:
        raw = self._ext_input.text().strip()
        if not raw:
            return SUPPORTED_EXTENSIONS
        return {e if e.startswith(".") else f".{e}" for e in raw.split()}

    def _add_files_dialog(self):
        ext_filter = (
            "Supported Files ("
            + " ".join(f"*{e}" for e in sorted(SUPPORTED_EXTENSIONS))
            + ");;All Files (*)"
        )
        paths, _ = QFileDialog.getOpenFileNames(self, "Select Files", "", ext_filter)
        for p in paths:
            self.add_path(p)

    def _add_folder_dialog(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not folder:
            return
        files = collect_files([folder], self._get_extensions(), self._recursive_check.isChecked())
        for f in files:
            self.add_path(str(f))

    # â”€â”€ drag and drop from OS file manager â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        exts = self._get_extensions()
        for url in event.mimeData().urls():
            local = url.toLocalFile()
            if not local:
                continue
            p = Path(local)
            if p.is_file() and p.suffix.lower() in exts:
                self.add_path(str(p))
            elif p.is_dir():
                for f in collect_files([str(p)], exts, self._recursive_check.isChecked()):
                    self.add_path(str(f))
        event.acceptProposedAction()


# â”€â”€ Main Window â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LocalFileAgent")
        self.setMinimumSize(900, 650)
        self.resize(1200, 800)

        # Chat state
        self._chat_messages: list = []
        self._chat_files_loaded = False
        self._chat_generation = 0

        # Streaming state
        self._stream_block: int | None = None   # QTextDocument block number
        self._stream_text = ""

        # Summarize results for current session
        self._summarize_results: dict = {}   # path_str -> summary_text

        # Session management
        self._session_manager = SessionManager()

        self._build_ui()
        self._fetch_models()

    # â”€â”€ layout â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_top_bar())

        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(8, 8, 8, 8)
        body_layout.setSpacing(8)

        self._sidebar = ContextSidebar()
        self._sidebar.setFixedWidth(260)
        self._sidebar.files_changed.connect(self._on_files_changed)
        self._sidebar.model_changed.connect(self._on_model_changed)
        body_layout.addWidget(self._sidebar)

        body_layout.addWidget(self._build_chat_panel(), 1)
        root.addWidget(body, 1)

        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready")

    def _build_top_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(44)
        bar.setStyleSheet("background: #ffffff; border-bottom: 1px solid #e2e8f0;")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 0, 16, 0)

        title = QLabel("â¬¡ LocalFileAgent")
        title.setStyleSheet("font-weight: 700; font-size: 15px; color: #1e293b;")
        layout.addWidget(title)
        layout.addStretch()

        self._sessions_btn = QPushButton("\U0001F550 Sessions")
        self._sessions_btn.clicked.connect(self._open_sessions)
        layout.addWidget(self._sessions_btn)

        return bar

    def _build_chat_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Chat history
        self._chat_history = QTextEdit()
        self._chat_history.setReadOnly(True)
        self._chat_history.setFont(QFont("Monospace", 10))
        self._chat_history.setPlaceholderText("Add files on the left, then start chattingâ€¦")
        layout.addWidget(self._chat_history, 1)

        # Summarize strip (hidden until files are loaded)
        self._summarize_strip = QWidget()
        strip_layout = QHBoxLayout(self._summarize_strip)
        strip_layout.setContentsMargins(4, 4, 4, 4)
        strip_layout.setSpacing(6)
        strip_layout.addWidget(QLabel("Summarize:"))
        self._summarize_btns_layout = QHBoxLayout()
        strip_layout.addLayout(self._summarize_btns_layout)
        strip_layout.addStretch()
        self._copy_all_btn = QPushButton("\U0001F4CB Copy All")
        self._copy_all_btn.setStyleSheet("font-size: 10px;")
        self._copy_all_btn.clicked.connect(self._copy_all_summaries)
        self._copy_all_btn.setEnabled(False)
        strip_layout.addWidget(self._copy_all_btn)
        self._summarize_strip.setVisible(False)
        layout.addWidget(self._summarize_strip)

        # Input row
        input_row = QHBoxLayout()
        self._chat_input = QLineEdit()
        self._chat_input.setPlaceholderText("Ask a question about the loaded filesâ€¦")
        self._chat_input.returnPressed.connect(self._send_chat)
        input_row.addWidget(self._chat_input, 1)
        self._send_btn = QPushButton("Send")
        self._send_btn.setMinimumWidth(80)
        self._send_btn.clicked.connect(self._send_chat)
        input_row.addWidget(self._send_btn)
        layout.addLayout(input_row)

        return panel

    # â”€â”€ model fetch â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _fetch_models(self):
        self._status_bar.showMessage("Connecting to Ollamaâ€¦")
        self._model_worker = ModelFetchWorker()
        self._model_worker.models_ready.connect(self._on_models_ready)
        self._model_worker.error.connect(self._on_model_error)
        self._model_worker.start()

    def _on_models_ready(self, models: list):
        self._sidebar.set_model_list(models)
        self._status_bar.showMessage(
            f"Ollama connected â€” {len(models)} model(s) available"
        )

    def _on_model_error(self, _msg: str):
        self._status_bar.showMessage(
            "âš   Ollama not reachable â€” start it with:  ollama serve"
        )

    # â”€â”€ sidebar signals â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _on_files_changed(self):
        self._chat_files_loaded = False
        self._rebuild_summarize_strip()

    def _on_model_changed(self, _new_model: str):
        if self._chat_files_loaded:
            self._chat_files_loaded = False
            self._append_system("Model changed â€” context will reload on next message.")

    # â”€â”€ chat helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _append_system(self, text: str):
        escaped = html.escape(text)
        self._chat_history.append(
            f'<p><i><span style="color:#888888">{escaped}</span></i></p>'
        )
        sb = self._chat_history.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _append_chat(self, label: str, text: str, color: str):
        escaped_label = html.escape(label)
        escaped_text = html.escape(text).replace("\n", "<br>")
        self._chat_history.append(
            f'<p><b><span style="color:{color}">{escaped_label}:</span></b>'
            f" {escaped_text}</p>"
        )
        sb = self._chat_history.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _start_stream_bubble(self, model_name: str):
        """Append an empty assistant bubble; track its block for token insertion."""
        escaped = html.escape(model_name)
        self._chat_history.append(
            f'<p><b><span style="color:#16a34a">{escaped}:</span></b> ▌</p>'
        )
        self._stream_block = self._chat_history.document().blockCount() - 1
        self._stream_text = ""
        sb = self._chat_history.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_token_ready(self, token: str):
        """Insert token before the ▌ cursor in the streaming bubble."""
        doc = self._chat_history.document()
        block = doc.findBlockByNumber(self._stream_block)
        cursor = QTextCursor(block)
        cursor.movePosition(QTextCursor.MoveOperation.EndOfBlock)
        # Select and replace the trailing ▌ with token + new ▌
        cursor.movePosition(
            QTextCursor.MoveOperation.PreviousCharacter,
            QTextCursor.MoveMode.KeepAnchor,
        )
        cursor.insertText(token + "▌")
        self._stream_text += token
        sb = self._chat_history.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _finish_stream(self):
        """Remove the ▌ cursor character from the completed bubble."""
        if self._stream_block is None:
            return
        doc = self._chat_history.document()
        block = doc.findBlockByNumber(self._stream_block)
        cursor = QTextCursor(block)
        cursor.movePosition(QTextCursor.MoveOperation.EndOfBlock)
        cursor.movePosition(
            QTextCursor.MoveOperation.PreviousCharacter,
            QTextCursor.MoveMode.KeepAnchor,
        )
        cursor.removeSelectedText()
        self._stream_block = None

    def _set_chat_input_enabled(self, enabled: bool):
        self._send_btn.setEnabled(enabled)
        self._chat_input.setEnabled(enabled)

    # -- chat send and reply handlers --

    def _send_chat(self):
        user_text = self._chat_input.text().strip()
        if not user_text:
            return

        model = self._sidebar.model()
        if not model:
            QMessageBox.warning(self, "No Model", "Please select or enter a model name.")
            return

        self._chat_input.clear()
        self._set_chat_input_enabled(False)

        generation = self._chat_generation

        if not self._chat_files_loaded:
            all_paths = self._sidebar.get_paths()
            valid_paths = [
                p for p in all_paths
                if self._sidebar._items[p].status() != FileItemWidget.STATUS_DELETED
            ]
            if not valid_paths:
                QMessageBox.warning(self, "No Files", "Please add files or a folder first.")
                self._set_chat_input_enabled(True)
                return

            files = [Path(p) for p in valid_paths[:CONTEXT_FILE_CAP]]
            if len(valid_paths) > CONTEXT_FILE_CAP:
                self._append_system(
                    f"⚠  Only first {CONTEXT_FILE_CAP} of {len(valid_paths)} files loaded (token limit)."
                )
            self._append_system(f"Loading {len(files)} file(s) into context…")
            self._append_chat("You", user_text, "#3b82f6")

            self._chat_worker = StreamingChatWorker(
                [], model, files_to_load=files, user_text=user_text
            )
            self._chat_worker.context_info.connect(self._on_context_info)
            self._chat_worker.file_status.connect(
                lambda p, s, t: self._sidebar.set_file_status(p, s, t)
            )
        else:
            self._chat_messages.append({"role": "user", "content": user_text})
            self._append_chat("You", user_text, "#3b82f6")
            self._chat_worker = StreamingChatWorker(list(self._chat_messages), model)

        self._start_stream_bubble(model)
        self._chat_worker.token_ready.connect(self._on_token_ready)
        self._chat_worker.finished.connect(
            lambda msgs, g=generation: self._on_chat_reply(msgs, g)
        )
        self._chat_worker.error.connect(self._on_chat_error)
        self._chat_worker.start()

    def _on_context_info(self, info: str):
        self._append_system(info)
        self._chat_files_loaded = True

    def _on_chat_reply(self, updated_messages: list, generation: int):
        self._finish_stream()
        if generation != self._chat_generation:
            self._set_chat_input_enabled(True)
            return
        self._chat_messages = updated_messages
        self._set_chat_input_enabled(True)
        self._chat_input.setFocus()
        self._auto_save()

    def _on_chat_error(self, msg: str):
        self._finish_stream()
        if self._stream_text:
            self._append_system("(response interrupted)")
        self._set_chat_input_enabled(True)
        QMessageBox.critical(self, "Ollama Error", msg)

    def _auto_save(self):
        pass   # implemented in Task 9

    # -- placeholder stubs (implemented in Tasks 8-9) --

    def _open_sessions(self):
        pass

    def _rebuild_summarize_strip(self):
        """Rebuild per-file summarize buttons to match current sidebar file list."""
        # Clear existing buttons
        while self._summarize_btns_layout.count():
            item = self._summarize_btns_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        paths = self._sidebar.get_paths()
        if not paths:
            self._summarize_strip.setVisible(False)
            return

        self._summarize_strip.setVisible(True)
        for path_str in paths:
            name = Path(path_str).name
            btn = QPushButton(f"{name} ▶")
            btn.setStyleSheet("font-size: 10px;")
            btn.clicked.connect(lambda checked=False, p=path_str: self._run_single_summarize(p))
            self._summarize_btns_layout.addWidget(btn)

    def _run_single_summarize(self, path_str: str):
        model = self._sidebar.model()
        self._set_chat_input_enabled(False)
        self._append_system(f"Summarizing {Path(path_str).name}…")

        self._summarize_worker = SummarizeWorker([Path(path_str)], model)
        self._summarize_worker.file_done.connect(self._on_summarize_done)
        self._summarize_worker.finished.connect(
            lambda: self._set_chat_input_enabled(True)
        )
        self._summarize_worker.error.connect(self._on_summarize_error)
        self._summarize_worker.start()

    def _on_summarize_done(self, path_str: str, summary: str):
        self._summarize_results[path_str] = summary
        name = Path(path_str).name
        self._append_system(f"── Summary: {name} ──")
        self._append_chat("Summary", summary, "#7c3aed")
        self._copy_all_btn.setEnabled(bool(self._summarize_results))
        self._auto_save()

    def _on_summarize_error(self, msg: str):
        self._set_chat_input_enabled(True)
        QMessageBox.critical(self, "Summarize Error", msg)

    def _copy_all_summaries(self):
        if not self._summarize_results:
            return
        lines = []
        for path_str, summary in self._summarize_results.items():
            lines.append(f"## {Path(path_str).name}\n\n{summary}\n")
        text = "\n".join(lines)
        QApplication.clipboard().setText(text)
        self._status_bar.showMessage("Summaries copied to clipboard.")


# â”€â”€ Entry point â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
