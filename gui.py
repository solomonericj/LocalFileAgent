#!/usr/bin/env python3
"""
gui.py — PySide6 graphical interface for LocalfileAgent.

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

# ── Workers ────────────────────────────────────────────────────────────────────

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
                    summary = "(skipped — file no longer accessible)"
                else:
                    summary = "(empty file)" if size == 0 else f"(skipped — too large: {size:,} bytes)"
            else:
                try:
                    summary = query_ollama_generate(
                        f"File: {path.name}\n\n{content}",
                        SUMMARISE_SYSTEM,
                        self.model,
                    )
                except TimeoutError as exc:
                    summary = f"(skipped — timeout: {exc})"
                except ConnectionError as exc:
                    self.error.emit(str(exc))
                    return
            self.file_done.emit(str(path), summary)
        self.finished.emit()


class ChatWorker(QThread):
    reply_ready = Signal(str, list)
    context_info = Signal(str)   # emitted after file loading on first message
    error = Signal(str)

    def __init__(self, messages: list, model: str, *,
                 files_to_load: list = None, user_text: str = None):
        super().__init__()
        self.messages = list(messages)   # snapshot — never share the live list
        self.model = model
        self.files_to_load = files_to_load
        self.user_text = user_text

    def run(self):
        try:
            if self.files_to_load is not None:
                file_block, skipped = build_file_block(self.files_to_load)
                if not file_block.strip():
                    self.error.emit("No readable content found in the selected files.")
                    return
                loaded = len(self.files_to_load) - len(skipped)
                info = f"Context ready: {loaded} file(s) loaded"
                if skipped:
                    info += f", {len(skipped)} skipped ({', '.join(skipped)})"
                self.context_info.emit(info)
                system_prompt = CHAT_SYSTEM_TEMPLATE.format(n=loaded, file_block=file_block)
                self.messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": self.user_text},
                ]
            reply, updated = query_ollama_chat(self.messages, self.model)
            self.reply_ready.emit(reply, updated)
        except (ConnectionError, TimeoutError) as exc:
            self.error.emit(str(exc))


# ── FileItemWidget ─────────────────────────────────────────────────────────────

class FileItemWidget(QWidget):
    remove_requested = Signal(str)   # emits path_str

    STATUS_PENDING = "pending"
    STATUS_LOADED  = "loaded"
    STATUS_SKIPPED = "skipped"
    STATUS_DELETED = "deleted"

    _STATUS_STYLES = {
        STATUS_PENDING: ("↻", "#64748b"),
        STATUS_LOADED:  ("✓", "#22c55e"),
        STATUS_SKIPPED: ("⚠", "#f59e0b"),
        STATUS_DELETED: ("✕", "#ef4444"),
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

        self._badge = QLabel("↻")
        self._badge.setFixedWidth(14)
        row.addWidget(self._badge)

        self._name_label = QLabel()
        self._name_label.setToolTip(self._path_str)
        self._name_label.setStyleSheet("font-size: 11px; color: #1e293b;")
        row.addWidget(self._name_label, 1)

        self._remove_btn = QPushButton("✕")
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
            self._token_label.setText("skipped — too large")
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


# ── ContextSidebar ─────────────────────────────────────────────────────────────

class ContextSidebar(QWidget):
    files_changed = Signal()        # emitted on any add/remove
    model_changed = Signal(str)     # emitted when model combo changes

    MAX_CONTEXT_TOKENS = 32_000

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items: dict[str, FileItemWidget] = {}   # path_str -> widget
        self.setAcceptDrops(True)
        self._build_ui()

    # ── construction ──────────────────────────────────────────────────────────

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

        self._ctx_label = QLabel("CONTEXT — 0 FILES")
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

    # ── public API ────────────────────────────────────────────────────────────

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

    # ── private helpers ───────────────────────────────────────────────────────

    def _refresh_counts(self):
        n = len(self._items)
        self._ctx_label.setText(f"CONTEXT — {n} FILE{'S' if n != 1 else ''}")

        total = sum(w.token_estimate() for w in self._items.values())
        k = total // 1000
        pct = min(100, total * 100 // self.MAX_CONTEXT_TOKENS)
        self._token_label.setText(f"~{k}k / {self.MAX_CONTEXT_TOKENS // 1000}k tokens")
        self._token_bar.setValue(pct)

        if pct >= 95:
            self._token_bar.setStyleSheet("QProgressBar::chunk { background: #ef4444; }")
            self._token_warning.setText(
                "Context nearly full — remove files or start a new session"
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

    # ── drag and drop from OS file manager ───────────────────────────────────

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


# ── Main Window ────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LocalFileAgent")
        self.setMinimumSize(900, 650)
        self.resize(1100, 780)

        self._chat_messages: list = []
        self._chat_files_loaded = False
        self._chat_generation = 0
        self._summarize_results: list = []

        self._build_ui()
        self._fetch_models()

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 4)
        root.setSpacing(8)

        root.addWidget(self._build_options_bar())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_file_panel())
        splitter.addWidget(self._build_tabs())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        root.addWidget(splitter, 1)

        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready")

    def _build_options_bar(self) -> QWidget:
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        layout.addWidget(QLabel("Model:"))
        self._model_combo = QComboBox()
        self._model_combo.setEditable(True)
        self._model_combo.addItem(DEFAULT_MODEL)
        self._model_combo.setMinimumWidth(180)
        self._model_combo.setToolTip("Ollama model — auto-populated from localhost:11434")
        layout.addWidget(self._model_combo)

        self._recursive_check = QCheckBox("Recursive")
        self._recursive_check.setToolTip("Recurse into subdirectories when adding a folder")
        layout.addWidget(self._recursive_check)

        layout.addWidget(QLabel("Extensions:"))
        self._ext_input = QLineEdit()
        self._ext_input.setPlaceholderText(".py .md .txt  (blank = all supported)")
        self._ext_input.setMaximumWidth(220)
        self._ext_input.setToolTip(
            "Space-separated extensions to include, e.g.  .py .md\n"
            "Leave blank to include all supported file types."
        )
        layout.addWidget(self._ext_input)

        layout.addStretch()
        return bar

    def _build_file_panel(self) -> QGroupBox:
        group = QGroupBox("Files")
        layout = QVBoxLayout(group)

        self._file_list = QListWidget()
        self._file_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self._file_list.setToolTip("Files to process — hover an item to see its full path")
        layout.addWidget(self._file_list)

        btn_row = QHBoxLayout()
        for label, slot in [
            ("Add Files…", self._add_files),
            ("Add Folder…", self._add_folder),
            ("Remove", self._remove_selected),
            ("Clear All", self._clear_files),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(slot)
            btn_row.addWidget(btn)
        layout.addLayout(btn_row)

        self._file_count_label = QLabel("No files selected")
        layout.addWidget(self._file_count_label)
        return group

    def _build_tabs(self) -> QTabWidget:
        tabs = QTabWidget()
        tabs.addTab(self._build_summarize_tab(), "Summarize")
        tabs.addTab(self._build_chat_tab(), "Chat")
        return tabs

    def _build_summarize_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        ctrl = QHBoxLayout()
        self._run_btn = QPushButton("▶  Run Summarize")
        self._run_btn.setMinimumHeight(34)
        self._run_btn.clicked.connect(self._run_summarize)
        ctrl.addWidget(self._run_btn)

        self._save_btn = QPushButton("Save Output…")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._save_output)
        ctrl.addWidget(self._save_btn)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)

        self._progress_label = QLabel("")
        layout.addWidget(self._progress_label)

        self._output_text = QTextEdit()
        self._output_text.setReadOnly(True)
        self._output_text.setFont(QFont("Monospace", 10))
        self._output_text.setPlaceholderText("Summaries will appear here…")
        layout.addWidget(self._output_text, 1)

        return w

    def _build_chat_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        info_row = QHBoxLayout()
        self._chat_info_label = QLabel("Select files and start chatting.")
        info_row.addWidget(self._chat_info_label)
        info_row.addStretch()

        clear_btn = QPushButton("Clear History")
        clear_btn.clicked.connect(self._clear_chat_history)
        info_row.addWidget(clear_btn)
        layout.addLayout(info_row)

        self._chat_history = QTextEdit()
        self._chat_history.setReadOnly(True)
        self._chat_history.setFont(QFont("Monospace", 10))
        self._chat_history.setPlaceholderText("Conversation will appear here…")
        layout.addWidget(self._chat_history, 1)

        input_row = QHBoxLayout()
        self._chat_input = QLineEdit()
        self._chat_input.setPlaceholderText("Ask a question about the loaded files…")
        self._chat_input.returnPressed.connect(self._send_chat)
        input_row.addWidget(self._chat_input, 1)

        self._send_btn = QPushButton("Send")
        self._send_btn.setMinimumWidth(80)
        self._send_btn.clicked.connect(self._send_chat)
        input_row.addWidget(self._send_btn)
        layout.addLayout(input_row)

        return w

    # ── File management ────────────────────────────────────────────────────────

    def _get_extensions(self) -> set:
        raw = self._ext_input.text().strip()
        if not raw:
            return SUPPORTED_EXTENSIONS
        return {e if e.startswith(".") else f".{e}" for e in raw.split()}

    def _add_files(self):
        ext_list = (
            "Supported Files ("
            + " ".join(f"*{e}" for e in sorted(SUPPORTED_EXTENSIONS))
            + ");;All Files (*)"
        )
        paths, _ = QFileDialog.getOpenFileNames(self, "Select Files", "", ext_list)
        for p in paths:
            self._add_path(p)
        self._update_count()

    def _add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not folder:
            return
        files = collect_files([folder], self._get_extensions(), self._recursive_check.isChecked())
        for f in files:
            self._add_path(str(f))
        self._update_count()
        self._status_bar.showMessage(f"Added {len(files)} file(s) from folder.")

    def _add_path(self, path_str: str):
        for i in range(self._file_list.count()):
            if self._file_list.item(i).data(Qt.ItemDataRole.UserRole) == path_str:
                return
        item = QListWidgetItem(Path(path_str).name)
        item.setData(Qt.ItemDataRole.UserRole, path_str)
        item.setToolTip(path_str)
        self._file_list.addItem(item)

    def _remove_selected(self):
        for item in self._file_list.selectedItems():
            self._file_list.takeItem(self._file_list.row(item))
        self._update_count()

    def _clear_files(self):
        self._file_list.clear()
        self._update_count()

    def _update_count(self):
        n = self._file_list.count()
        self._file_count_label.setText(f"{n} file(s) selected" if n else "No files selected")
        self._chat_files_loaded = False

    def _selected_paths(self) -> list:
        return [
            Path(self._file_list.item(i).data(Qt.ItemDataRole.UserRole))
            for i in range(self._file_list.count())
        ]

    # ── Model fetch ────────────────────────────────────────────────────────────

    def _fetch_models(self):
        self._status_bar.showMessage("Connecting to Ollama…")
        self._model_worker = ModelFetchWorker()
        self._model_worker.models_ready.connect(self._on_models_ready)
        self._model_worker.error.connect(self._on_model_error)
        self._model_worker.start()

    def _on_models_ready(self, models: list):
        current = self._model_combo.currentText()
        self._model_combo.clear()
        self._model_combo.addItems(models if models else [DEFAULT_MODEL])
        idx = self._model_combo.findText(current)
        if idx >= 0:
            self._model_combo.setCurrentIndex(idx)
        self._status_bar.showMessage(
            f"Ollama connected — {len(models)} model(s) available"
        )

    def _on_model_error(self, _msg: str):
        self._status_bar.showMessage(
            "⚠  Ollama not reachable — start it with:  ollama serve"
        )

    # ── Summarize ──────────────────────────────────────────────────────────────

    def _run_summarize(self):
        files = self._selected_paths()
        if not files:
            QMessageBox.warning(self, "No Files", "Please add files or a folder first.")
            return
        model = self._model_combo.currentText().strip()
        if not model:
            QMessageBox.warning(self, "No Model", "Please select or enter a model name.")
            return

        self._run_btn.setEnabled(False)
        self._save_btn.setEnabled(False)
        self._output_text.clear()
        self._summarize_results.clear()
        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, len(files))
        self._progress_bar.setValue(0)
        self._progress_label.setText("")

        self._summarize_worker = SummarizeWorker(files, model)
        self._summarize_worker.progress.connect(self._on_summarize_progress)
        self._summarize_worker.file_done.connect(self._on_file_done)
        self._summarize_worker.finished.connect(self._on_summarize_finished)
        self._summarize_worker.error.connect(self._on_summarize_error)
        self._summarize_worker.start()

    def _on_summarize_progress(self, current: int, total: int, filename: str):
        self._progress_bar.setValue(current - 1)
        self._progress_label.setText(f"[{current}/{total}] Processing {filename}…")

    def _on_file_done(self, path_str: str, summary: str):
        self._summarize_results.append((path_str, summary))
        name = Path(path_str).name
        sep = "─" * 60
        self._output_text.append(sep)
        self._output_text.append(f"FILE: {name}")
        self._output_text.append(f"PATH: {path_str}")
        self._output_text.append(sep)
        self._output_text.append(summary)
        self._output_text.append("")

    def _on_summarize_finished(self):
        n = len(self._summarize_results)
        self._progress_bar.setVisible(False)
        self._progress_label.setText(f"Done — {n} file(s) summarized.")
        self._run_btn.setEnabled(True)
        self._save_btn.setEnabled(bool(self._summarize_results))
        self._status_bar.showMessage(f"Summarization complete: {n} file(s).")

    def _on_summarize_error(self, msg: str):
        self._run_btn.setEnabled(True)
        self._progress_bar.setVisible(False)
        self._progress_label.setText("")
        QMessageBox.critical(self, "Ollama Error", msg)

    def _save_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Output", "summaries.md",
            "Markdown (*.md);;Plain Text (*.txt);;All Files (*)",
        )
        if not path:
            return
        output_path = Path(path)
        if output_path.suffix.lower() == ".md":
            lines = ["# File Summaries\n"]
            for p_str, summary in self._summarize_results:
                p = Path(p_str)
                lines += [f"## `{p.name}`", f"**Path:** `{p_str}`\n", summary, ""]
            text = "\n".join(lines)
        else:
            sep = "─" * 60
            parts = []
            for p_str, summary in self._summarize_results:
                parts += [sep, f"FILE : {p_str}", sep, summary, ""]
            text = "\n".join(parts)
        try:
            output_path.write_text(text, encoding="utf-8")
        except OSError as exc:
            QMessageBox.critical(self, "Save Error", f"Could not write file:\n{exc}")
            return
        self._status_bar.showMessage(f"Saved to {output_path}")

    # ── Chat ───────────────────────────────────────────────────────────────────

    def _send_chat(self):
        user_text = self._chat_input.text().strip()
        if not user_text:
            return

        model = self._model_combo.currentText().strip()
        if not model:
            QMessageBox.warning(self, "No Model", "Please select or enter a model name.")
            return

        self._chat_input.clear()
        self._set_chat_input_enabled(False)

        if not self._chat_files_loaded:
            files = self._selected_paths()
            if not files:
                QMessageBox.warning(self, "No Files", "Please add files or a folder first.")
                self._set_chat_input_enabled(True)
                return

            capped = files[:CONTEXT_FILE_CAP]
            if len(files) > CONTEXT_FILE_CAP:
                self._append_system(
                    f"⚠  Only first {CONTEXT_FILE_CAP} of {len(files)} files loaded (token limit)."
                )
            self._append_system(f"Loading {len(capped)} file(s) into context…")
            self._append_chat("You", user_text, color="#4A90D9")
            generation = self._chat_generation
            self._chat_worker = ChatWorker(
                [], model, files_to_load=capped, user_text=user_text
            )
            self._chat_worker.context_info.connect(self._on_context_info)
        else:
            self._chat_messages.append({"role": "user", "content": user_text})
            self._append_chat("You", user_text, color="#4A90D9")
            generation = self._chat_generation
            self._chat_worker = ChatWorker(list(self._chat_messages), model)

        self._chat_worker.reply_ready.connect(
            lambda reply, msgs, g=generation: self._on_chat_reply(reply, msgs, g)
        )
        self._chat_worker.error.connect(self._on_chat_error)
        self._chat_worker.start()

    def _on_context_info(self, info: str):
        self._append_system(info)
        self._chat_info_label.setText(info)
        self._chat_files_loaded = True

    def _on_chat_reply(self, reply: str, updated_messages: list, generation: int):
        if generation != self._chat_generation:
            self._set_chat_input_enabled(True)
            return
        self._chat_messages = updated_messages
        model = self._model_combo.currentText()
        self._append_chat(model, reply, color="#27AE60")
        self._set_chat_input_enabled(True)
        self._chat_input.setFocus()

    def _on_chat_error(self, msg: str):
        self._set_chat_input_enabled(True)
        QMessageBox.critical(self, "Ollama Error", msg)

    def _set_chat_input_enabled(self, enabled: bool):
        self._send_btn.setEnabled(enabled)
        self._chat_input.setEnabled(enabled)

    def _append_system(self, text: str):
        escaped = html.escape(text)
        self._chat_history.append(
            f'<p><i><span style="color:#888888">{escaped}</span></i></p>'
        )
        self._scroll_chat()

    def _append_chat(self, label: str, text: str, color: str = "#000000"):
        escaped_label = html.escape(label)
        escaped_text = html.escape(text).replace("\n", "<br>")
        self._chat_history.append(
            f'<p><b><span style="color:{color}">{escaped_label}:</span></b> {escaped_text}</p>'
        )
        self._scroll_chat()

    def _scroll_chat(self):
        sb = self._chat_history.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _clear_chat_history(self):
        self._chat_history.clear()
        self._chat_messages = []
        self._chat_files_loaded = False
        self._chat_generation += 1
        self._chat_info_label.setText("History cleared — select files and start chatting.")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("LocalFileAgent")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
