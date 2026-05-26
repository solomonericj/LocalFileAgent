#!/usr/bin/env python3
"""
gui.py — PySide6 graphical interface for LocalfileAgent.

Run with:  python gui.py
"""

import html
import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QListWidget, QListWidgetItem, QPushButton, QLabel,
    QComboBox, QCheckBox, QLineEdit, QTextEdit, QProgressBar,
    QFileDialog, QStatusBar, QMessageBox,
    QFrame, QScrollArea, QDialog,
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QElapsedTimer
from PySide6.QtGui import QFont, QTextCursor

sys.path.insert(0, str(Path(__file__).parent))
from LocalfileAgent import (
    SUPPORTED_EXTENSIONS, DEFAULT_MODEL, OLLAMA_TAGS,
    SUMMARISE_SYSTEM, CHAT_SYSTEM_TEMPLATE, CONTEXT_FILE_CAP,
    RAG_SYSTEM, DEFAULT_EMBED_MODEL,
    read_file_safe, collect_files,
    query_ollama_generate, stream_ollama_chat,
)
from rag import build_index, retrieve, build_rag_prompt
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
    progress = Signal(int, int, str)      # current, total, filename
    file_done = Signal(str, str, float)   # path_str, summary, elapsed_seconds
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
            elapsed = 0.0
            if content is None:
                try:
                    size = path.stat().st_size
                except OSError:
                    summary = "(skipped — file no longer accessible)"
                else:
                    summary = "(empty file)" if size == 0 else f"(skipped — too large: {size:,} bytes)"
            else:
                try:
                    t0 = time.monotonic()
                    summary = query_ollama_generate(
                        f"File: {path.name}\n\n{content}",
                        SUMMARISE_SYSTEM,
                        self.model,
                    )
                    elapsed = time.monotonic() - t0
                except TimeoutError as exc:
                    summary = f"(skipped — timeout: {exc})"
                except ConnectionError as exc:
                    self.error.emit(str(exc))
                    return
            self.file_done.emit(str(path), summary, elapsed)
        self.finished.emit()


class StreamingChatWorker(QThread):
    token_ready  = Signal(str)
    finished     = Signal(list)    # updated_messages (plain history + assistant)
    timing       = Signal(float)   # elapsed seconds for the LLM call
    context_info = Signal(str)     # human-readable summary after file load
    file_status  = Signal(str, str, int)  # path_str, status constant, token_count
    index_ready  = Signal(object)  # emits the built VectorIndex for the window to cache
    error        = Signal(str)

    def __init__(self, messages: list, model: str, *,
                 files_to_load: list = None, user_text: str = None,
                 rag_index=None, embed_model: str = DEFAULT_EMBED_MODEL,
                 top_k: int = 5, use_rag: bool = True,
                 preserve_history: bool = False):
        super().__init__()
        self.messages = list(messages)         # snapshot — never share the live list
        self.model = model
        self.files_to_load = files_to_load     # list[Path] or None
        self.user_text = user_text
        self.rag_index = rag_index
        self.embed_model = embed_model
        self.top_k = top_k
        self.use_rag = use_rag
        # When True, build/rebuild the index but DON'T reset self.messages — used
        # to re-index a resumed session on its next turn without wiping history.
        self.preserve_history = preserve_history

    def run(self):
        try:
            if self.files_to_load is not None:
                if self.preserve_history:
                    # Resumed session: rebuild the index but keep replayed history.
                    self._rebuild_index_keep_history()
                elif self.use_rag:
                    index = self._build_rag_index()
                    if index is not None:
                        self.rag_index = index
                        self.index_ready.emit(index)
                        self.messages = [
                            {"role": "system", "content": RAG_SYSTEM},
                            {"role": "user", "content": self.user_text},
                        ]
                    else:
                        # numpy missing or nothing indexable — tell the user we
                        # are not retrieving, then load full file contents.
                        self.context_info.emit("RAG unavailable — using full file context.")
                        self._load_full_context()
                else:
                    self._load_full_context()

            api_messages = self._compose_api_messages()

            accumulated = ""
            t0 = time.monotonic()
            for token in stream_ollama_chat(api_messages, self.model):
                if self.isInterruptionRequested():
                    break   # window is closing — stop promptly so the thread can exit
                accumulated += token
                self.token_ready.emit(token)
            elapsed = time.monotonic() - t0

            updated = self.messages + [{"role": "assistant", "content": accumulated}]
            self.timing.emit(elapsed)
            self.finished.emit(updated)

        except (ConnectionError, TimeoutError) as exc:
            self.error.emit(str(exc))
        except RuntimeError as exc:
            # Expected, user-facing failures (e.g. no readable content) — emit
            # the message as-is rather than wrapping it as "Unexpected error".
            self.error.emit(str(exc))
        except Exception as exc:
            self.error.emit(f"Unexpected error: {exc}")

    def _build_rag_index(self):
        """Build the vector index; return None to signal fallback to full-context."""
        try:
            index = build_index(self.files_to_load, self.embed_model)
        except (ImportError, ValueError):
            # numpy missing (rag._require_numpy) or the embed model returned no
            # usable vectors — fall back to full-context stuffing rather than
            # failing the whole turn. (ConnectionError/TimeoutError propagate to
            # run()'s handler and surface as a normal error.)
            return None
        if len(index) == 0:
            return None
        self.context_info.emit(
            f"Indexed {len(index)} chunk(s) from {len(self.files_to_load)} file(s)"
        )
        for path in self.files_to_load:
            try:
                est = path.stat().st_size // 4   # rough token estimate for the sidebar
            except OSError:
                est = 0
            self.file_status.emit(str(path), FileItemWidget.STATUS_LOADED, est)
        return index

    def _rebuild_index_keep_history(self):
        """Resumed-session path: rebuild the vector index without disturbing the
        replayed history (self.messages already ends with the new user turn).
        The next turn's retrieval then fires via the freshly cached index."""
        index = self._build_rag_index()
        if index is not None:
            self.rag_index = index
            self.index_ready.emit(index)
        else:
            self.context_info.emit(
                "Could not rebuild the index — answering without retrieval this turn."
            )

    def _load_full_context(self):
        """Original context-stuffing path; sets self.messages in place."""
        parts, skipped_names = [], []
        for path in self.files_to_load:
            content = read_file_safe(path)
            if content is None:
                skipped_names.append(path.name)
                self.file_status.emit(str(path), FileItemWidget.STATUS_SKIPPED, 0)
            else:
                parts.append(f"### {path.name}\nPath: {path}\n\n{content}")
                self.file_status.emit(str(path), FileItemWidget.STATUS_LOADED, len(content) // 4)
        file_block = "\n\n---\n\n".join(parts)
        if not file_block.strip():
            raise RuntimeError("No readable content found in the selected files.")
        loaded = len(self.files_to_load) - len(skipped_names)
        info = f"Context ready: {loaded} file(s) loaded"
        if skipped_names:
            info += f", {len(skipped_names)} skipped ({', '.join(skipped_names)})"
        self.context_info.emit(info)
        self.messages = [
            {"role": "system", "content": CHAT_SYSTEM_TEMPLATE.format(n=loaded, file_block=file_block)},
            {"role": "user", "content": self.user_text},
        ]

    def _compose_api_messages(self):
        """Return messages for the API call; inject RAG context into the last user turn."""
        if self.rag_index is not None and self.use_rag and self.user_text is not None:
            chunks = retrieve(self.rag_index, self.user_text, self.embed_model, self.top_k)
            composed = build_rag_prompt(chunks, self.user_text)
            return self.messages[:-1] + [{"role": "user", "content": composed}]
        return list(self.messages)


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
    embed_model_changed = Signal(str)  # emitted when embed-model combo changes

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

        embed_lbl = QLabel("EMBED MODEL")
        embed_lbl.setStyleSheet(
            "font-size: 9px; color: #64748b; letter-spacing: 1px; font-weight: 600;"
        )
        layout.addWidget(embed_lbl)

        self._embed_combo = QComboBox()
        self._embed_combo.setEditable(True)
        self._embed_combo.addItem(DEFAULT_EMBED_MODEL)
        self._embed_combo.currentTextChanged.connect(self.embed_model_changed.emit)
        layout.addWidget(self._embed_combo)

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

    def embed_model(self) -> str:
        return self._embed_combo.currentText().strip()

    def set_model_list(self, models: list[str]):
        current = self.model()
        self._model_combo.clear()
        self._model_combo.addItems(models if models else [DEFAULT_MODEL])
        idx = self._model_combo.findText(current)
        if idx >= 0:
            self._model_combo.setCurrentIndex(idx)

        current_embed = self.embed_model()
        self._embed_combo.clear()
        embed_choices = list(models) if models else []
        if DEFAULT_EMBED_MODEL not in embed_choices:
            embed_choices.insert(0, DEFAULT_EMBED_MODEL)
        self._embed_combo.addItems(embed_choices)
        eidx = self._embed_combo.findText(current_embed)
        if eidx >= 0:
            self._embed_combo.setCurrentIndex(eidx)
        else:
            self._embed_combo.setEditText(current_embed or DEFAULT_EMBED_MODEL)

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
        self.blockSignals(True)
        try:
            for p in paths:
                self.add_path(p)
                if not Path(p).exists():
                    self.set_file_status(p, FileItemWidget.STATUS_DELETED)
        finally:
            self.blockSignals(False)
        self._refresh_counts()
        self.files_changed.emit()

    def get_valid_paths(self) -> list[str]:
        """Return paths that are not marked as deleted."""
        return [
            p for p, w in self._items.items()
            if w.status() != FileItemWidget.STATUS_DELETED
        ]

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
                "Large context — RAG sends only the most relevant chunks each turn "
                "(full-context mode, --no-rag, may exceed the model's window)"
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
            # Skip files the app can't read (e.g. picked via the "All Files"
            # filter), matching the drag-and-drop path.
            if Path(p).suffix.lower() in SUPPORTED_EXTENSIONS:
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


class SessionDialog(QDialog):
    def __init__(self, session_manager: SessionManager, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sessions")
        self.setMinimumSize(640, 380)
        self._sm = session_manager
        self._selected: dict | None = None
        self._new_requested = False
        self._build_ui()
        self._load_list()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        self._list = QListWidget()
        self._list.itemSelectionChanged.connect(self._on_sel_changed)
        layout.addWidget(self._list)

        btn_row = QHBoxLayout()

        self._load_btn = QPushButton("Load Session")
        self._load_btn.setEnabled(False)
        self._load_btn.clicked.connect(self.accept)
        btn_row.addWidget(self._load_btn)

        self._del_btn = QPushButton("Delete")
        self._del_btn.setEnabled(False)
        self._del_btn.clicked.connect(self._delete_selected)
        btn_row.addWidget(self._del_btn)

        btn_row.addStretch()

        new_btn = QPushButton("New Session")
        new_btn.clicked.connect(self._on_new)
        btn_row.addWidget(new_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        layout.addLayout(btn_row)

    def _load_list(self):
        self._list.clear()
        for s in self._sm.list():
            created = s.get("created", "?")
            model   = s.get("model", "?")
            n_files = len(s.get("files", []))
            user_msgs = [m for m in s.get("messages", []) if m.get("role") == "user"]
            preview = (user_msgs[0].get("content", "")[:60] + "\u2026") if user_msgs else "(no messages)"
            label = f"{created}  \u00b7  {model}  \u00b7  {n_files} file(s)  \u00b7  {preview}"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, s["_path"])
            self._list.addItem(item)

    def _on_sel_changed(self):
        has = bool(self._list.selectedItems())
        self._load_btn.setEnabled(has)
        self._del_btn.setEnabled(has)
        if has:
            path = self._list.currentItem().data(Qt.ItemDataRole.UserRole)
            try:
                self._selected = self._sm.load(path)
            except (OSError, ValueError):
                self._selected = None
                self._load_btn.setEnabled(False)
        else:
            self._selected = None

    def _delete_selected(self):
        item = self._list.currentItem()
        if not item:
            return
        ans = QMessageBox.question(
            self, "Delete Session",
            "Permanently delete this session?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ans != QMessageBox.StandardButton.Yes:
            return
        self._sm.delete(item.data(Qt.ItemDataRole.UserRole))
        self._load_list()

    def _on_new(self):
        self._new_requested = True
        self.accept()

    def selected_session(self) -> dict | None:
        return self._selected

    def new_requested(self) -> bool:
        return self._new_requested


# ── Main Window ────────────────────────────────────────────────────────────────

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
        self._rag_index = None            # cached VectorIndex for the loaded files
        self._session_created = None      # stable id for the active session's file

        # Streaming state
        self._stream_cursor = None   # QTextCursor parked just before the ▌ marker
        self._stream_text = ""

        # Summarize results for current session
        self._summarize_results: dict = {}   # path_str -> summary_text

        # Session management
        self._session_manager = SessionManager()

        self._build_ui()
        self._fetch_models()

    # ── layout ────────────────────────────────────────────────────────────────────

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
        self._sidebar.embed_model_changed.connect(self._on_embed_model_changed)
        body_layout.addWidget(self._sidebar)

        body_layout.addWidget(self._build_chat_panel(), 1)
        root.addWidget(body, 1)

        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)

        # Elapsed-time readout + animated "busy" indicator, pinned to the right
        # of the status bar and shown only while a worker runs, so the app never
        # looks frozen during model thinking/indexing.
        self._timer_label = QLabel()
        self._timer_label.setStyleSheet("font-size: 10px; color: #64748b; padding-right: 4px;")
        self._timer_label.hide()
        self._status_bar.addPermanentWidget(self._timer_label)

        self._busy_bar = QProgressBar()
        self._busy_bar.setRange(0, 0)
        self._busy_bar.setMaximumWidth(120)
        self._busy_bar.setFixedHeight(14)
        self._busy_bar.setTextVisible(False)
        self._busy_bar.hide()
        self._status_bar.addPermanentWidget(self._busy_bar)

        # Ticks the elapsed-time label ~10x/sec while busy.
        self._busy_elapsed = QElapsedTimer()
        self._busy_timer = QTimer(self)
        self._busy_timer.setInterval(100)
        self._busy_timer.timeout.connect(self._tick_busy)

        self._status_bar.showMessage("Ready")

    def _build_top_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(44)
        bar.setStyleSheet("background: #ffffff; border-bottom: 1px solid #e2e8f0;")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 0, 16, 0)

        title = QLabel("⬡ LocalFileAgent")
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
        self._chat_history.setPlaceholderText("Add files on the left, then start chatting…")
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
        self._save_summaries_btn = QPushButton("Save…")
        self._save_summaries_btn.setStyleSheet("font-size: 10px;")
        self._save_summaries_btn.clicked.connect(self._save_summaries_to_file)
        self._save_summaries_btn.setEnabled(False)
        strip_layout.addWidget(self._save_summaries_btn)
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
        self._chat_input.setPlaceholderText("Ask a question about the loaded files…")
        self._chat_input.returnPressed.connect(self._send_chat)
        input_row.addWidget(self._chat_input, 1)
        self._send_btn = QPushButton("Send")
        self._send_btn.setMinimumWidth(80)
        self._send_btn.clicked.connect(self._send_chat)
        input_row.addWidget(self._send_btn)
        layout.addLayout(input_row)

        return panel

    # ── model fetch ───────────────────────────────────────────────────────────────

    def _fetch_models(self):
        self._set_busy("Connecting to Ollama…")
        self._model_worker = ModelFetchWorker()
        self._model_worker.models_ready.connect(self._on_models_ready)
        self._model_worker.error.connect(self._on_model_error)
        self._model_worker.start()

    def _on_models_ready(self, models: list):
        self._sidebar.set_model_list(models)
        self._clear_busy(f"Ollama connected — {len(models)} model(s) available")

    def _on_model_error(self, _msg: str):
        self._clear_busy("⚠  Ollama not reachable — start it with:  ollama serve")

    # ── sidebar signals ───────────────────────────────────────────────────────────

    def _on_files_changed(self):
        self._chat_files_loaded = False
        self._rag_index = None
        self._rebuild_summarize_strip()

    def _on_model_changed(self, _new_model: str):
        if self._chat_files_loaded:
            self._chat_files_loaded = False
            self._append_system("Model changed — context will reload on next message.")

    def _on_embed_model_changed(self, _new_model: str):
        # Always drop the index (it's keyed by embed model); the flag reset
        # below is a no-op unless a conversation had already loaded files.
        self._rag_index = None
        if self._chat_files_loaded:
            self._chat_files_loaded = False
            self._append_system("Embed model changed — files will re-index on next message.")

    # ── chat helpers ──────────────────────────────────────────────────────────────

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
        """Append an empty assistant bubble and park a cursor just before the ▌
        so streamed tokens (which may contain newlines) insert in the right spot."""
        escaped = html.escape(model_name)
        self._chat_history.append(
            f'<p><b><span style="color:#16a34a">{escaped}:</span></b> ▌</p>'
        )
        cursor = self._chat_history.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.movePosition(QTextCursor.MoveOperation.PreviousCharacter)
        self._stream_cursor = cursor
        self._stream_text = ""
        sb = self._chat_history.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_token_ready(self, token: str):
        """Insert a token just before the ▌ marker via the parked cursor.

        A persistent cursor (rather than a cached block number) keeps insertion
        correct even when a token contains newlines: QTextCursor.insertText turns
        '\\n' into new blocks, which would strand a block-number-based cursor and
        scramble the text.
        """
        if self._stream_cursor is None:
            return
        if self._stream_text == "":
            # First token has arrived — the model is no longer just thinking.
            self._status_bar.showMessage("Responding…")
        self._stream_cursor.insertText(token)
        self._stream_text += token
        sb = self._chat_history.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _finish_stream(self):
        """Remove the trailing ▌ marker (it sits just after the parked cursor)."""
        if self._stream_cursor is None:
            return
        self._stream_cursor.movePosition(
            QTextCursor.MoveOperation.NextCharacter,
            QTextCursor.MoveMode.KeepAnchor,
        )
        self._stream_cursor.removeSelectedText()
        self._stream_cursor = None

    def _set_chat_input_enabled(self, enabled: bool):
        self._send_btn.setEnabled(enabled)
        self._chat_input.setEnabled(enabled)

    def _set_busy(self, message: str):
        """Show the animated busy indicator + elapsed timer with a status message."""
        self._status_bar.showMessage(message)
        self._busy_bar.show()
        self._busy_elapsed.restart()
        self._timer_label.setText(self._format_elapsed(0))
        self._timer_label.show()
        self._busy_timer.start()

    def _clear_busy(self, message: str = "Ready"):
        """Hide the busy indicator + timer and reset the status message."""
        self._busy_timer.stop()
        self._timer_label.hide()
        self._busy_bar.hide()
        self._status_bar.showMessage(message)

    def _tick_busy(self):
        self._timer_label.setText(self._format_elapsed(self._busy_elapsed.elapsed()))

    @staticmethod
    def _format_elapsed(ms: int) -> str:
        """Render elapsed milliseconds as '3.4s' under a minute, else 'm:ss'."""
        seconds = ms / 1000
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes, secs = divmod(int(seconds), 60)
        return f"{minutes}:{secs:02d}"

    # -- worker lifecycle --

    def _operation_in_progress(self) -> bool:
        """True while a chat or summarize worker is still running, so we never
        start a second one on top of it — that would race the busy indicator and
        drop the reference to a live QThread (crash on GC)."""
        for attr in ("_chat_worker", "_summarize_worker"):
            worker = getattr(self, attr, None)
            if worker is None:
                continue
            try:
                if worker.isRunning():
                    return True
            except (RuntimeError, AttributeError):
                continue
        return False

    def _shutdown_workers(self):
        """Interrupt and wait for any running worker threads, so closing the
        window never destroys a QThread mid-run."""
        for attr in ("_model_worker", "_chat_worker", "_summarize_worker"):
            worker = getattr(self, attr, None)
            if worker is None:
                continue
            try:
                running = worker.isRunning()
            except (RuntimeError, AttributeError):
                continue
            if running:
                worker.requestInterruption()
                worker.wait(3000)

    def closeEvent(self, event):
        self._shutdown_workers()
        super().closeEvent(event)

    # -- chat send and reply handlers --

    def _send_chat(self):
        user_text = self._chat_input.text().strip()
        if not user_text:
            return

        model = self._sidebar.model()
        if not model:
            QMessageBox.warning(self, "No Model", "Please select or enter a model name.")
            return

        if self._operation_in_progress():
            return   # a worker is still running; ignore until it finishes

        self._chat_input.clear()
        self._set_chat_input_enabled(False)

        generation = self._chat_generation

        embed_model = self._sidebar.embed_model() or DEFAULT_EMBED_MODEL

        if not self._chat_files_loaded:
            valid_paths = self._sidebar.get_valid_paths()
            if not valid_paths:
                QMessageBox.warning(self, "No Files", "No accessible files. Check for deleted files in the sidebar.")
                self._set_chat_input_enabled(True)
                return

            files = [Path(p) for p in valid_paths[:CONTEXT_FILE_CAP]]
            if len(valid_paths) > CONTEXT_FILE_CAP:
                self._append_system(
                    f"⚠  Only first {CONTEXT_FILE_CAP} of {len(valid_paths)} files loaded (token limit)."
                )
            self._append_system(f"Indexing {len(files)} file(s)…")
            self._append_chat("You", user_text, "#3b82f6")

            self._chat_worker = StreamingChatWorker(
                [], model, files_to_load=files, user_text=user_text,
                embed_model=embed_model,
            )
            self._chat_worker.context_info.connect(self._on_context_info)
            self._chat_worker.index_ready.connect(self._on_index_ready)
            self._chat_worker.file_status.connect(
                lambda p, s, t: self._sidebar.set_file_status(p, s, t)
            )
            busy_msg = "Indexing…"
        else:
            self._chat_messages.append({"role": "user", "content": user_text})
            self._append_chat("You", user_text, "#3b82f6")
            valid_paths = self._sidebar.get_valid_paths()
            if self._rag_index is None and valid_paths:
                # Resumed session: the index wasn't rebuilt on load. Rebuild it on
                # this turn while keeping the replayed history, so retrieval fires
                # instead of sending a contextless RAG prompt.
                files = [Path(p) for p in valid_paths[:CONTEXT_FILE_CAP]]
                self._append_system(f"Re-indexing {len(files)} file(s)…")
                self._chat_worker = StreamingChatWorker(
                    list(self._chat_messages), model,
                    files_to_load=files, user_text=user_text,
                    preserve_history=True, embed_model=embed_model,
                )
                self._chat_worker.context_info.connect(self._on_context_info)
                self._chat_worker.index_ready.connect(self._on_index_ready)
                self._chat_worker.file_status.connect(
                    lambda p, s, t: self._sidebar.set_file_status(p, s, t)
                )
                busy_msg = "Indexing…"
            else:
                self._chat_worker = StreamingChatWorker(
                    list(self._chat_messages), model,
                    rag_index=self._rag_index, embed_model=embed_model,
                    user_text=user_text,
                )
                busy_msg = "Thinking…"

        self._last_elapsed = 0.0
        self._start_stream_bubble(model)
        self._chat_worker.token_ready.connect(self._on_token_ready)
        self._chat_worker.timing.connect(lambda t: setattr(self, "_last_elapsed", t))
        self._chat_worker.finished.connect(
            lambda msgs, g=generation: self._on_chat_reply(msgs, g)
        )
        self._chat_worker.error.connect(self._on_chat_error)
        self._set_busy(busy_msg)
        self._chat_worker.start()

    def _on_context_info(self, info: str):
        self._append_system(info)
        self._chat_files_loaded = True

    def _on_index_ready(self, index):
        self._rag_index = index

    def _on_chat_reply(self, updated_messages: list, generation: int):
        self._finish_stream()
        self._clear_busy()
        if generation != self._chat_generation:
            self._set_chat_input_enabled(True)
            return
        self._chat_messages = updated_messages
        self._append_system(f"⏱  {self._last_elapsed:.1f}s")
        self._set_chat_input_enabled(True)
        self._chat_input.setFocus()
        self._auto_save()

    def _on_chat_error(self, msg: str):
        self._finish_stream()
        self._clear_busy()
        if self._stream_text:
            self._append_system("(response interrupted)")
        # If the very first (indexing) turn failed, no conversation was
        # established — drop the half-built state so the next message rebuilds
        # cleanly with the system prompt instead of sending a bare user turn.
        if not self._chat_messages:
            self._chat_files_loaded = False
            self._rag_index = None
        self._set_chat_input_enabled(True)
        QMessageBox.critical(self, "Ollama Error", msg)

    def _auto_save(self):
        session = {
            "model":       self._sidebar.model(),
            "embed_model": self._sidebar.embed_model(),
            "files":       self._sidebar.get_paths(),
            "messages":    self._chat_messages,
            "summaries":   self._summarize_results,
        }
        # Reuse this session's identity so every reply updates ONE file rather
        # than spawning a new file per turn (the filename derives from 'created').
        if self._session_created:
            session["created"] = self._session_created
        try:
            self._session_manager.save(session)
        except OSError:
            return   # non-fatal; don't interrupt the user
        self._session_created = session["created"]

    def _open_sessions(self):
        dlg = SessionDialog(self._session_manager, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        if dlg.new_requested():
            self._new_session()
        else:
            session = dlg.selected_session()
            if session:
                self._load_session(session)

    def _new_session(self):
        self._chat_history.clear()
        self._chat_messages = []
        self._rag_index = None
        self._session_created = None
        self._chat_files_loaded = False
        self._chat_generation += 1
        self._summarize_results = {}
        self._sidebar.clear_files()
        self._rebuild_summarize_strip()
        self._copy_all_btn.setEnabled(False)
        self._save_summaries_btn.setEnabled(False)
        self._status_bar.showMessage("New session started.")

    def _load_session(self, session: dict):
        self._new_session()

        model_name = session.get("model", DEFAULT_MODEL)
        idx = self._sidebar._model_combo.findText(model_name)
        if idx >= 0:
            self._sidebar._model_combo.setCurrentIndex(idx)

        embed_name = session.get("embed_model", DEFAULT_EMBED_MODEL)
        eidx = self._sidebar._embed_combo.findText(embed_name)
        if eidx >= 0:
            self._sidebar._embed_combo.setCurrentIndex(eidx)
        else:
            self._sidebar._embed_combo.setEditText(embed_name)

        self._sidebar.populate_from_paths(session.get("files", []))
        self._chat_messages = session.get("messages", [])
        self._summarize_results = session.get("summaries", {})
        # Continue writing to the loaded session's file, not a fresh one.
        self._session_created = session.get("created")

        # Replay visible history from messages (skip system prompt)
        for msg in self._chat_messages:
            role = msg.get("role")
            if role == "user":
                self._append_chat("You", msg.get("content", ""), "#3b82f6")
            elif role == "assistant":
                model = session.get("model", DEFAULT_MODEL)
                self._append_chat(model, msg.get("content", ""), "#16a34a")

        for path_str, summary in self._summarize_results.items():
            self._append_system(f"\u2500\u2500 Summary: {Path(path_str).name} \u2500\u2500")
            self._append_chat("Summary", summary, "#7c3aed")

        if self._chat_messages:
            self._chat_files_loaded = True
        self._rebuild_summarize_strip()
        self._copy_all_btn.setEnabled(bool(self._summarize_results))
        self._save_summaries_btn.setEnabled(bool(self._summarize_results))
        self._status_bar.showMessage(f"Session loaded \u2014 {len(self._sidebar.get_paths())} file(s).")

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
        if self._operation_in_progress():
            return
        model = self._sidebar.model()
        self._set_chat_input_enabled(False)
        self._append_system(f"Summarizing {Path(path_str).name}…")
        self._set_busy(f"Summarizing {Path(path_str).name}…")

        self._summarize_worker = SummarizeWorker([Path(path_str)], model)
        self._summarize_worker.file_done.connect(self._on_summarize_done)
        self._summarize_worker.finished.connect(self._on_summarize_finished)
        self._summarize_worker.error.connect(self._on_summarize_error)
        self._summarize_worker.start()

    def _on_summarize_finished(self):
        self._set_chat_input_enabled(True)
        self._clear_busy()

    def _on_summarize_done(self, path_str: str, summary: str, elapsed: float):
        self._summarize_results[path_str] = summary
        name = Path(path_str).name
        timing = f"  ⏱  {elapsed:.1f}s" if elapsed > 0 else ""
        self._append_system(f"── Summary: {name} ──{timing}")
        self._append_chat("Summary", summary, "#7c3aed")
        self._copy_all_btn.setEnabled(bool(self._summarize_results))
        self._save_summaries_btn.setEnabled(bool(self._summarize_results))
        self._auto_save()

    def _on_summarize_error(self, msg: str):
        self._set_chat_input_enabled(True)
        self._clear_busy()
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

    def _save_summaries_to_file(self):
        if not self._summarize_results:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Summaries", "summaries.md",
            "Markdown (*.md);;Plain Text (*.txt);;All Files (*)",
        )
        if not path:
            return
        output = Path(path)
        lines = ["# File Summaries\n"]
        for path_str, summary in self._summarize_results.items():
            p = Path(path_str)
            lines += [f"## `{p.name}`", "", f"**Path:** `{path_str}`", "", summary, ""]
        try:
            output.write_text("\n".join(lines), encoding="utf-8")
        except OSError as exc:
            QMessageBox.critical(self, "Save Error", f"Could not write file:\n{exc}")
            return
        self._status_bar.showMessage(f"Summaries saved to {str(output)}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
