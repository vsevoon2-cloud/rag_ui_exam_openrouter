import json
import faulthandler
import json
import sys
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from .config import AppConfig, load_config, save_config, data_dir, ensure_dirs
from .index_builder import build_index_from_pdf
from .openrouter_client import OpenRouterClient
from .retrieval import LocalIndex


class AreaSelectionDialog(QtWidgets.QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowFlags(
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.Dialog
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
        )
        self.setWindowState(QtCore.Qt.WindowState.WindowFullScreen)
        self.setCursor(QtCore.Qt.CursorShape.CrossCursor)
        self.setModal(True)

        self._start = QtCore.QPoint()
        self._selection = QtCore.QRect()
        self._rubber_band = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Shape.Rectangle, self)

        # Capture the full virtual desktop (all monitors) BEFORE showing the dialog.
        screens = QtGui.QGuiApplication.screens()
        if not screens:
            self._virtual_geom = QtCore.QRect()
            self._background = QtGui.QPixmap()
            return

        virtual = screens[0].virtualGeometry()
        self._virtual_geom = QtCore.QRect(virtual)
        self.setGeometry(self._virtual_geom)
        self.move(self._virtual_geom.topLeft())

        composed = QtGui.QPixmap(self._virtual_geom.size())
        composed.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(composed)
        for s in screens:
            g = s.geometry()
            shot = s.grabWindow(0)
            # Map screen-local pixels into virtual desktop coordinates.
            x = g.left() - self._virtual_geom.left()
            y = g.top() - self._virtual_geom.top()
            painter.drawPixmap(x, y, shot)
        painter.end()
        self._background = composed

    def selected_rect(self) -> QtCore.QRect:
        return self._selection.normalized()

    def background_pixmap(self) -> QtGui.QPixmap:
        return self._background

    def virtual_geometry(self) -> QtCore.QRect:
        return self._virtual_geom

    def paintEvent(self, event) -> None:
        painter = QtGui.QPainter(self)
        painter.drawPixmap(0, 0, self._background)
        painter.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 80))

    def mousePressEvent(self, event) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._start = event.pos()
            self._rubber_band.setGeometry(QtCore.QRect(self._start, QtCore.QSize()))
            self._rubber_band.show()

    def mouseMoveEvent(self, event) -> None:
        if self._rubber_band.isVisible():
            self._rubber_band.setGeometry(QtCore.QRect(self._start, event.pos()).normalized())

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._selection = self._rubber_band.geometry().normalized()
            self._rubber_band.hide()
            self.accept()

    def keyPressEvent(self, event) -> None:
        if event.key() == QtCore.Qt.Key.Key_Escape:
            self.reject()


@dataclass
class IndexState:
    index_dir: Path | None = None
    collection: str = "examrag"
    embed_model: str = "intfloat/multilingual-e5-large"
    chunking_mode: str = "page"


class AnswerOverlay(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__(None)
        self.setWindowFlags(
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.Tool
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

        container = QtWidgets.QFrame(self)
        container.setStyleSheet(
            "QFrame { background-color: rgba(10, 10, 10, 140); border: 1px solid rgba(255,255,255,50); border-radius: 14px; }"
            "QLabel { color: white; font-size: 16px; }"
        )
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(container)
        inner = QtWidgets.QVBoxLayout(container)
        inner.setContentsMargins(16, 14, 16, 14)
        self.label = QtWidgets.QLabel("")
        self.label.setWordWrap(True)
        self.label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        inner.addWidget(self.label)
        self.resize(560, 220)
        self._drag_active = False
        self._drag_offset = QtCore.QPoint()
        self._loading_timer: QtCore.QTimer | None = None
        self._loading_base: str = ""
        self._loading_tick: int = 0

    def show_message(self, text: str) -> None:
        self.stop_loading()
        self.label.setText(text.strip())
        self._fit_to_text()
        # If user never moved it, place it in the top-right; otherwise keep last position.
        if not getattr(self, "_user_moved", False):
            screen = QtWidgets.QApplication.primaryScreen()
            if screen:
                geom = screen.availableGeometry()
                self.move(geom.right() - self.width() - 28, geom.top() + 28)
        self.show()
        self.raise_()

    def clear_message(self) -> None:
        self.stop_loading()
        self.hide()

    def start_loading(self, base_text: str = "Загрузка") -> None:
        self._loading_base = (base_text or "Загрузка").strip()
        self._loading_tick = 0
        if self._loading_timer is None:
            self._loading_timer = QtCore.QTimer(self)
            self._loading_timer.setInterval(180)
            self._loading_timer.timeout.connect(self._on_loading_tick)
        if not self._loading_timer.isActive():
            self._loading_timer.start()
        self._on_loading_tick()
        self._fit_to_text()
        if not getattr(self, "_user_moved", False):
            screen = QtWidgets.QApplication.primaryScreen()
            if screen:
                geom = screen.availableGeometry()
                self.move(geom.right() - self.width() - 28, geom.top() + 28)
        self.show()
        self.raise_()

    def stop_loading(self) -> None:
        if self._loading_timer is not None and self._loading_timer.isActive():
            self._loading_timer.stop()

    def _on_loading_tick(self) -> None:
        self._loading_tick = (self._loading_tick + 1) % 4
        dots = "." * self._loading_tick
        self.label.setText(f"{self._loading_base}{dots}")
        self._fit_to_text()

    def _fit_to_text(self) -> None:
        # Keep overlay readable: grow height to fit content (up to a limit).
        try:
            screen = QtWidgets.QApplication.primaryScreen()
            if not screen:
                return
            geom = screen.availableGeometry()
            max_w = min(self.width(), 820)
            max_h = min(geom.height() - 40, 520)

            self.label.setFixedWidth(max_w - 32)
            hint = self.label.sizeHint()
            target_h = max(80, min(max_h, hint.height() + 32))
            self.resize(max_w, target_h)
        except Exception:
            return

    def set_preset_size(self, preset: str) -> None:
        preset = (preset or "").lower()
        if preset == "small":
            # Half-size variant (requested): keep it compact for unobtrusive overlay.
            self.resize(210, 80)
        elif preset == "large":
            self.resize(820, 360)
        else:
            self.resize(560, 220)

    def mousePressEvent(self, event) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._drag_active = True
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            self._user_moved = True
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_active:
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._drag_active = False
            event.accept()
            return
        super().mouseReleaseEvent(event)


class MainWindow(QtWidgets.QMainWindow):
    ui_call = QtCore.Signal(object)

    def __init__(self) -> None:
        super().__init__()
        ensure_dirs()
        self.cfg: AppConfig = load_config()
        self.state = IndexState(embed_model=self.cfg.embed_model, chunking_mode=self.cfg.chunking_mode)
        self.index: LocalIndex | None = None
        self.capture_image: bytes | None = None
        self._capture_busy = False
        self._index_loading = False
        self.overlay = AnswerOverlay()

        self.setWindowTitle("ExamRAG")
        # Don't steal focus when showing after a capture; overlay provides feedback.
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.resize(1200, 800)

        self._build_ui()
        self._setup_tray()
        self.ui_call.connect(self._dispatch_ui_call)
        self._setup_hotkey()

    @QtCore.Slot(object)
    def _dispatch_ui_call(self, fn) -> None:
        fn()

    def _setup_hotkey(self) -> None:
        """Устанавливает глобальный хоткей Ctrl+Shift+X"""
        try:
            import keyboard
            keyboard.add_hotkey('ctrl+shift+x', self._on_hotkey)
            self.capture_status.setText("✓ Hotkey Ctrl+Shift+X ready")
        except ImportError:
            self.capture_status.setText("⚠️ Hotkey not available (pip install keyboard)")
        except Exception as e:
            self.capture_status.setText(f"⚠️ Hotkey error: {e}")

    def _on_hotkey(self) -> None:
        """Вызывается по хоткею"""
        self.ui_call.emit(lambda: self.status.setText("Hotkey pressed"))
        self.ui_call.emit(self._capture_screen)

    def _setup_tray(self) -> None:
        self.tray = None
        if not QtWidgets.QSystemTrayIcon.isSystemTrayAvailable():
            return
        icon = self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_ComputerIcon)
        self.tray = QtWidgets.QSystemTrayIcon(icon, self)
        self.tray.setToolTip("ExamRAG")
        menu = QtWidgets.QMenu(self)
        capture_action = menu.addAction("Capture")
        capture_action.triggered.connect(self._capture_screen)
        restore_action = menu.addAction("Show")
        restore_action.triggered.connect(self._restore_from_tray)
        quit_action = menu.addAction("Quit")
        quit_action.triggered.connect(QtWidgets.QApplication.quit)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._on_tray_activated)
        self.tray.show()

    def _hide_to_tray(self) -> None:
        self.hide()
        if self.tray is not None:
            self.tray.show()

    def _restore_from_tray(self) -> None:
        self.show()
        self.raise_()
        self.activateWindow()

    def _on_tray_activated(self, reason) -> None:
        if reason == QtWidgets.QSystemTrayIcon.ActivationReason.DoubleClick:
            self._restore_from_tray()

    def closeEvent(self, event) -> None:
        if self.tray is not None and self.tray.isVisible():
            event.ignore()
            self._hide_to_tray()
            return
        super().closeEvent(event)

    def _build_ui(self) -> None:
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        layout = QtWidgets.QVBoxLayout(root)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        self.api_key = QtWidgets.QLineEdit(self.cfg.openrouter_api_key)
        self.api_key.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self.ocr_model = QtWidgets.QComboBox()
        self.ocr_model.addItems(["google/gemini-2.0-flash-001", "google/gemini-2.5-flash-image", "google/gemini-2.5-pro"])
        self.ocr_model.setCurrentText(self.cfg.ocr_model)
        self.answer_model = QtWidgets.QComboBox()
        self.answer_model.addItems(["google/gemini-2.5-pro", "google/gemini-2.5-flash", "google/gemini-2.0-flash-001"])
        self.answer_model.setCurrentText(self.cfg.answer_model)
        self.embed_model = QtWidgets.QComboBox()
        self.embed_model.setEditable(True)
        self.embed_model.addItems(
            [
                "intfloat/multilingual-e5-large",
                "intfloat/multilingual-e5-base",
                "BAAI/bge-m3",
            ]
        )
        self.embed_model.setCurrentText(self.cfg.embed_model)
        self.chunking_mode = QtWidgets.QComboBox()
        self.chunking_mode.addItems(["page", "chars"])
        self.chunking_mode.setCurrentText(self.cfg.chunking_mode)
        self.retrieval_mode = QtWidgets.QComboBox()
        self.retrieval_mode.addItems(["hybrid", "vector"])
        self.retrieval_mode.setCurrentText(self.cfg.retrieval_mode)
        self.retry_x2 = QtWidgets.QCheckBox("x2")
        self.retry_x3 = QtWidgets.QCheckBox("x3")
        retry_count = str(getattr(self.cfg, "retrieval_retry_count", "1") or "1")
        self.retry_x2.setChecked(retry_count == "2")
        self.retry_x3.setChecked(retry_count == "3")
        self.retry_x2.toggled.connect(lambda checked: self._on_retry_toggle("2", checked))
        self.retry_x3.toggled.connect(lambda checked: self._on_retry_toggle("3", checked))
        retry_widget = QtWidgets.QWidget()
        retry_layout = QtWidgets.QHBoxLayout(retry_widget)
        retry_layout.setContentsMargins(0, 0, 0, 0)
        retry_layout.addWidget(self.retry_x2)
        retry_layout.addWidget(self.retry_x3)
        retry_layout.addStretch()
        self.reverify_count = QtWidgets.QSpinBox()
        self.reverify_count.setRange(1, 5)
        self.reverify_count.setValue(int(self.cfg.reverify_count) if str(self.cfg.reverify_count).isdigit() else 1)
        self.reverify_count.setSuffix(" times")
        self.calls_mode = QtWidgets.QComboBox()
        self.calls_mode.addItems(["two_step", "single_step"])
        self.calls_mode.setCurrentText(self.cfg.ocr_calls_mode)
        self.ocr_max_tokens = QtWidgets.QSpinBox()
        self.ocr_max_tokens.setRange(200, 8000)
        self.ocr_max_tokens.setValue(int(self.cfg.ocr_max_tokens) if str(self.cfg.ocr_max_tokens).isdigit() else 1400)
        self.answer_max_tokens = QtWidgets.QSpinBox()
        self.answer_max_tokens.setRange(200, 8000)
        self.answer_max_tokens.setValue(int(self.cfg.answer_max_tokens) if str(self.cfg.answer_max_tokens).isdigit() else 2000)

        self.save_btn = QtWidgets.QPushButton("Save Settings")
        self.save_btn.setObjectName("PrimaryButton")
        self.save_btn.clicked.connect(self._save_settings)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        layout.addWidget(self.tabs, 1)

        quick_tab = QtWidgets.QWidget()
        quick_layout = QtWidgets.QVBoxLayout(quick_tab)
        quick_layout.setContentsMargins(0, 0, 0, 0)
        quick_layout.setSpacing(12)

        library_card = self._create_card("Library")
        library_layout = QtWidgets.QVBoxLayout(library_card)
        library_layout.setSpacing(10)
        self.index_path_label = QtWidgets.QLabel("Index folder: not selected")
        self.index_path_label.setWordWrap(True)
        self.index_path_label.setMinimumWidth(420)
        browse = QtWidgets.QPushButton("Add PDF…")
        browse.clicked.connect(self._pick_pdf)
        browse.setEnabled(False)
        build = QtWidgets.QPushButton("Build Index")
        build.clicked.connect(self._build_index)
        build.setEnabled(False)
        self.load_index_btn = QtWidgets.QPushButton("📂 Load Index from Folder")
        self.load_index_btn.setObjectName("PrimaryButton")
        self.load_index_btn.clicked.connect(self._load_index_from_folder)
        library_actions = QtWidgets.QHBoxLayout()
        library_actions.addWidget(self.index_path_label, 1)
        library_actions.addWidget(self.load_index_btn)
        library_layout.addLayout(library_actions)
        quick_layout.addWidget(library_card)

        capture_card = self._create_card("Screen Capture")
        capture_layout = QtWidgets.QVBoxLayout(capture_card)
        capture_layout.setSpacing(10)
        self.capture_btn = QtWidgets.QPushButton("📸 Capture (Ctrl+Shift+X)")
        self.capture_btn.setObjectName("CaptureButton")
        self.capture_btn.clicked.connect(self._capture_screen)
        self.capture_status = QtWidgets.QLabel("Ready")
        self.capture_btn.setMinimumHeight(52)
        capture_layout.addWidget(self.capture_btn)
        capture_layout.addWidget(self.capture_status)
        quick_layout.addWidget(capture_card)

        workspace_card = self._create_card("Workspace")
        workspace_layout = QtWidgets.QVBoxLayout(workspace_card)
        workspace_layout.setSpacing(10)
        overlay_layout = QtWidgets.QHBoxLayout()
        overlay_layout.addWidget(QtWidgets.QLabel("Overlay size"))
        self.overlay_size = QtWidgets.QComboBox()
        self.overlay_size.addItems(["Medium", "Small", "Large"])
        self.overlay_size.setCurrentText("Medium")
        self.overlay_size.currentTextChanged.connect(self._on_overlay_size_changed)
        overlay_layout.addWidget(self.overlay_size)
        overlay_layout.addStretch()
        workspace_layout.addLayout(overlay_layout)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.question = QtWidgets.QPlainTextEdit()
        self.question.setPlaceholderText("Question will appear here (or enter manually)")
        self.answer = QtWidgets.QPlainTextEdit()
        self.answer.setPlaceholderText("Answer will appear here")
        self.sources = QtWidgets.QPlainTextEdit()
        self.sources.setPlaceholderText("Sources from textbook will appear here")
        splitter.addWidget(self.question)
        splitter.addWidget(self.answer)
        splitter.addWidget(self.sources)
        splitter.setSizes([160, 280, 260])
        workspace_layout.addWidget(splitter, 1)
        btn_layout = QtWidgets.QHBoxLayout()
        self.ask_btn = QtWidgets.QPushButton("🔍 Ask (manual question)")
        self.ask_btn.setObjectName("DarkButton")
        self.ask_btn.clicked.connect(self._on_ask_clicked)
        btn_layout.addStretch()
        btn_layout.addWidget(self.ask_btn)
        workspace_layout.addLayout(btn_layout)
        quick_layout.addWidget(workspace_card, 1)
        self.tabs.addTab(quick_tab, "Quick Start")

        settings_tab = QtWidgets.QWidget()
        settings_layout = QtWidgets.QVBoxLayout(settings_tab)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.setSpacing(12)

        api_card = self._create_card("API")
        api_form = QtWidgets.QFormLayout(api_card)
        api_form.setContentsMargins(16, 16, 16, 16)
        api_form.addRow("OpenRouter Key", self.api_key)
        settings_layout.addWidget(api_card)

        models_card = self._create_card("Models")
        models_form = QtWidgets.QFormLayout(models_card)
        models_form.setContentsMargins(16, 16, 16, 16)
        models_form.addRow("OCR", self.ocr_model)
        models_form.addRow("Answer", self.answer_model)
        models_form.addRow("Embed", self.embed_model)
        settings_layout.addWidget(models_card)

        quality_card = self._create_card("Quality Control")
        quality_form = QtWidgets.QFormLayout(quality_card)
        quality_form.setContentsMargins(16, 16, 16, 16)
        quality_form.addRow("Retry reformulate", retry_widget)
        quality_form.addRow("Re-verify answer", self.reverify_count)
        settings_layout.addWidget(quality_card)
        settings_layout.addWidget(self.save_btn, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        settings_layout.addStretch(1)
        self.tabs.addTab(settings_tab, "Settings")

        advanced_tab = QtWidgets.QWidget()
        advanced_layout = QtWidgets.QVBoxLayout(advanced_tab)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        advanced_layout.setSpacing(12)

        processing_card = self._create_card("Processing")
        processing_form = QtWidgets.QFormLayout(processing_card)
        processing_form.setContentsMargins(16, 16, 16, 16)
        processing_form.addRow("Chunking", self.chunking_mode)
        processing_form.addRow("Retrieval", self.retrieval_mode)
        processing_form.addRow("OCR calls", self.calls_mode)
        advanced_layout.addWidget(processing_card)

        tokens_card = self._create_card("Token Limits")
        tokens_form = QtWidgets.QFormLayout(tokens_card)
        tokens_form.setContentsMargins(16, 16, 16, 16)
        tokens_form.addRow("OCR max", self.ocr_max_tokens)
        tokens_form.addRow("Answer max", self.answer_max_tokens)
        advanced_layout.addWidget(tokens_card)

        build_card = self._create_card("Index Build")
        build_layout = QtWidgets.QVBoxLayout(build_card)
        build_layout.setSpacing(10)
        build_layout.addWidget(QtWidgets.QLabel("Optional local index build tools"))
        build_actions = QtWidgets.QHBoxLayout()
        build_actions.addWidget(browse)
        build_actions.addWidget(build)
        build_actions.addStretch()
        build_layout.addLayout(build_actions)
        advanced_layout.addWidget(build_card)
        advanced_layout.addStretch(1)
        self.tabs.addTab(advanced_tab, "Advanced")

        status_card = self._create_card("Status")
        status_layout = QtWidgets.QVBoxLayout(status_card)
        status_layout.setSpacing(8)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.status = QtWidgets.QLabel("Ready")
        status_layout.addWidget(self.progress)
        status_layout.addWidget(self.status)
        layout.addWidget(status_card)
        self._apply_styles()

    def _create_card(self, title: str) -> QtWidgets.QGroupBox:
        card = QtWidgets.QGroupBox(title)
        card.setObjectName("Card")
        return card

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #ffffff;
                color: #212529;
                font-family: "Inter", "Segoe UI";
                font-size: 13px;
            }
            QTabWidget::pane {
                border: 1px solid #e9ecef;
                border-radius: 8px;
                background: #ffffff;
                top: -1px;
            }
            QTabBar::tab {
                background: #f1f3f5;
                border: 1px solid #dee2e6;
                padding: 10px 18px;
                margin-right: 6px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                min-width: 120px;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                border-bottom-color: #ffffff;
                color: #0d6efd;
                font-weight: 600;
            }
            QGroupBox#Card {
                background: #f8f9fa;
                border: 1px solid #e9ecef;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: 600;
            }
            QGroupBox#Card::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px;
            }
            QLineEdit, QComboBox, QSpinBox, QPlainTextEdit {
                background: #ffffff;
                border: 1px solid #dee2e6;
                border-radius: 6px;
                padding: 7px 10px;
                selection-background-color: #cfe2ff;
            }
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QPlainTextEdit:focus {
                border: 1px solid #0d6efd;
            }
            QPlainTextEdit {
                padding: 10px;
            }
            QPushButton {
                background: #ffffff;
                color: #212529;
                border: 1px solid #dee2e6;
                border-radius: 6px;
                padding: 8px 14px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: #f1f3f5;
            }
            QPushButton:pressed {
                background: #e9ecef;
            }
            QPushButton:disabled {
                color: #adb5bd;
                background: #f8f9fa;
            }
            QPushButton#PrimaryButton {
                background: #0d6efd;
                color: white;
                border: none;
            }
            QPushButton#PrimaryButton:hover {
                background: #0b5ed7;
            }
            QPushButton#CaptureButton {
                background: #198754;
                color: white;
                border: none;
                font-size: 15px;
                font-weight: 600;
            }
            QPushButton#CaptureButton:hover {
                background: #157347;
            }
            QPushButton#DarkButton {
                background: #212529;
                color: white;
                border: none;
            }
            QPushButton#DarkButton:hover {
                background: #343a40;
            }
            QLabel {
                background: transparent;
            }
            QProgressBar {
                border: 1px solid #dee2e6;
                border-radius: 6px;
                background: #ffffff;
                text-align: center;
                min-height: 18px;
            }
            QProgressBar::chunk {
                background: #0d6efd;
                border-radius: 5px;
            }
            QCheckBox {
                spacing: 8px;
            }
            """
        )

    def _on_overlay_size_changed(self, text: str) -> None:
        preset = (text or "Medium").strip().lower()
        if "small" in preset:
            self.overlay.set_preset_size("small")
        elif "large" in preset:
            self.overlay.set_preset_size("large")
        else:
            self.overlay.set_preset_size("medium")

    def _on_retry_toggle(self, level: str, checked: bool) -> None:
        if not checked:
            return
        if level == "2":
            self.retry_x3.blockSignals(True)
            self.retry_x3.setChecked(False)
            self.retry_x3.blockSignals(False)
        elif level == "3":
            self.retry_x2.blockSignals(True)
            self.retry_x2.setChecked(False)
            self.retry_x2.blockSignals(False)

    def _get_retrieval_retry_count(self) -> int:
        if self.retry_x3.isChecked():
            return 3
        if self.retry_x2.isChecked():
            return 2
        return 1

    def _get_reverify_count(self) -> int:
        return int(self.reverify_count.value())

    @staticmethod
    def _pick_best_comparison(comparisons: list[dict], options: dict[str, str]) -> dict:
        if not comparisons:
            return {"correct_option": "?", "confidence": 0.0, "reasoning": "", "correct_text": ""}

        buckets: dict[str, list[dict]] = {}
        for comparison in comparisons:
            option = str(comparison.get("correct_option") or "?").strip().upper()
            buckets.setdefault(option, []).append(comparison)

        ranked = sorted(
            buckets.items(),
            key=lambda item: (
                len(item[1]),
                max(float(entry.get("confidence") or 0.0) for entry in item[1]),
            ),
            reverse=True,
        )
        selected_option, selected_group = ranked[0]
        selected = max(selected_group, key=lambda entry: float(entry.get("confidence") or 0.0))
        if selected_option in options and not selected.get("correct_text"):
            selected["correct_text"] = options[selected_option]
        return selected

    def _save_settings(self) -> None:
        self.cfg.openrouter_api_key = self.api_key.text().strip()
        self.cfg.ocr_model = self.ocr_model.currentText().strip()
        self.cfg.answer_model = self.answer_model.currentText().strip()
        self.cfg.embed_model = self.embed_model.currentText().strip()
        self.cfg.chunking_mode = self.chunking_mode.currentText().strip()
        self.cfg.retrieval_mode = self.retrieval_mode.currentText().strip()
        self.cfg.retrieval_retry_count = str(self._get_retrieval_retry_count())
        self.cfg.reverify_count = str(self._get_reverify_count())
        self.cfg.ocr_calls_mode = self.calls_mode.currentText().strip()
        self.cfg.ocr_max_tokens = str(int(self.ocr_max_tokens.value()))
        self.cfg.answer_max_tokens = str(int(self.answer_max_tokens.value()))
        self.state.embed_model = self.cfg.embed_model
        self.state.chunking_mode = self.cfg.chunking_mode
        save_config(self.cfg)
        self.status.setText("Settings saved")

    def _pick_pdf(self) -> None:
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select PDF", "", "PDF files (*.pdf)")
        if not p:
            return
        self.cfg.last_pdf_path = p
        self.index_path_label.setText(f"PDF selected: {p}")

    def _build_index(self) -> None:
        pdf = Path(self.cfg.last_pdf_path.strip())
        if not pdf.exists():
            QtWidgets.QMessageBox.critical(self, "ExamRAG", "Select a valid PDF first.")
            return
        if not self.api_key.text().strip():
            QtWidgets.QMessageBox.critical(self, "ExamRAG", "OpenRouter key is empty.")
            return

        self._save_settings()
        api_key = self.api_key.text().strip()
        ocr_model = self.ocr_model.currentText().strip()
        client = OpenRouterClient(api_key)
        index_dir = data_dir() / "indexes" / pdf.stem
        self.state.index_dir = index_dir
        embed_model = self.state.embed_model
        collection = self.state.collection
        chunking_mode = self.state.chunking_mode

        def progress_cb(p):
            if p.total > 0:
                val = int(100.0 * (p.current / p.total))
                self.progress.setValue(max(0, min(100, val)))
            self.status.setText(f"{p.stage}: {p.current}/{p.total} cost=${p.cost_usd:.4f}")

        def worker():
            try:
                build_index_from_pdf(
                    pdf_path=pdf,
                    out_dir=index_dir,
                    openrouter_client=client,
                    ocr_model=ocr_model,
                    embed_model=embed_model,
                    collection=collection,
                    chunking_mode=chunking_mode,
                    target_chars=2000,
                    overlap_chars=250,
                    progress_cb=lambda p: self.ui_call.emit(lambda p=p: progress_cb(p)),
                )
                self.ui_call.emit(self._load_index)
            except Exception as e:
                msg = str(e)
                self.ui_call.emit(lambda msg=msg: QtWidgets.QMessageBox.critical(self, "ExamRAG", msg))

        threading.Thread(target=worker, daemon=True).start()

    def _load_index(self) -> None:
        if not self.state.index_dir:
            return
        if self._index_loading:
            return
        self._index_loading = True
        self.progress.setRange(0, 0)
        self.status.setText("Loading index…")
        self.capture_status.setText("Loading index…")
        index_dir = self.state.index_dir
        collection = self.state.collection
        embed_model = self.state.embed_model

        def worker() -> None:
            try:
                loaded_index = LocalIndex(index_dir, collection, embed_model)

                def finish_success() -> None:
                    self.index = loaded_index
                    self.progress.setRange(0, 100)
                    self.progress.setValue(100)
                    self.status.setText("Index loaded")
                    self.capture_status.setText("✓ Index ready")
                    self._index_loading = False

                self.ui_call.emit(finish_success)
            except Exception as e:
                msg = str(e)

                def finish_error(msg=msg) -> None:
                    self.progress.setRange(0, 100)
                    self.progress.setValue(0)
                    self.status.setText("Index load error")
                    self.capture_status.setText("Index load error")
                    self._index_loading = False
                    QtWidgets.QMessageBox.critical(self, "ExamRAG", msg)

                self.ui_call.emit(finish_error)

        threading.Thread(target=worker, daemon=True).start()

    # ========== MANUAL QUESTION ==========
    def _on_ask_clicked(self) -> None:
        """Ручной ввод вопроса"""
        question = self.question.toPlainText().strip()
        if not question:
            self.answer.setPlainText("❌ Введите вопрос в поле выше")
            return

        if not self.index:
            self.answer.setPlainText("❌ Индекс не загружен. Сначала постройте индекс из PDF.")
            return

        if not self.api_key.text().strip():
            self.answer.setPlainText("❌ Введите API ключ OpenRouter в настройках")
            return

        index = self.index
        api_key = self.api_key.text().strip()
        retrieval_mode = self.retrieval_mode.currentText().strip()
        answer_model = self.answer_model.currentText().strip()

        self.status.setText("Поиск в учебнике...")
        self.answer.setPlainText("⏳ Ищу ответ...")

        def worker():
            try:
                # 1. Поиск в индексе
                results = index.search(
                    query=question,
                    top_k=5,
                    mode=retrieval_mode
                )

                if not results:
                    self.ui_call.emit(lambda: self.answer.setPlainText("❌ Ничего не найдено в учебнике"))
                    return

                # 2. Собираем контекст
                sources_text = "\n\n---\n\n".join([r["text"] for r in results])

                # 3. Отправляем в LLM
                client = OpenRouterClient(api_key)
                answer, usage = client.grounded_answer(
                    question=question,
                    sources_text=sources_text,
                    model=answer_model
                )

                # 4. Показываем ответ
                answer_text = f"{answer}\n\n---\n💰 Cost: ${usage.cost:.6f}" if usage.cost else answer
                self.ui_call.emit(lambda: self.answer.setPlainText(answer_text))
                self.ui_call.emit(lambda: self.overlay.show_message(answer_text))

                # 5. Показываем источники
                sources_preview = []
                for i, r in enumerate(results[:3]):
                    page = r.get("page", "?")
                    text_preview = r["text"][:400] + "..." if len(r["text"]) > 400 else r["text"]
                    sources_preview.append(f"[{i+1}] стр. {page}\n{text_preview}")
                self.ui_call.emit(lambda: self.sources.setPlainText("\n\n---\n\n".join(sources_preview)))

                self.ui_call.emit(lambda: self.status.setText("Готово"))

            except Exception as e:
                msg = str(e)
                self.ui_call.emit(lambda msg=msg: self.answer.setPlainText(f"❌ Ошибка: {msg}"))

        threading.Thread(target=worker, daemon=True).start()

    # ========== SCREEN CAPTURE ==========
    def _capture_screen(self) -> None:
        """Захват области экрана"""
        if self._capture_busy:
            return
        self._capture_busy = True
        try:
            import datetime
            import time

            self.overlay.clear_message()
            self.capture_status.setText("Выдели область мышкой. Esc - отмена")
            self.status.setText("Capturing…")
            was_visible = self.isVisible()
            overlay_was_visible = self.overlay.isVisible()
            if was_visible:
                self._hide_to_tray()
            if overlay_was_visible:
                self.overlay.hide()
            QtWidgets.QApplication.processEvents()
            time.sleep(0.20)

            picker = AreaSelectionDialog(None)
            if picker.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                self.capture_status.setText("Capture cancelled")
                self.status.setText("Capture cancelled")
                if overlay_was_visible:
                    self.overlay.show()
                self._capture_busy = False
                return

            rect = picker.selected_rect()
            if rect.width() < 10 or rect.height() < 10:
                self.capture_status.setText("Capture cancelled (empty area)")
                self.status.setText("Capture cancelled (empty area)")
                if overlay_was_visible:
                    self.overlay.show()
                self._capture_busy = False
                return

            captures_dir = data_dir() / "captures"
            captures_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_path = captures_dir / f"capture_{ts}.png"
            full_path = captures_dir / f"full_{ts}.png"
            meta_path = captures_dir / f"meta_{ts}.txt"

            composed = picker.background_pixmap()
            virtual_geom = picker.virtual_geometry()
            if composed.isNull():
                raise RuntimeError("Empty desktop capture")
            composed.save(str(full_path), "PNG")

            crop = QtCore.QRect(rect.left(), rect.top(), rect.width(), rect.height())
            cropped = composed.copy(crop)
            if cropped.isNull():
                raise RuntimeError("Empty capture")
            if not cropped.save(str(temp_path), "PNG"):
                raise RuntimeError("Failed to save capture image")
            meta_path.write_text(
                "Qt rect (logical px): "
                f"left={rect.left()} top={rect.top()} w={rect.width()} h={rect.height()}\n"
                "Virtual desktop: "
                f"left={virtual_geom.left()} top={virtual_geom.top()} w={virtual_geom.width()} h={virtual_geom.height()}\n"
                "Crop rect (virtual px): "
                f"left={crop.left()} top={crop.top()} w={crop.width()} h={crop.height()}\n"
                f"full={full_path}\n"
                f"crop={temp_path}\n",
                encoding="utf-8",
            )

            if overlay_was_visible:
                self.overlay.show()
            QtWidgets.QApplication.processEvents()

            self.capture_status.setText("Sending to OCR...")
            self.status.setText("Sending to OCR…")
            self.overlay.start_loading("OCR")
            self._send_to_ocr(temp_path)

        except Exception as e:
            self._capture_busy = False
            self.capture_status.setText(f"Capture error: {e}")
            self.overlay.show_message(f"❌ Capture error: {e}")

    def _send_to_ocr(self, image_path: Path) -> None:
        """Извлекает вопрос+варианты, делает retrieval и grounded answer."""
        api_key = self.api_key.text().strip()
        index = self.index
        ocr_model = self.ocr_model.currentText().strip()
        answer_model = self.answer_model.currentText().strip()
        retrieval_mode = self.retrieval_mode.currentText().strip()
        retrieval_retry_count = self._get_retrieval_retry_count()
        reverify_count = self._get_reverify_count()
        ocr_max_tokens = int(self.ocr_max_tokens.value())
        answer_max_tokens = int(self.answer_max_tokens.value())

        if not api_key:
            self._capture_busy = False
            QtWidgets.QMessageBox.critical(self, "Error", "OpenRouter API key is required for OCR")
            return

        if not index:
            self._capture_busy = False
            QtWidgets.QMessageBox.critical(self, "Error", "Index not loaded. Build index from PDF first.")
            return

        # Preprocess screenshot for OCR: upscale + contrast/sharpness to reduce misses on clean UI fonts.
        try:
            from PIL import Image, ImageEnhance

            img = Image.open(image_path)
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")
            else:
                img = img.convert("RGB")
            w, h = img.size
            img = img.resize((int(w * 2), int(h * 2)), resample=Image.Resampling.LANCZOS)
            img = ImageEnhance.Contrast(img).enhance(1.6)
            img = ImageEnhance.Sharpness(img).enhance(1.6)
            from io import BytesIO

            buf = BytesIO()
            img.save(buf, format="PNG", optimize=True)
            img_bytes = buf.getvalue()
        except Exception:
            with open(image_path, "rb") as f:
                img_bytes = f.read()

        client = OpenRouterClient(api_key)

        def worker():
            try:
                # 1. Извлекаем вопрос и варианты
                parsed, usage1, raw_debug = client.extract_question_and_options(
                    image_bytes=img_bytes,
                    mime="image/png",
                    model=ocr_model,
                    max_tokens=ocr_max_tokens,
                )

                question = parsed.get("question", "")
                options = parsed.get("options", {})

                def _looks_like_ui_chrome(text: str) -> bool:
                    t = (text or "").lower()
                    if not t:
                        return True
                    bad_markers = [
                        "ocr model",
                        "answer model",
                        "embed model",
                        "max_tokens",
                        "index folder",
                        "library / index",
                        "screen capture",
                        "overlay",
                        "capture (",
                        "ctrl+shift",
                        "build index",
                        "load index",
                        "openrouter key",
                        "settings",
                    ]
                    return any(m in t for m in bad_markers)

                if not question:
                    self.ui_call.emit(
                        lambda raw_debug=raw_debug: self.answer.setPlainText(
                            "❌ Не удалось распознать вопрос.\n\n"
                            "Ниже сырой OCR/ответ модели для отладки:\n\n"
                            f"{raw_debug}"
                        )
                    )
                    self.ui_call.emit(lambda raw_debug=raw_debug: self.question.setPlainText(raw_debug))
                    self.ui_call.emit(lambda: self.status.setText("OCR: question not found"))
                    return

                # Guardrail: иногда OCR цепляет интерфейс приложения вместо вопроса.
                if _looks_like_ui_chrome(str(question)) and not options:
                    self.ui_call.emit(
                        lambda raw_debug=raw_debug: self.answer.setPlainText(
                            "❌ OCR распознал не текст вопроса (похоже на интерфейс приложения).\n"
                            "Попробуй выделить область ТОЛЬКО с вопросом и вариантами.\n\n"
                            "Ниже сырой OCR/ответ модели для отладки:\n\n"
                            f"{raw_debug}"
                        )
                    )
                    self.ui_call.emit(lambda raw_debug=raw_debug: self.question.setPlainText(raw_debug))
                    self.ui_call.emit(lambda: self.status.setText("OCR: bad region"))
                    self.ui_call.emit(lambda: self.capture_status.setText("OCR: bad region"))
                    self.ui_call.emit(lambda: self.overlay.show_message("❌ OCR распознал интерфейс. Выдели только вопрос."))
                    return

                # Показываем распознанный вопрос в UI
                options_text = "\n".join([f"{l}) {t}" for l, t in options.items()])
                self.ui_call.emit(lambda: self.question.setPlainText(f"{question}\n\n{options_text}"))
                self.ui_call.emit(lambda: self.status.setText("OCR: ok"))
                self.ui_call.emit(lambda: self.capture_status.setText("OCR: ok"))
                # Чтобы не было ощущения "пусто" после выделения: показываем что распознали.
                self.ui_call.emit(
                    lambda q=question, ot=options_text: self.overlay.show_message(
                        f"✅ OCR распознал:\n{q}" + (f"\n\n{ot}" if ot.strip() else "")
                    )
                )

                retrieval_input = f"{question}\n\nВарианты:\n{options_text}" if options_text.strip() else question
                usage_rewrite_total = 0.0
                usage_judge_total = 0.0
                search_results: list[dict] = []
                rewritten_query = retrieval_input
                last_retry_reason = ""
                last_score = 0.0
                retry_note = ""

                for attempt_idx in range(retrieval_retry_count):
                    attempt_no = attempt_idx + 1
                    self.ui_call.emit(
                        lambda attempt_no=attempt_no, total=retrieval_retry_count: self.capture_status.setText(
                            f"LLM: переформулировка запроса {attempt_no}/{total}…"
                        )
                    )
                    self.ui_call.emit(lambda: self.overlay.start_loading("Переформулировка"))

                    reformulation_input = retrieval_input
                    if retry_note:
                        reformulation_input = (
                            f"{retrieval_input}\n\n"
                            f"Предыдущий запрос: {rewritten_query}\n"
                            f"Проблема: {retry_note}\n"
                            "Сформулируй новый поисковый запрос точнее и ближе к теме."
                        )

                    rewritten_query, usage_rewrite = client.reformulate_for_retrieval(
                        question=reformulation_input,
                        model=answer_model,
                    )
                    usage_rewrite_total += usage_rewrite.cost or 0.0
                    rewritten_query = rewritten_query.strip()
                    query_terms = [t for t in rewritten_query.replace("ё", "е").split() if len(t) > 3]
                    if not rewritten_query or _looks_like_ui_chrome(rewritten_query) or len(query_terms) < 4:
                        rewritten_query = retrieval_input

                    self.ui_call.emit(
                        lambda rq=rewritten_query, attempt_no=attempt_no, total=retrieval_retry_count: self.capture_status.setText(
                            f"Retrieval {attempt_no}/{total}: {rq[:100]}"
                        )
                    )
                    self.ui_call.emit(
                        lambda rq=rewritten_query, attempt_no=attempt_no, total=retrieval_retry_count: self.sources.setPlainText(
                            f"Attempt {attempt_no}/{total}\n\nRetrieval query:\n{rq}\n\nSources from textbook will appear here"
                        )
                    )

                    self.ui_call.emit(lambda: self.capture_status.setText("Retrieval…"))
                    self.ui_call.emit(lambda: self.overlay.start_loading("Поиск по учебнику"))
                    search_results = index.search(
                        query=rewritten_query,
                        top_k=5,
                        mode=retrieval_mode,
                    )

                    if not search_results:
                        last_retry_reason = "Поиск не нашёл ни одного фрагмента"
                        retry_note = last_retry_reason
                        continue

                    sources_preview = []
                    for i, r in enumerate(search_results[:5]):
                        page = r.get("page", "?")
                        text_preview = r["text"][:500] + "..." if len(r["text"]) > 500 else r["text"]
                        sources_preview.append(f"[{i+1}] стр. {page}\n{text_preview}")
                    self.ui_call.emit(
                        lambda rq=rewritten_query, sp=sources_preview, attempt_no=attempt_no, total=retrieval_retry_count: self.sources.setPlainText(
                            f"Attempt {attempt_no}/{total}\n\nRetrieval query:\n{rq}\n\n---\n\n" + "\n\n---\n\n".join(sp)
                        )
                    )

                    best = search_results[0]
                    self.ui_call.emit(lambda: self.capture_status.setText("LLM: оценка релевантности…"))
                    self.ui_call.emit(lambda: self.overlay.start_loading("Проверка релевантности"))
                    judge, usage_judge = client.judge_retrieval_relevance(
                        question=question,
                        retrieved_text=str(best.get("text") or ""),
                        model=answer_model,
                        retries=1,
                    )
                    usage_judge_total += usage_judge.cost or 0.0
                    try:
                        score = float(judge.get("score") or 0.0)
                    except Exception:
                        score = 0.0
                    last_score = score
                    is_relevant = bool(judge.get("is_relevant"))
                    reason = str(judge.get("reason") or "").strip()
                    if is_relevant and score >= 0.55:
                        last_retry_reason = ""
                        break

                    last_retry_reason = reason or "Judge посчитал результат нерелевантным"
                    retry_note = f"{last_retry_reason}. score={score:.2f}"
                    search_results = []

                if not search_results:
                    msg = (
                        "⚠️ Не удалось подобрать релевантный запрос к векторной базе.\n"
                        f"Попыток: {retrieval_retry_count}\n"
                        + (f"Последняя причина: {last_retry_reason}\n" if last_retry_reason else "")
                        + (f"Последний score: {last_score:.2f}\n" if last_score else "")
                        + "Ответ не сформирован."
                    )
                    self.ui_call.emit(lambda msg=msg: self.answer.setPlainText(msg))
                    self.ui_call.emit(lambda msg=msg: self.overlay.show_message(msg))
                    self.ui_call.emit(lambda: self.capture_status.setText("⚠️ Нерелевантно"))
                    return

                # 5. Собираем контекст
                pages_in_sources: list[int] = []
                ctx_blocks: list[str] = []
                for i, r in enumerate(search_results, start=1):
                    page = int(r.get("page") or 0)
                    if page:
                        pages_in_sources.append(page)
                    ctx_blocks.append(f"[source {i} | стр. {page}]\n{r['text']}")
                sources_text = "\n\n---\n\n".join(ctx_blocks)

                # 6. Сравниваем варианты с учебником (LLM call #3 = финальный ответ)
                comparisons: list[dict] = []
                usage_answer_total = 0.0
                for verify_idx in range(reverify_count):
                    verify_no = verify_idx + 1
                    self.ui_call.emit(
                        lambda verify_no=verify_no, total=reverify_count: self.capture_status.setText(
                            f"LLM: финальный ответ {verify_no}/{total}…"
                        )
                    )
                    self.ui_call.emit(lambda: self.overlay.start_loading("Финальный ответ"))
                    comparison, usage2 = client.compare_multiple_choice(
                        question=question,
                        options=options,
                        sources_text=sources_text,
                        model=answer_model,
                        max_tokens=answer_max_tokens,
                        retries=1,
                    )
                    comparisons.append(comparison)
                    usage_answer_total += usage2.cost or 0.0

                best_comparison = self._pick_best_comparison(comparisons, options)
                correct_option = best_comparison.get("correct_option", "?")
                confidence = best_comparison.get("confidence", 0)
                reasoning = best_comparison.get("reasoning", "")
                correct_text = best_comparison.get("correct_text", "")

                # 5. Формируем ответ
                total_cost = (usage1.cost or 0) + usage_rewrite_total + usage_judge_total + usage_answer_total

                # Полный ответ в поле Answer
                                # Полный ответ в поле Answer
                # Полный ответ
                full_answer = (
                    f"✅ **Правильный вариант: {correct_option}**\n"
                    f"⭐ Уверенность: {confidence*100:.0f}%\n\n"
                    f"📖 **Обоснование:**\n{reasoning}\n\n"
                    f"📝 **Текст ответа:**\n{correct_text}\n\n"
                    f"🔁 **Проверок ответа:** {reverify_count}\n\n"
                    "---\n"
                    f"💰 Cost: ${total_cost:.6f}"
                )

                # В оверлей тоже отправляем полный ответ (как было раньше с grounded_answer)
                self.ui_call.emit(lambda: self.answer.setPlainText(full_answer))
                self.ui_call.emit(lambda: self.overlay.show_message(full_answer))

                self.ui_call.emit(lambda: self.capture_status.setText("✓ Готово"))
                self.ui_call.emit(lambda: self.status.setText("Готово"))

            except Exception as e:
                msg = str(e)
                self.ui_call.emit(lambda msg=msg: self.answer.setPlainText(f"❌ Ошибка: {msg}"))
                self.ui_call.emit(lambda: self.status.setText("Error"))
                self.ui_call.emit(lambda msg=msg: self.overlay.show_message(f"❌ Ошибка: {msg}"))
            finally:
                self.ui_call.emit(lambda: setattr(self, "_capture_busy", False))

        threading.Thread(target=worker, daemon=True).start()

    def _load_index_from_folder(self) -> None:
        if self._index_loading:
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select ChromaDB Index Folder")
        if not folder:
            return
        self.index_path_label.setText(f"Index folder: {folder}")
        index_dir = Path(folder)
        embed_model = self.state.embed_model
        collection_hint = self.state.collection

        meta_path = index_dir / "index_meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                meta_embed_model = str(meta.get("embed_model") or "").strip()
                meta_collection = str(meta.get("collection") or "").strip()
                meta_chunking_mode = str(meta.get("chunking_mode") or "").strip()
                if meta_embed_model:
                    embed_model = meta_embed_model
                    self.state.embed_model = meta_embed_model
                    self.embed_model.setCurrentText(meta_embed_model)
                if meta_collection:
                    collection_hint = meta_collection
                    self.state.collection = meta_collection
                if meta_chunking_mode:
                    self.state.chunking_mode = meta_chunking_mode
                    self.chunking_mode.setCurrentText(meta_chunking_mode)
            except Exception:
                pass

        chroma_dir = index_dir / "chroma"
        chunks_path = index_dir / "chunks.jsonl"
        manifest_path = index_dir / "manifest.jsonl"
        has_old_layout = chroma_dir.exists() and chunks_path.exists()
        has_direct_chroma_layout = manifest_path.exists()
        if not has_old_layout and not has_direct_chroma_layout:
            QtWidgets.QMessageBox.critical(
                self,
                "Error",
                "Выбрана не папка готового индекса. "
                "Ожидаются либо ('chroma' + 'chunks.jsonl'), либо 'manifest.jsonl' в выбранной папке.",
            )
            return

        self._index_loading = True
        self.load_index_btn.setEnabled(False)
        self.progress.setRange(0, 0)
        self.status.setText("Загрузка индекса…")
        self.capture_status.setText("Loading index…")

        def worker() -> None:
            try:
                self.ui_call.emit(lambda: self.status.setText("Чтение коллекций Chroma…"))
                candidate_collections: list[str] = []
                try:
                    import chromadb
                    from chromadb.config import Settings

                    client = chromadb.PersistentClient(
                        path=str(chroma_dir),
                        settings=Settings(anonymized_telemetry=False),
                    )
                    raw_collections = client.list_collections()
                    for c in raw_collections:
                        name = getattr(c, "name", None) or str(c)
                        if name and name not in candidate_collections:
                            candidate_collections.append(name)
                except Exception as e:
                    raise RuntimeError(f"Не удалось прочитать коллекции из '{chroma_dir}': {e}") from e

                for fallback in (collection_hint, "phis_book_v3", "examrag"):
                    if fallback and fallback not in candidate_collections:
                        candidate_collections.append(fallback)

                self.ui_call.emit(lambda: self.status.setText("Загрузка embedding-модели и индекса…"))
                last_error: Exception | None = None
                for collection in candidate_collections:
                    try:
                        loaded_index = LocalIndex(
                            index_dir=index_dir,
                            collection=collection,
                            embed_model=embed_model,
                        )

                        def finish_success(
                            loaded_index=loaded_index,
                            collection=collection,
                            index_dir=index_dir,
                            embed_model=embed_model,
                            folder=folder,
                        ) -> None:
                            self.index = loaded_index
                            self.state.index_dir = index_dir
                            self.state.collection = collection
                            self.state.embed_model = embed_model
                            self.status.setText(f"✓ Индекс загружен: {folder} ({collection})")
                            self.capture_status.setText("✓ Index ready")
                            self.progress.setRange(0, 100)
                            self.progress.setValue(100)
                            self.load_index_btn.setEnabled(True)
                            self._index_loading = False

                        self.ui_call.emit(finish_success)
                        return
                    except Exception as e:
                        last_error = e

                available = ", ".join(candidate_collections) if candidate_collections else "none"
                raise RuntimeError(
                    f"{last_error}. Available collections: [{available}]"
                    if last_error
                    else f"Unknown index loading error. Available collections: [{available}]"
                )
            except Exception as e:
                msg = str(e)

                def finish_error(msg=msg) -> None:
                    self.progress.setRange(0, 100)
                    self.progress.setValue(0)
                    self.load_index_btn.setEnabled(True)
                    self._index_loading = False
                    self.status.setText("❌ Ошибка загрузки индекса")
                    self.capture_status.setText("Index load error")
                    QtWidgets.QMessageBox.critical(self, "Error", f"Ошибка загрузки:\n{msg}")

                self.ui_call.emit(finish_error)

        threading.Thread(target=worker, daemon=True).start()

def run() -> None:
    ensure_dirs()
    crash_log = data_dir() / "crash.log"
    crash_file = crash_log.open("a", encoding="utf-8", buffering=1)
    faulthandler.enable(crash_file, all_threads=True)

    def log_exception(exc_type, exc_value, exc_traceback) -> None:
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=crash_file)
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    sys.excepthook = log_exception
    app = QtWidgets.QApplication([])
    app.setQuitOnLastWindowClosed(False)
    w = MainWindow()
    w.show()
    app.exec()
