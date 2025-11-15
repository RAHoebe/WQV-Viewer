"""PyQt GUI for browsing and upscaling WQV wrist camera captures."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from PIL import Image
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QAction, QIcon, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .parser import WQVImage, load_wqv_pdb
from .upscaling import (
    ALLOWED_CONVENTIONAL_SCALES,
    Upscaler,
    ai_upscalers,
    conventional_upscalers,
    upscale_sequence,
)


PipelineStage = Tuple[str, Upscaler, int]


logger = logging.getLogger(__name__)


def _pil_to_qpixmap(image: Image.Image) -> QPixmap:
    """Convert a :class:`PIL.Image.Image` into a :class:`QPixmap`."""

    if image.mode not in {"RGB", "RGBA"}:
        pil_image = image.convert("RGBA")
    else:
        pil_image = image.copy()

    mode = pil_image.mode
    buffer = pil_image.tobytes("raw", mode)
    if mode == "RGB":
        qimage = QImage(buffer, pil_image.width, pil_image.height, QImage.Format.Format_RGB888)
    else:  # RGBA
        qimage = QImage(buffer, pil_image.width, pil_image.height, QImage.Format.Format_RGBA8888)

    return QPixmap.fromImage(qimage.copy())


class ImageLabel(QLabel):
    """Label that scales pixmaps while keeping the aspect ratio."""

    def __init__(self, placeholder: str, *, smooth: bool = True) -> None:
        super().__init__(placeholder)
        self._placeholder = placeholder
        self._smooth = smooth
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumSize(240, 240)
        self._pixmap: Optional[QPixmap] = None

    def set_image(self, image: Image.Image | WQVImage) -> None:
        if isinstance(image, WQVImage):
            pixmap = QPixmap.fromImage(image.to_qimage())
        else:
            pixmap = _pil_to_qpixmap(image)
        self._pixmap = pixmap
        self.setText("")
        self._update_scaled_pixmap()

    def clear(self) -> None:  # type: ignore[override]
        self._pixmap = None
        super().setPixmap(QPixmap())
        super().setText(self._placeholder)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self) -> None:
        if not self._pixmap:
            return
        mode = (
            Qt.TransformationMode.SmoothTransformation
            if self._smooth
            else Qt.TransformationMode.FastTransformation
        )
        scaled = self._pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            mode,
        )
        super().setPixmap(scaled)


class ImageBrowser(QWidget):
    """Main widget providing image navigation, previews and upscaling controls."""

    def __init__(self) -> None:
        super().__init__()
        self._images: List[WQVImage] = []
        self._current_image: Optional[WQVImage] = None
        self._last_upscaled_image: Optional[Image.Image] = None
        self._conventional_upscalers = conventional_upscalers()
        self._ai_upscalers = ai_upscalers()
        self._conventional_map = {upscaler.id: upscaler for upscaler in self._conventional_upscalers}
        self._ai_map = {upscaler.id: upscaler for upscaler in self._ai_upscalers}
        self._controls_ready = False

        self._build_ui()
        self._populate_controls()
        self._controls_ready = True
        self._update_controls_enabled()

    # ------------------------------------------------------------------ setup
    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Left column: file list
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.list_widget.setViewMode(QListView.ViewMode.IconMode)
        self.list_widget.setIconSize(QSize(120, 120))
        self.list_widget.setResizeMode(QListView.ResizeMode.Adjust)
        self.list_widget.setMovement(QListView.Movement.Static)
        self.list_widget.setSpacing(8)
        self.list_widget.setWrapping(True)
        splitter.addWidget(self.list_widget)

        # Right column: previews, controls and metadata
        right = QWidget()
        right_layout = QVBoxLayout(right)
        splitter.addWidget(right)

        previews = QHBoxLayout()
        right_layout.addLayout(previews)

        original_group = QGroupBox("Original")
        original_layout = QVBoxLayout(original_group)
        self.original_label = ImageLabel("Load an image…", smooth=False)
        original_layout.addWidget(self.original_label)
        previews.addWidget(original_group)

        upscaled_group = QGroupBox("Upscaled")
        upscaled_layout = QVBoxLayout(upscaled_group)
        self.upscaled_label = ImageLabel("Configure upscaling…", smooth=False)
        upscaled_layout.addWidget(self.upscaled_label)
        previews.addWidget(upscaled_group)

        self.upscale_status = QLabel("Upscale: pipeline disabled")
        self.upscale_status.setFrameShape(QFrame.Shape.StyledPanel)
        self.upscale_status.setMinimumHeight(32)
        right_layout.addWidget(self.upscale_status)

        controls_group = QGroupBox("Upscaling controls")
        controls_layout = QVBoxLayout(controls_group)

        conventional_row = QHBoxLayout()
        self.conventional_checkbox = QCheckBox("Conventional")
        self.conventional_checkbox.setChecked(True)
        self.conventional_combo = QComboBox()
        self.conventional_scale_combo = QComboBox()
        conventional_row.addWidget(self.conventional_checkbox)
        conventional_row.addWidget(self.conventional_combo)
        conventional_row.addWidget(self.conventional_scale_combo)
        controls_layout.addLayout(conventional_row)

        ai_row = QHBoxLayout()
        self.ai_checkbox = QCheckBox("AI")
        self.ai_checkbox.setChecked(False)
        self.ai_combo = QComboBox()
        self.ai_scale_combo = QComboBox()
        ai_row.addWidget(self.ai_checkbox)
        ai_row.addWidget(self.ai_combo)
        ai_row.addWidget(self.ai_scale_combo)
        controls_layout.addLayout(ai_row)

        device_row = QHBoxLayout()
        self.device_label = QLabel("Device")
        self.device_combo = QComboBox()
        device_row.addWidget(self.device_label)
        device_row.addWidget(self.device_combo)
        device_row.addStretch(1)
        controls_layout.addLayout(device_row)

        right_layout.addWidget(controls_group)
        right_layout.addStretch(1)

        # Signal wiring
        self.list_widget.currentItemChanged.connect(self._on_selection_changed)
        self.list_widget.itemSelectionChanged.connect(self._on_selection_changed)
        self.conventional_checkbox.toggled.connect(self._on_pipeline_changed)
        self.conventional_combo.currentIndexChanged.connect(self._on_pipeline_changed)
        self.conventional_scale_combo.currentIndexChanged.connect(self._on_pipeline_changed)
        self.ai_checkbox.toggled.connect(self._on_ai_toggled)
        self.ai_combo.currentIndexChanged.connect(self._on_ai_upscaler_changed)
        self.ai_scale_combo.currentIndexChanged.connect(self._on_pipeline_changed)
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)

    def _populate_controls(self) -> None:
        self.conventional_combo.blockSignals(True)
        self.conventional_combo.clear()
        for upscaler in self._conventional_upscalers:
            self.conventional_combo.addItem(upscaler.label, upscaler.id)
        self.conventional_combo.blockSignals(False)

        self.conventional_scale_combo.blockSignals(True)
        self.conventional_scale_combo.clear()
        for scale in ALLOWED_CONVENTIONAL_SCALES:
            self.conventional_scale_combo.addItem(f"{scale}×", scale)
        self.conventional_scale_combo.blockSignals(False)
        self.conventional_scale_combo.setCurrentIndex(0)

        self.ai_combo.blockSignals(True)
        self.ai_combo.clear()
        default_ai_index: Optional[int] = None
        for index, upscaler in enumerate(self._ai_upscalers):
            self.ai_combo.addItem(upscaler.label, upscaler.id)
            if default_ai_index is None and 2 in upscaler.supported_scales():
                default_ai_index = index
        self.ai_combo.blockSignals(False)
        if self.ai_combo.count():
            self.ai_combo.setCurrentIndex(default_ai_index or 0)

        self.ai_checkbox.setEnabled(bool(self._ai_upscalers))
        self._refresh_ai_scales(preferred_scale=2)
        self._populate_device_controls()

    # --------------------------------------------------------------- data-load
    def load_images(self, images: Sequence[WQVImage]) -> None:
        self._images = list(images)
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for image in self._images:
            item = QListWidgetItem(image.path.name)
            thumbnail = _pil_to_qpixmap(image.image)
            thumbnail = thumbnail.scaled(
                self.list_widget.iconSize(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
            item.setIcon(QIcon(thumbnail))
            item.setData(Qt.ItemDataRole.UserRole, image)
            item.setSizeHint(QSize(140, 160))
            self.list_widget.addItem(item)
        self.list_widget.blockSignals(False)

        if self.list_widget.count():
            self.list_widget.setCurrentRow(0)
        else:
            self._set_current_image(None)

    def clear(self) -> None:
        self._images.clear()
        self.list_widget.clear()
        self._set_current_image(None)

    def last_upscaled_image(self) -> Optional[Image.Image]:
        if self._last_upscaled_image is None:
            return None
        return self._last_upscaled_image.copy()

    def current_image_basename(self) -> Optional[str]:
        if self._current_image is None:
            return None
        stem = self._current_image.path.stem
        return stem or None

    # -------------------------------------------------------------- selections
    def _on_selection_changed(self, *_args) -> None:
        item = self.list_widget.currentItem()
        image = item.data(Qt.ItemDataRole.UserRole) if item else None
        if isinstance(image, WQVImage):
            self._set_current_image(image)
        else:
            self._set_current_image(None)

    def _set_current_image(self, image: Optional[WQVImage]) -> None:
        self._current_image = image
        self._last_upscaled_image = None
        if image is None:
            self.original_label.clear()
            self.upscaled_label.clear()
            self.upscale_status.setText("Select an image from the list")
        else:
            self.original_label.set_image(image)
            self._apply_upscale()

    # ------------------------------------------------------------- pipelines
    def _on_pipeline_changed(self) -> None:
        if not self._controls_ready:
            return
        self._update_controls_enabled()
        self._apply_upscale()

    def _on_ai_toggled(self, _checked: bool) -> None:
        if not self._controls_ready:
            return
        self._update_controls_enabled()
        self._apply_upscale()

    def _on_ai_upscaler_changed(self, _index: int) -> None:
        if not self._controls_ready:
            return
        self._refresh_ai_scales()
        self._apply_upscale()

    def _update_controls_enabled(self) -> None:
        conventional_enabled = self.conventional_checkbox.isChecked()
        self.conventional_combo.setEnabled(conventional_enabled)
        self.conventional_scale_combo.setEnabled(conventional_enabled)

        ai_enabled = self.ai_checkbox.isChecked() and bool(self._ai_upscalers)
        self.ai_combo.setEnabled(ai_enabled)
        self.ai_scale_combo.setEnabled(ai_enabled)
        self.device_label.setEnabled(ai_enabled)
        self.device_combo.setEnabled(ai_enabled)

    def _refresh_ai_scales(self, *, preferred_scale: Optional[int] = None) -> None:
        self.ai_scale_combo.blockSignals(True)
        previous = self.ai_scale_combo.currentData()
        self.ai_scale_combo.clear()
        upscaler = self._resolve_ai_upscaler()
        if upscaler is not None:
            supported = list(upscaler.supported_scales())
            for scale in supported:
                self.ai_scale_combo.addItem(f"{scale}×", scale)
            if preferred_scale in supported:
                index = supported.index(preferred_scale)
            else:
                if isinstance(previous, int) and previous in supported:
                    index = supported.index(previous)
                else:
                    index = 0
            if self.ai_scale_combo.count():
                self.ai_scale_combo.setCurrentIndex(index)
        self.ai_scale_combo.blockSignals(False)

    def _populate_device_controls(self) -> None:
        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        self.device_combo.addItem("Auto (GPU→CPU fallback)", "auto")
        self.device_combo.addItem("GPU only", "gpu")
        self.device_combo.addItem("CPU only", "cpu")
        self.device_combo.setCurrentIndex(0)
        self.device_combo.blockSignals(False)

    def _on_device_changed(self, _index: int) -> None:
        if not self._controls_ready:
            return
        self._apply_upscale()

    def _selected_device_policy(self) -> str:
        data = self.device_combo.currentData()
        if isinstance(data, str):
            return data
        return "auto"

    def _device_policy_label(self, policy: str) -> str:
        mapping = {"auto": "Auto", "gpu": "GPU", "cpu": "CPU"}
        return mapping.get(policy, policy.upper())

    def _resolve_conventional_upscaler(self) -> Optional[Upscaler]:
        data = self.conventional_combo.currentData()
        if isinstance(data, str):
            return self._conventional_map.get(data)
        if self._conventional_upscalers:
            return self._conventional_upscalers[0]
        return None

    def _current_conventional_scale(self) -> int:
        data = self.conventional_scale_combo.currentData()
        if isinstance(data, int):
            return data
        return ALLOWED_CONVENTIONAL_SCALES[0]

    def _resolve_ai_upscaler(self) -> Optional[Upscaler]:
        if not self._ai_upscalers:
            return None
        data = self.ai_combo.currentData()
        if isinstance(data, str) and data in self._ai_map:
            return self._ai_map[data]
        return self._ai_upscalers[0]

    def _current_ai_scale(self) -> Optional[int]:
        upscaler = self._resolve_ai_upscaler()
        if upscaler is None:
            return None
        data = self.ai_scale_combo.currentData()
        supported = list(upscaler.supported_scales())
        if isinstance(data, int) and data in supported:
            return data
        return supported[0] if supported else None

    def _build_pipeline(self) -> List[PipelineStage]:
        pipeline: List[PipelineStage] = []
        if self.conventional_checkbox.isChecked():
            upscaler = self._resolve_conventional_upscaler()
            if upscaler is not None:
                pipeline.append(("Conventional", upscaler, self._current_conventional_scale()))
        if self.ai_checkbox.isChecked():
            upscaler = self._resolve_ai_upscaler()
            scale = self._current_ai_scale()
            if upscaler is not None and scale is not None:
                pipeline.append(("AI", upscaler, scale))
        return pipeline

    def _apply_upscale(self) -> None:
        if self._current_image is None:
            self.upscaled_label.clear()
            self.upscale_status.setText("Select an image from the list")
            return

        self._last_upscaled_image = None
        pipeline = self._build_pipeline()
        source_image = self._current_image.image.copy()

        if not pipeline:
            self._last_upscaled_image = source_image.copy()
            self.upscaled_label.set_image(source_image)
            self.upscale_status.setText(
                f"Original — {source_image.width}×{source_image.height}"
            )
            return

        ai_policy = self._selected_device_policy()
        has_ai_stage = any(label == "AI" for label, *_ in pipeline)
        attempt_policies: List[str] = [ai_policy]
        if has_ai_stage and ai_policy != "cpu":
            attempt_policies.append("cpu")

        last_error: Optional[Exception] = None
        for attempt_index, attempt_policy in enumerate(attempt_policies):
            try:
                upscaled_image, summary_text, fallback_used = self._execute_pipeline(
                    pipeline,
                    source_image.copy(),
                    attempt_policy,
                    original_policy=ai_policy,
                )
            except Exception as exc:  # pragma: no cover - GUI feedback path
                last_error = exc
                logger.debug("Upscale attempt with policy %s failed: %s", attempt_policy, exc, exc_info=True)
                continue

            self.upscaled_label.set_image(upscaled_image)
            if fallback_used and attempt_policy == "cpu" and ai_policy != "cpu":
                summary_text += " (GPU fallback)"
            self.upscale_status.setText(summary_text)
            self._last_upscaled_image = upscaled_image.copy()
            return

        self.upscaled_label.clear()
        if last_error is not None:
            self.upscale_status.setText(f"Upscale failed: {last_error}")
        else:
            self.upscale_status.setText("Upscale failed: no valid pipeline stage")

    def _execute_pipeline(
        self,
        pipeline: List[PipelineStage],
        source_image: Image.Image,
        attempt_policy: str,
        *,
        original_policy: str,
    ) -> Tuple[Image.Image, str, bool]:
        policy_overrides: List[Optional[str]] = []
        revert_stack: List[Optional[Tuple]] = []
        policy_error: Optional[Exception] = None

        for label, upscaler, _ in pipeline:
            if label != "AI":
                policy_overrides.append(None)
                revert_stack.append(None)
                continue

            setter = getattr(upscaler, "set_device_policy", None)
            getter = getattr(upscaler, "device_policy", None)
            if not callable(setter):
                policy_overrides.append(None)
                revert_stack.append(None)
                continue

            original = None
            if callable(getter):
                try:
                    original = getter()
                except Exception:  # pragma: no cover - defensive guard
                    logger.debug("Unable to read current device policy", exc_info=True)

            try:
                setter(attempt_policy)
            except Exception as exc:
                policy_error = exc
                break

            policy_overrides.append(attempt_policy)
            if original is not None and original != attempt_policy:
                revert_stack.append((setter, original))
            else:
                revert_stack.append(None)

        if policy_error is not None:
            for entry in reversed(revert_stack):
                if entry:
                    setter, previous = entry
                    try:
                        setter(previous)
                    except Exception:
                        logger.debug("Failed to restore device policy after error", exc_info=True)
            raise policy_error

        steps = [(upscaler, scale) for _, upscaler, scale in pipeline]

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            upscaled_image = upscale_sequence(source_image, steps)
        finally:
            QApplication.restoreOverrideCursor()

        stage_texts: List[str] = []
        for index, (label, upscaler, scale) in enumerate(pipeline):
            descriptor = upscaler.label
            if label == "AI":
                describe_backend = getattr(upscaler, "describe_backend", None)
                summary: Optional[str] = None
                if callable(describe_backend):
                    summary = describe_backend()
                if not summary:
                    policy = policy_overrides[index]
                    if policy:
                        summary = self._device_policy_label(policy)
                if summary:
                    descriptor = f"{descriptor} [{summary}]"
            stage_texts.append(f"{label}: {descriptor} ×{scale}")

        summary_text = f"{' → '.join(stage_texts)} — {upscaled_image.width}×{upscaled_image.height}"

        for entry in reversed(revert_stack):
            if entry:
                setter, previous = entry
                try:
                    setter(previous)
                except Exception:  # pragma: no cover - defensive guard
                    logger.debug("Failed to restore device policy", exc_info=True)

        fallback_used = attempt_policy != original_policy and any(policy_overrides)
        return upscaled_image, summary_text, fallback_used


class MainWindow(QMainWindow):
    """Main application window hosting the browser widget."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("WQV Wristcam Viewer")
        self.resize(1100, 720)
        self._apply_window_icon()

        self.browser = ImageBrowser()
        self.setCentralWidget(self.browser)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self._create_actions()
        self._build_menus()
        self._build_toolbar()

    # ---------------------------------------------------------------- actions
    def _resources_dir(self) -> Path:
        return Path(__file__).resolve().parent.parent / "resources"

    def _resource_icon(self, name: str) -> QIcon:
        icon_path = self._resources_dir() / name
        if not icon_path.exists():
            return QIcon()
        return QIcon(str(icon_path))

    def _app_icon(self) -> Optional[QIcon]:
        for candidate in ("app.ico", "app.png"):
            icon = self._resource_icon(candidate)
            if not icon.isNull():
                return icon
        return None

    def _apply_window_icon(self) -> None:
        icon = self._app_icon()
        if icon is not None:
            self.setWindowIcon(icon)

    def _create_actions(self) -> None:
        self.open_pdb_action = QAction("Open WQVLinkDB…", self)
        self.open_pdb_action.setIcon(self._resource_icon("open.png"))
        self.open_pdb_action.triggered.connect(self.open_pdb)

        self.export_action = QAction("Export Selected…", self)
        self.export_action.setIcon(self._resource_icon("export.png"))
        self.export_action.triggered.connect(self.export_selected)

        self.clear_action = QAction("Clear", self)
        self.clear_action.setIcon(self._resource_icon("clear.png"))
        self.clear_action.triggered.connect(self.browser.clear)

        self.exit_action = QAction("Exit", self)
        self.exit_action.setIcon(self._resource_icon("exit.png"))
        self.exit_action.triggered.connect(self.close)

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(self.open_pdb_action)
        file_menu.addSeparator()
        file_menu.addAction(self.export_action)
        file_menu.addSeparator()
        file_menu.addAction(self.clear_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Main")
        toolbar.addAction(self.open_pdb_action)
        toolbar.addAction(self.export_action)
        toolbar.addSeparator()
        toolbar.addAction(self.clear_action)
        toolbar.addSeparator()
        toolbar.addAction(self.exit_action)
        self.addToolBar(toolbar)

    # --------------------------------------------------------------- file ops
    def open_pdb(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open WQV Palm database",
            str(Path.home()),
            "WQV Palm database (*.pdb);;All files (*)",
        )
        if not path:
            return
        count = self._load_pdb(Path(path))
        self.status_bar.showMessage(f"Loaded {count} images from {Path(path).name}", 5000)

    def export_selected(self) -> None:
        image = self.browser.last_upscaled_image()
        if image is None:
            self.status_bar.showMessage("No upscaled image to export", 3000)
            return

        stem = self.browser.current_image_basename() or "upscaled"
        default_name = f"{stem}_upscaled.png"
        default_path = Path.home() / default_name

        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Save upscaled image",
            str(default_path),
            "PNG Images (*.png)",
        )
        if not path_str:
            return

        path = Path(path_str)
        if path.suffix.lower() != ".png":
            path = path.with_suffix(".png")

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path, format="PNG")
        except Exception as exc:
            self.status_bar.showMessage(f"Failed to save image: {exc}", 5000)
            return

        self.status_bar.showMessage(f"Saved {path.name}", 5000)

    # ------------------------------------------------------------ test hooks
    def _load_pdb(self, path: Path) -> int:
        images = load_wqv_pdb(path)
        self.browser.load_images(images)
        return len(images)

    def _export_upscaled_image(self, path: Path) -> bool:
        image = self.browser.last_upscaled_image()
        if image is None:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path, format="PNG")
        return True


def run() -> None:
    app = QApplication.instance() or QApplication([])

    resources_dir = Path(__file__).resolve().parent.parent / "resources"
    for candidate in ("app.ico", "app.png"):
        icon_path = resources_dir / candidate
        if icon_path.exists():
            app_icon = QIcon(str(icon_path))
            if not app_icon.isNull():
                app.setWindowIcon(app_icon)
                break

    if sys.platform.startswith("win"):
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("RAHoebe.WQVViewer")
        except Exception:  # pragma: no cover - Windows only optional call
            logger.debug("Failed to set Windows AppUserModelID", exc_info=True)
    window = MainWindow()
    window.show()
    app.exec()
