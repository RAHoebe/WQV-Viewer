"""PyQt GUI for browsing and upscaling WQV wrist camera captures."""

from __future__ import annotations

import json
import logging
import sys
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt6.QtCore import (
    Qt,
    QSize,
    QSettings,
    QThread,
    QObject,
    pyqtSignal,
    pyqtSlot,
    QByteArray,
    QEvent,
    QPointF,
    QRectF,
    QItemSelectionModel,
    QMimeData,
    QUrl,
)
from PyQt6.QtGui import (
    QAction,
    QActionGroup,
    QIcon,
    QImage,
    QPixmap,
    QPalette,
    QPainter,
    QColor,
    QPen,
    QPolygonF,
    QWheelEvent,
    QKeySequence,
    QShortcut,
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
)
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QInputDialog,
    QFrame,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .exporter import export_result
from .parser import (
    WQVImage,
    WQVImageKind,
    WQVLoadReport,
    delete_wqv_pdb_records,
    load_wqv_image,
    load_wqv_pdb,
)
from .pipeline import PipelineConfig, PipelineResult, PipelineStage, build_pipeline, run_pipeline
from .upscaling import ALLOWED_CONVENTIONAL_SCALES, Upscaler, ai_upscalers, conventional_upscalers


logger = logging.getLogger(__name__)

_LANCZOS_FILTER = getattr(getattr(Image, "Resampling", None), "LANCZOS", None) or Image.LANCZOS


def _pil_to_qpixmap(image: Image.Image) -> QPixmap:
    """Convert a :class:`PIL.Image.Image` into a :class:`QPixmap`."""

    if image.mode not in {"RGB", "RGBA"}:
        pil_image = image.convert("RGBA")
    else:
        pil_image = image.copy()

    qimage = ImageQt(pil_image)
    if not isinstance(qimage, QImage):
        qimage = QImage(qimage)

    return QPixmap.fromImage(qimage.copy())


class ZoomableImageView(QWidget):
    """Image view that supports fit/actual zoom modes and Ctrl+wheel zoom."""

    zoomModeChanged = pyqtSignal(str)
    zoomFactorChanged = pyqtSignal(float)

    _MIN_ZOOM = 0.1
    _MAX_ZOOM = 8.0
    _ZOOM_STEP = 1.25

    def __init__(self, placeholder: str, *, smooth: bool = True) -> None:
        super().__init__()
        self._placeholder = placeholder
        self._smooth = smooth
        self._mode = "fit"
        self._scale = 1.0
        self._pixmap: Optional[QPixmap] = None

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._scroll)

        self._label = QLabel(placeholder)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setFrameShape(QFrame.Shape.StyledPanel)
        self._label.setMinimumSize(240, 240)
        self._label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._scroll.setWidget(self._label)

        self._scroll.viewport().installEventFilter(self)
        self._label.installEventFilter(self)

    def set_image(self, image: Image.Image | WQVImage) -> None:
        if isinstance(image, WQVImage):
            pixmap = _pil_to_qpixmap(image.image)
        else:
            pixmap = _pil_to_qpixmap(image)
        self._pixmap = pixmap
        self._scale = 1.0
        self._label.setText("")
        self._update_view()

    def clear(self) -> None:
        self._pixmap = None
        self._scale = 1.0
        self._mode = "fit"
        self._scroll.setWidgetResizable(True)
        self._label.setPixmap(QPixmap())
        self._label.setText(self._placeholder)
        self.zoomFactorChanged.emit(1.0)

    def set_mode(self, mode: str) -> None:
        if mode not in {"fit", "actual"}:
            return
        if self._mode == mode:
            return
        self._mode = mode
        if mode == "fit":
            self._scroll.setWidgetResizable(True)
            self._scale = 1.0
        else:
            self._scroll.setWidgetResizable(False)
            self._scale = 1.0
        self._update_view()
        self.zoomModeChanged.emit(self._mode)

    def toggle_mode(self) -> None:
        self.set_mode("actual" if self._mode == "fit" else "fit")

    def mode(self) -> str:
        return self._mode

    def zoom_in(self) -> None:
        self._adjust_zoom(self._ZOOM_STEP)

    def zoom_out(self) -> None:
        self._adjust_zoom(1 / self._ZOOM_STEP)

    def reset_zoom(self) -> None:
        if self._mode == "fit":
            self._update_view()
        else:
            self._scale = 1.0
            self._update_view()
        self.zoomFactorChanged.emit(self.zoom_factor())

    def current_pixmap(self) -> Optional[QPixmap]:
        pixmap = self._label.pixmap()
        if pixmap is None or pixmap.isNull():
            return None
        return QPixmap(pixmap)

    def zoom_factor(self) -> float:
        return 0.0 if self._mode == "fit" else self._scale

    def eventFilter(self, obj, event):  # type: ignore[override]
        if event.type() == QEvent.Type.Resize:
            if obj is self._scroll.viewport() and self._mode == "fit":
                self._update_view()
        elif event.type() == QEvent.Type.Wheel:
            wheel_event = event  # type: ignore[assignment]
            if isinstance(wheel_event, QWheelEvent):
                if wheel_event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                    if wheel_event.angleDelta().y() > 0:
                        self.zoom_in()
                    else:
                        self.zoom_out()
                    return True
        return super().eventFilter(obj, event)

    def _adjust_zoom(self, factor: float) -> None:
        if not self._pixmap:
            return
        if self._mode == "fit":
            # Switching to actual mode for manual zoom control.
            self.set_mode("actual")
        new_scale = max(self._MIN_ZOOM, min(self._MAX_ZOOM, self._scale * factor))
        if abs(new_scale - self._scale) < 1e-3:
            return
        self._scale = new_scale
        self._update_view()
        self.zoomFactorChanged.emit(self._scale)

    def _update_view(self) -> None:
        if not self._pixmap:
            return
        mode = (
            Qt.TransformationMode.SmoothTransformation
            if self._smooth
            else Qt.TransformationMode.FastTransformation
        )
        if self._mode == "fit":
            viewport_size = self._scroll.viewport().size()
            scaled = self._pixmap.scaled(
                viewport_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                mode,
            )
            self._label.setPixmap(scaled)
            self._label.resize(viewport_size)
            self.zoomFactorChanged.emit(0.0)
        else:
            width = max(1, int(self._pixmap.width() * self._scale))
            height = max(1, int(self._pixmap.height() * self._scale))
            scaled = self._pixmap.scaled(
                width,
                height,
                Qt.AspectRatioMode.KeepAspectRatio,
                mode,
            )
            self._label.setPixmap(scaled)
            self._label.resize(scaled.size())
            self.zoomFactorChanged.emit(self._scale)



class PipelineWorker(QObject):
    """Background worker that executes an upscale pipeline."""

    finished = pyqtSignal(int, PipelineResult)
    failed = pyqtSignal(int, str)

    def __init__(
        self,
        job_id: int,
        pipeline: Sequence[PipelineStage],
        source_image: Image.Image,
        device_policy: str,
    ) -> None:
        super().__init__()
        self._job_id = job_id
        self._pipeline = list(pipeline)
        self._source_image = source_image.copy()
        self._device_policy = device_policy

    @pyqtSlot()
    def run(self) -> None:
        try:
            result = run_pipeline(
                self._pipeline,
                self._source_image,
                original_policy=self._device_policy,
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            self.failed.emit(self._job_id, str(exc))
            return

        self.finished.emit(self._job_id, result)


class ImageBrowser(QWidget):
    """Main widget providing image navigation, previews and upscaling controls."""

    upscale_started = pyqtSignal()
    upscale_completed = pyqtSignal()
    upscale_failed = pyqtSignal(str)
    upscale_cancelled = pyqtSignal()

    _EXTERNAL_THUMB_LIMIT = 200

    def __init__(self, *, async_enabled: bool = True) -> None:
        super().__init__()
        self._images: List[WQVImage] = []
        self._current_image: Optional[WQVImage] = None
        self._last_upscaled_image: Optional[Image.Image] = None
        self._last_results: Dict[str, PipelineResult] = {}
        self._conventional_upscalers = conventional_upscalers()
        self._ai_upscalers = ai_upscalers()
        self._conventional_map = {upscaler.id: upscaler for upscaler in self._conventional_upscalers}
        self._ai_map = {upscaler.id: upscaler for upscaler in self._ai_upscalers}
        self._controls_ready = False
        self._async_enabled = async_enabled
        self._job_counter = 0
        self._active_job_id: Optional[int] = None
        self._cancelled_jobs: Dict[int, bool] = {}
        self._workers: Dict[int, Tuple[PipelineWorker, QThread]] = {}

        self._build_ui()
        self._populate_controls()
        self._controls_ready = True
        self._update_controls_enabled()

    # ------------------------------------------------------------------ setup
    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(self.splitter)

        # Left column: file list + metadata
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.list_widget.setViewMode(QListView.ViewMode.IconMode)
        self.list_widget.setIconSize(QSize(120, 120))
        self.list_widget.setResizeMode(QListView.ResizeMode.Adjust)
        self.list_widget.setMovement(QListView.Movement.Static)
        self.list_widget.setSpacing(8)
        self.list_widget.setWrapping(True)
        left_layout.addWidget(self.list_widget, 1)

        metadata_group = QGroupBox("Metadata")
        metadata_layout = QFormLayout(metadata_group)
        metadata_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        self._metadata_labels: Dict[str, QLabel] = {}
        for key, label_text in (
            ("filename", "Filename"),
            ("captured", "Captured"),
            ("resolution", "Resolution"),
            ("record", "Record"),
        ):
            value_label = QLabel("—")
            value_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            metadata_layout.addRow(f"{label_text}:", value_label)
            self._metadata_labels[key] = value_label
        left_layout.addWidget(metadata_group, 0)

        self.splitter.addWidget(left)

        # Right column: previews, controls and metadata
        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.splitter.addWidget(right)

        previews = QHBoxLayout()
        right_layout.addLayout(previews, 1)

        original_group = QGroupBox("Original")
        original_layout = QVBoxLayout(original_group)
        self.original_view = ZoomableImageView("Load an image…", smooth=False)
        self.original_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        original_layout.addWidget(self.original_view)
        original_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        previews.addWidget(original_group)

        upscaled_group = QGroupBox("Upscaled")
        upscaled_layout = QVBoxLayout(upscaled_group)
        self.upscaled_view = ZoomableImageView("Configure upscaling…", smooth=False)
        self.upscaled_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        upscaled_layout.addWidget(self.upscaled_view)
        upscaled_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        previews.addWidget(upscaled_group)

        self.primary_status = QLabel("Pipeline disabled")
        self.primary_status.setFrameShape(QFrame.Shape.StyledPanel)
        self.primary_status.setMinimumHeight(32)
        right_layout.addWidget(self.primary_status)

        controls_group = QGroupBox("Upscaling controls")
        controls_layout = QVBoxLayout(controls_group)
        controls_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        primary_group = QGroupBox("Upscaling pipeline")
        primary_layout = QVBoxLayout(primary_group)

        conventional_row = QHBoxLayout()
        self.conventional_checkbox = QCheckBox("Conventional")
        self.conventional_checkbox.setChecked(True)
        self.conventional_combo = QComboBox()
        self.conventional_scale_combo = QComboBox()
        conventional_row.addWidget(self.conventional_checkbox)
        conventional_row.addWidget(self.conventional_combo)
        conventional_row.addWidget(self.conventional_scale_combo)
        primary_layout.addLayout(conventional_row)

        ai_row = QHBoxLayout()
        self.ai_checkbox = QCheckBox("AI")
        self.ai_checkbox.setChecked(False)
        self.ai_combo = QComboBox()
        self.ai_scale_combo = QComboBox()
        ai_row.addWidget(self.ai_checkbox)
        ai_row.addWidget(self.ai_combo)
        ai_row.addWidget(self.ai_scale_combo)
        primary_layout.addLayout(ai_row)

        order_row = QHBoxLayout()
        self.order_label = QLabel("Order")
        self.order_combo = QComboBox()
        order_row.addWidget(self.order_label)
        order_row.addWidget(self.order_combo)
        order_row.addStretch(1)
        primary_layout.addLayout(order_row)

        controls_layout.addWidget(primary_group)

        device_row = QHBoxLayout()
        self.device_label = QLabel("Device")
        self.device_combo = QComboBox()
        device_row.addWidget(self.device_label)
        device_row.addWidget(self.device_combo)
        device_row.addStretch(1)
        controls_layout.addLayout(device_row)

        controls_layout.addStretch(1)

        right_layout.addWidget(controls_group)

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
        self.order_combo.currentIndexChanged.connect(self._on_pipeline_changed)

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
        self.conventional_scale_combo.setCurrentIndex(0)
        self.conventional_scale_combo.blockSignals(False)

        default_ai_index: Optional[int] = None
        for index, upscaler in enumerate(self._ai_upscalers):
            if default_ai_index is None and 2 in upscaler.supported_scales():
                default_ai_index = index

        self.ai_combo.blockSignals(True)
        self.ai_combo.clear()
        for upscaler in self._ai_upscalers:
            self.ai_combo.addItem(upscaler.label, upscaler.id)
        if self.ai_combo.count():
            self.ai_combo.setCurrentIndex(default_ai_index or 0)
        self.ai_combo.blockSignals(False)

        ai_available = bool(self._ai_upscalers)
        self.ai_checkbox.setEnabled(ai_available)
        self._refresh_ai_scales(self.ai_combo, self.ai_scale_combo, preferred_scale=2)
        self._populate_device_controls()

        self.order_combo.blockSignals(True)
        self.order_combo.clear()
        self.order_combo.addItem("Conventional → AI", "conventional-first")
        self.order_combo.addItem("AI → Conventional", "ai-first")
        self.order_combo.setCurrentIndex(1)
        self.order_combo.blockSignals(False)

    # --------------------------------------------------------------- data-load
    def load_images(self, images: Sequence[WQVImage]) -> None:
        self._images = list(images)
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for image in self._images:
            item = QListWidgetItem(image.path.name)
            preview_image = image.image
            if image.kind == WQVImageKind.EXTERNAL_RGB:
                preview_image = image.image.copy()
                preview_image.thumbnail(
                    (self._EXTERNAL_THUMB_LIMIT, self._EXTERNAL_THUMB_LIMIT),
                    _LANCZOS_FILTER,
                )
            thumbnail = _pil_to_qpixmap(preview_image)
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
        self.cancel_active_upscale()
        self._images.clear()
        self.list_widget.clear()
        self._set_current_image(None)

    def last_upscaled_image(self) -> Optional[Image.Image]:
        if self._last_upscaled_image is None:
            return None
        return self._last_upscaled_image.copy()

    def last_pipeline_results(self) -> Dict[str, PipelineResult]:
        return {key: value for key, value in self._last_results.items()}

    def current_image_basename(self) -> Optional[str]:
        if self._current_image is None:
            return None
        stem = self._current_image.path.stem
        return stem or None

    def current_image(self) -> Optional[WQVImage]:
        return self._current_image

    def selected_images(self) -> List[WQVImage]:
        images: List[WQVImage] = []
        selected_indexes = sorted(self.list_widget.selectedIndexes(), key=lambda index: index.row())
        seen_rows: set[int] = set()
        for model_index in selected_indexes:
            row = model_index.row()
            if row in seen_rows:
                continue
            seen_rows.add(row)
            item = self.list_widget.item(row)
            if item is None:
                continue
            candidate = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(candidate, WQVImage):
                images.append(candidate)
        return images

    def run_pipeline_for_image(self, image: WQVImage) -> PipelineResult:
        pipeline_config = self._pipeline_config()
        pipeline = build_pipeline(
            pipeline_config,
            conventional_map=self._conventional_map,
            ai_map=self._ai_map,
        )
        device_policy = self._selected_device_policy()
        source = image.image.copy()
        return run_pipeline(
            pipeline,
            source,
            original_policy=device_policy,
        )

    def set_context_actions(self, actions: Sequence[QAction]) -> None:
        self.list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.ActionsContextMenu)
        for action in actions:
            self.list_widget.addAction(action)

    def save_state(self) -> Dict[str, Any]:
        return {
            "conventional_enabled": self.conventional_checkbox.isChecked(),
            "conventional_id": self.conventional_combo.currentData(),
            "conventional_scale": self.conventional_scale_combo.currentData(),
            "ai_enabled": self.ai_checkbox.isChecked(),
            "ai_id": self.ai_combo.currentData(),
            "ai_scale": self.ai_scale_combo.currentData(),
            "device_policy": self.device_combo.currentData(),
            "pipeline_order": self.order_combo.currentData(),
        }

    def restore_state(self, state: Mapping[str, Any]) -> None:
        if not state:
            return

        def _set_combo_data(combo: QComboBox, value: Any) -> None:
            if value is None:
                return
            index = combo.findData(value)
            if index >= 0:
                combo.blockSignals(True)
                combo.setCurrentIndex(index)
                combo.blockSignals(False)

        self._controls_ready = False
        try:
            conventional_enabled = bool(state.get("conventional_enabled", True))
            self.conventional_checkbox.setChecked(conventional_enabled)
            _set_combo_data(self.conventional_combo, state.get("conventional_id"))
            _set_combo_data(self.conventional_scale_combo, state.get("conventional_scale"))

            ai_enabled = bool(state.get("ai_enabled", False))
            self.ai_checkbox.setChecked(ai_enabled)
            _set_combo_data(self.ai_combo, state.get("ai_id"))

            preferred_ai_scale = state.get("ai_scale")
            if isinstance(preferred_ai_scale, int):
                self._refresh_ai_scales(
                    self.ai_combo,
                    self.ai_scale_combo,
                    preferred_scale=preferred_ai_scale,
                )
            else:
                self._refresh_ai_scales(self.ai_combo, self.ai_scale_combo)

            _set_combo_data(self.order_combo, state.get("pipeline_order"))

            device_policy = state.get("device_policy")
            if isinstance(device_policy, str):
                _set_combo_data(self.device_combo, device_policy)
        finally:
            self._controls_ready = True
            self._update_controls_enabled()
            self._apply_upscale()

    def current_selection_signature(self) -> Dict[str, str]:
        image = self._current_image
        if image is None:
            return {}

        signature: Dict[str, str] = {}
        metadata = image.metadata
        unique_id = metadata.get("record_unique_id")
        index = metadata.get("record_index")
        if unique_id:
            signature["record_unique_id"] = str(unique_id)
        if index:
            signature["record_index"] = str(index)
        signature["path_name"] = image.path.name
        return signature

    def select_by_signature(self, signature: Mapping[str, Any]) -> bool:
        if not signature:
            return False

        def _to_str(value: Any) -> Optional[str]:
            if value is None:
                return None
            text = str(value)
            return text or None

        unique_id = _to_str(signature.get("record_unique_id"))
        index = _to_str(signature.get("record_index"))
        path_name = _to_str(signature.get("path_name"))

        for row in range(self.list_widget.count()):
            item = self.list_widget.item(row)
            if item is None:
                continue
            candidate = item.data(Qt.ItemDataRole.UserRole)
            if not isinstance(candidate, WQVImage):
                continue
            metadata = candidate.metadata

            if unique_id and metadata.get("record_unique_id") != unique_id:
                continue
            if index and metadata.get("record_index") != index:
                continue
            if unique_id or index:
                if path_name and candidate.path.name != path_name:
                    continue
            else:
                if not path_name or candidate.path.name != path_name:
                    continue

            self.list_widget.setCurrentRow(row)
            self.list_widget.scrollToItem(item)
            return True

        return False

    def _update_metadata_panel(self, image: Optional[WQVImage]) -> None:
        if not self._metadata_labels:
            return

        def _set(key: str, value: str) -> None:
            label = self._metadata_labels.get(key)
            if label is not None:
                label.setText(value)

        if image is None:
            for label in self._metadata_labels.values():
                label.setText("—")
            return

        metadata = image.metadata
        _set("filename", image.path.name)

        captured = image.captured_at or metadata.get("captured_at") or "—"
        _set("captured", captured)

        resolution = f"{image.image.width}×{image.image.height}"
        _set("resolution", resolution)

        unique_id = metadata.get("record_unique_id")
        index = metadata.get("record_index")
        record_parts: List[str] = []
        if unique_id:
            record_parts.append(f"UID {unique_id}")
        if index:
            record_parts.append(f"Index {index}")
        record_text = ", ".join(record_parts) if record_parts else "—"
        _set("record", record_text)

    def metadata_snapshot(self) -> Dict[str, str]:
        return {key: label.text() for key, label in self._metadata_labels.items()}

    # -------------------------------------------------------------- selections
    def _on_selection_changed(self, *_args) -> None:
        item = self._first_selected_item()
        if item is None:
            self._set_current_image(None)
            return

        if self.list_widget.currentItem() is not item:
            self.list_widget.setCurrentItem(item, QItemSelectionModel.SelectionFlag.NoUpdate)

        image = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(image, WQVImage):
            self._set_current_image(image)
        else:
            self._set_current_image(None)

    def _first_selected_item(self) -> Optional[QListWidgetItem]:
        selected_indexes = self.list_widget.selectedIndexes()
        if not selected_indexes:
            return None
        first_index = min(selected_indexes, key=lambda index: index.row())
        return self.list_widget.item(first_index.row())

    def _set_current_image(self, image: Optional[WQVImage]) -> None:
        if image is not self._current_image:
            self.cancel_active_upscale()
        self._current_image = image
        self._last_upscaled_image = None
        if image is None:
            self.original_view.clear()
            self.upscaled_view.clear()
            self.primary_status.setText("Select an image from the list")
            self._update_metadata_panel(None)
        else:
            self.original_view.set_image(image)
            self._update_metadata_panel(image)
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
        self._refresh_ai_scales(self.ai_combo, self.ai_scale_combo)
        self._apply_upscale()

    def _update_controls_enabled(self) -> None:
        conventional_enabled = self.conventional_checkbox.isChecked()
        self.conventional_combo.setEnabled(conventional_enabled)
        self.conventional_scale_combo.setEnabled(conventional_enabled)

        ai_enabled = self.ai_checkbox.isChecked() and bool(self._ai_upscalers)
        self.ai_combo.setEnabled(ai_enabled)
        self.ai_scale_combo.setEnabled(ai_enabled)
        device_enabled = ai_enabled
        self.device_label.setEnabled(device_enabled)
        self.device_combo.setEnabled(device_enabled)
        order_enabled = conventional_enabled and ai_enabled
        self.order_label.setEnabled(order_enabled)
        self.order_combo.setEnabled(order_enabled)

    def _refresh_ai_scales(
        self,
        combo: QComboBox,
        scale_combo: QComboBox,
        *,
        preferred_scale: Optional[int] = None,
    ) -> None:
        scale_combo.blockSignals(True)
        previous = scale_combo.currentData()
        scale_combo.clear()
        upscaler = self._resolve_ai_upscaler(combo)
        if upscaler is not None:
            supported = list(upscaler.supported_scales())
            for scale in supported:
                scale_combo.addItem(f"{scale}×", scale)
            if preferred_scale in supported:
                index = supported.index(preferred_scale)
            else:
                if isinstance(previous, int) and previous in supported:
                    index = supported.index(previous)
                else:
                    index = 0
            if scale_combo.count():
                scale_combo.setCurrentIndex(index)
        scale_combo.blockSignals(False)

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

    def _resolve_conventional_upscaler(self, combo: QComboBox) -> Optional[Upscaler]:
        data = combo.currentData()
        if isinstance(data, str):
            return self._conventional_map.get(data)
        if self._conventional_upscalers:
            return self._conventional_upscalers[0]
        return None

    def _current_conventional_scale(self, scale_combo: QComboBox) -> int:
        data = scale_combo.currentData()
        if isinstance(data, int):
            return data
        return ALLOWED_CONVENTIONAL_SCALES[0]

    def _resolve_ai_upscaler(self, combo: QComboBox) -> Optional[Upscaler]:
        if not self._ai_upscalers:
            return None
        data = combo.currentData()
        if isinstance(data, str) and data in self._ai_map:
            return self._ai_map[data]
        return self._ai_upscalers[0]

    def _current_ai_scale(self, combo: QComboBox, scale_combo: QComboBox) -> Optional[int]:
        upscaler = self._resolve_ai_upscaler(combo)
        if upscaler is None:
            return None
        data = scale_combo.currentData()
        supported = list(upscaler.supported_scales())
        if isinstance(data, int) and data in supported:
            return data
        return supported[0] if supported else None

    def suggest_pipeline_preset_name(self, state: Mapping[str, Any]) -> str:
        def _sanitize(label: str) -> str:
            cleaned = "".join(ch for ch in label if ch.isalnum())
            return cleaned or "Preset"

        def _ai_segment() -> Optional[str]:
            if not state.get("ai_enabled"):
                return None
            ai_id = state.get("ai_id")
            ai_label = None
            if isinstance(ai_id, str):
                upscaler = self._ai_map.get(ai_id)
                ai_label = upscaler.label if upscaler else ai_id
            scale = state.get("ai_scale")
            label = _sanitize(ai_label or "AI")
            scale_text = f"{scale}x" if isinstance(scale, int) else "?x"
            return f"AI_{label}_{scale_text}"

        def _conv_segment() -> Optional[str]:
            if not state.get("conventional_enabled"):
                return None
            conv_id = state.get("conventional_id")
            conv_label = None
            if isinstance(conv_id, str):
                upscaler = self._conventional_map.get(conv_id)
                conv_label = upscaler.label if upscaler else conv_id
            scale = state.get("conventional_scale")
            label = _sanitize(conv_label or "Conv")
            scale_text = f"{scale}x" if isinstance(scale, int) else "?x"
            return f"Conv_{label}_{scale_text}"

        ai_segment = _ai_segment()
        conv_segment = _conv_segment()
        segments: List[str] = []
        order = state.get("pipeline_order") or "ai-first"
        if order == "ai-first":
            if ai_segment:
                segments.append(ai_segment)
            if conv_segment:
                segments.append(conv_segment)
        else:
            if conv_segment:
                segments.append(conv_segment)
            if ai_segment:
                segments.append(ai_segment)
        if not segments:
            return "DefaultPipeline"
        return "__".join(segments)

    def _pipeline_config(self) -> PipelineConfig:
        enable_conventional = self.conventional_checkbox.isChecked()
        enable_ai = self.ai_checkbox.isChecked() and bool(self._ai_upscalers)

        return PipelineConfig(
            enable_conventional=enable_conventional,
            conventional_id=self.conventional_combo.currentData() if enable_conventional else None,
            conventional_scale=self._current_conventional_scale(self.conventional_scale_combo),
            enable_ai=enable_ai,
            ai_id=self.ai_combo.currentData() if enable_ai else None,
            ai_scale=self._current_ai_scale(self.ai_combo, self.ai_scale_combo) if enable_ai else None,
            ai_before_conventional=(self.order_combo.currentData() == "ai-first"),
        )

    def _apply_upscale(self) -> None:
        if self._current_image is None:
            self.cancel_active_upscale()
            self.upscaled_view.clear()
            self.primary_status.setText("Select an image from the list")
            self._last_results.clear()
            self._last_upscaled_image = None
            return

        self._last_results.clear()
        self._last_upscaled_image = None

        source_image = self._current_image.image.copy()
        device_policy = self._selected_device_policy()
        pipeline_config = self._pipeline_config()
        pipeline = build_pipeline(
            pipeline_config,
            conventional_map=self._conventional_map,
            ai_map=self._ai_map,
        )

        if not pipeline:
            result = run_pipeline(
                pipeline,
                source_image,
                original_policy=device_policy,
            )
            self._apply_pipeline_result(result)
            self.upscale_completed.emit()
            return

        if not self._async_enabled:
            self.upscale_started.emit()
            self.primary_status.setText("Upscaling…")
            try:
                result = run_pipeline(
                    pipeline,
                    source_image,
                    original_policy=device_policy,
                )
            except Exception as exc:  # pragma: no cover - defensive guard
                self._handle_pipeline_failure(f"Upscale failed: {exc}")
                return
            self._apply_pipeline_result(result)
            self.upscale_completed.emit()
            return

        self._start_async_upscale(pipeline, source_image, device_policy)

    def _start_async_upscale(
        self,
        pipeline: Sequence[PipelineStage],
        source_image: Image.Image,
        device_policy: str,
    ) -> None:
        self.cancel_active_upscale()
        self._job_counter += 1
        job_id = self._job_counter
        self._active_job_id = job_id
        worker = PipelineWorker(job_id, pipeline, source_image, device_policy)
        thread = QThread(self)
        worker.moveToThread(thread)
        worker.finished.connect(self._handle_worker_finished)
        worker.failed.connect(self._handle_worker_failed)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)
        thread.started.connect(worker.run)
        self._workers[job_id] = (worker, thread)
        self.upscale_started.emit()
        self.primary_status.setText("Upscaling…")
        thread.start()

    def _handle_worker_finished(self, job_id: int, result: PipelineResult) -> None:
        worker_entry = self._workers.pop(job_id, None)
        if worker_entry:
            _, thread = worker_entry
            if thread.isRunning():
                thread.quit()

        cancelled_flag = self._cancelled_jobs.pop(job_id, None)
        if cancelled_flag is not None:
            if job_id == self._active_job_id:
                self._active_job_id = None
            if cancelled_flag:
                self.primary_status.setText("Upscale cancelled")
                self.upscale_cancelled.emit()
            return

        if job_id != self._active_job_id:
            return

        self._active_job_id = None
        self._apply_pipeline_result(result)
        self.upscale_completed.emit()

    def _handle_worker_failed(self, job_id: int, message: str) -> None:
        worker_entry = self._workers.pop(job_id, None)
        if worker_entry:
            _, thread = worker_entry
            if thread.isRunning():
                thread.quit()

        if job_id == self._active_job_id:
            self._active_job_id = None
            self._handle_pipeline_failure(message)
        else:
            logger.debug("Ignoring failure from stale job %s: %s", job_id, message)
        self._cancelled_jobs.pop(job_id, None)

    def cancel_active_upscale(self, *, user_requested: bool = False) -> bool:
        if self._active_job_id is None:
            return False
        job_id = self._active_job_id
        if user_requested:
            self._cancelled_jobs[job_id] = True
        else:
            self._cancelled_jobs.setdefault(job_id, False)
        if self._async_enabled:
            self.primary_status.setText("Cancelling…")
        return True

    def upscale_in_progress(self) -> bool:
        return self._active_job_id is not None and self._active_job_id not in self._cancelled_jobs

    def _apply_pipeline_result(self, result: PipelineResult) -> None:
        self.upscaled_view.set_image(result.image)
        self.primary_status.setText(self._format_pipeline_summary(result))
        self._last_results["primary"] = result
        self._last_upscaled_image = result.image.copy()

    def _handle_pipeline_failure(self, message: str) -> None:
        logger.debug("Pipeline failure: %s", message)
        self.upscaled_view.clear()
        self.primary_status.setText(message)
        self._last_results.clear()
        self._last_upscaled_image = None
        self.upscale_failed.emit(message)

    def current_zoom_mode(self) -> str:
        return self.upscaled_view.mode()

    def set_zoom_mode(self, mode: str) -> None:
        self.original_view.set_mode(mode)
        self.upscaled_view.set_mode(mode)

    def toggle_zoom_mode(self) -> str:
        new_mode = "actual" if self.current_zoom_mode() == "fit" else "fit"
        self.set_zoom_mode(new_mode)
        return new_mode

    def zoom_in(self) -> None:
        self.original_view.zoom_in()
        self.upscaled_view.zoom_in()

    def zoom_out(self) -> None:
        self.original_view.zoom_out()
        self.upscaled_view.zoom_out()

    def reset_zoom(self) -> None:
        self.original_view.reset_zoom()
        self.upscaled_view.reset_zoom()

    def save_layout_state(self) -> Dict[str, str]:
        state: Dict[str, str] = {}
        splitter_state = self.splitter.saveState().toBase64().data().decode("ascii")
        state["splitter"] = splitter_state
        state["zoom_mode"] = self.current_zoom_mode()
        return state

    def restore_layout_state(self, state: Mapping[str, Any]) -> None:
        if not state:
            return

        splitter_state = state.get("splitter")
        if isinstance(splitter_state, str) and splitter_state:
            try:
                self.splitter.restoreState(QByteArray.fromBase64(splitter_state.encode("ascii")))
            except Exception:  # pragma: no cover - defensive guard
                logger.debug("Failed to restore splitter state", exc_info=True)

        zoom_mode = state.get("zoom_mode")
        if isinstance(zoom_mode, str) and zoom_mode in {"fit", "actual"}:
            self.set_zoom_mode(zoom_mode)

    def _format_pipeline_summary(self, result: PipelineResult) -> str:
        summary = result.summary
        if result.fallback_used and result.applied_policy.lower() != result.original_policy.lower():
            summary += f" (fallback to {result.applied_policy.upper()})"
        return summary


class MainWindow(QMainWindow):
    """Main application window hosting the browser widget."""

    def __init__(
        self,
        *,
        settings: Optional[QSettings] = None,
        enable_async_upscale: bool = True,
    ) -> None:
        super().__init__()
        self.setWindowTitle("WQV Wristcam Viewer")
        self.resize(1100, 720)
        self._settings = settings or QSettings("RAHoebe", "WQVViewer")
        self._current_pdb_path: Optional[Path] = None
        self._last_loaded_pdb_path: Optional[Path] = None
        self._pending_selection_signature: Optional[Dict[str, str]] = None
        self._last_selection_signature: Dict[str, str] = {}
        self._recent_files: List[str] = []
        self._resource_icon_cache: Dict[str, QIcon] = {}
        self._zoom_icon_cache: Dict[str, QIcon] = {}
        self._preset_icon_cache: Dict[str, QIcon] = {}
        self._image_icon_cache: Dict[str, QIcon] = {}
        self._action_icons: Dict[QAction, Callable[[], QIcon] | str] = {}
        self._delete_disabled_reason: Optional[str] = None
        self._last_load_report: Optional[WQVLoadReport] = None
        self._pipeline_presets: List[Dict[str, Any]] = []

        self._apply_window_icon()

        self.browser = ImageBrowser(async_enabled=enable_async_upscale)
        self.setCentralWidget(self.browser)
        self.setAcceptDrops(True)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self._create_progress_widget()

        self._create_actions()
        self._build_menus()
        self._build_toolbar()
        self.browser.set_context_actions([self.delete_action])
        self.browser.list_widget.currentItemChanged.connect(self._record_selection_signature)
        self.browser.original_view.zoomModeChanged.connect(self._sync_zoom_actions)
        self.browser.upscaled_view.zoomModeChanged.connect(self._sync_zoom_actions)

        self._sync_zoom_actions()

        self._recent_files = self._load_recent_files()
        self._update_recent_menu()
        self._pipeline_presets = self._load_pipeline_presets()
        self._restore_session()

        self._connect_pipeline_signals()

    def _make_preset_icon(self, kind: str) -> QIcon:
        palette = self.palette()
        color = palette.color(QPalette.ColorRole.ButtonText)
        key = f"preset:{kind}:{color.rgba()}"
        cached = self._preset_icon_cache.get(key)
        if cached is not None:
            return cached

        size = 18
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(color)
        pen.setWidth(2)
        painter.setPen(pen)

        rect = QRectF(2, 3, size - 4, size - 6)
        painter.drawRoundedRect(rect, 3, 3)

        mid_x = rect.center().x()
        arrow_top = rect.top() + 3
        arrow_bottom = rect.bottom() - 3
        arrow_width = 4

        if kind == "save":
            painter.drawLine(QPointF(mid_x, arrow_top), QPointF(mid_x, arrow_bottom))
            painter.drawLine(
                QPointF(mid_x, arrow_bottom),
                QPointF(mid_x - arrow_width, arrow_bottom - arrow_width),
            )
            painter.drawLine(
                QPointF(mid_x, arrow_bottom),
                QPointF(mid_x + arrow_width, arrow_bottom - arrow_width),
            )
        else:  # load
            painter.drawLine(QPointF(mid_x, arrow_bottom), QPointF(mid_x, arrow_top))
            painter.drawLine(
                QPointF(mid_x, arrow_top),
                QPointF(mid_x - arrow_width, arrow_top + arrow_width),
            )
            painter.drawLine(
                QPointF(mid_x, arrow_top),
                QPointF(mid_x + arrow_width, arrow_top + arrow_width),
            )

        painter.end()
        icon = QIcon(pixmap)
        self._preset_icon_cache[key] = icon
        return icon

    def _make_open_image_icon(self) -> QIcon:
        palette = self.palette()
        frame_color = palette.color(QPalette.ColorRole.ButtonText)
        accent_color = palette.color(QPalette.ColorRole.Highlight)
        background_color = palette.color(QPalette.ColorRole.Base)
        key = f"open-image:{frame_color.rgba()}:{accent_color.rgba()}:{background_color.rgba()}"
        cached = self._image_icon_cache.get(key)
        if cached is not None:
            return cached

        size = 18
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        frame_pen = QPen(frame_color)
        frame_pen.setWidth(2)
        painter.setPen(frame_pen)
        painter.setBrush(background_color)
        outer_rect = QRectF(2, 3, size - 4, size - 5)
        painter.drawRoundedRect(outer_rect, 3, 3)

        # Draw a simple landscape thumbnail inside the frame.
        inner_rect = outer_rect.adjusted(3, 3, -3, -3)
        painter.setBrush(Qt.GlobalColor.transparent)
        painter.drawRect(inner_rect)

        mountain_pen = QPen(accent_color)
        mountain_pen.setWidth(2)
        painter.setPen(mountain_pen)
        painter.setBrush(Qt.GlobalColor.transparent)
        mountain_path = QPolygonF(
            [
                QPointF(inner_rect.left(), inner_rect.bottom() - 3),
                QPointF(inner_rect.center().x() - 2, inner_rect.center().y()),
                QPointF(inner_rect.center().x() + 2, inner_rect.bottom() - 3),
            ]
        )
        painter.drawPolyline(mountain_path)

        sun_radius = 2
        sun_center = QPointF(inner_rect.right() - 3, inner_rect.top() + 3)
        painter.setBrush(accent_color)
        painter.drawEllipse(sun_center, sun_radius, sun_radius)

        painter.end()

        icon = QIcon(pixmap)
        self._image_icon_cache[key] = icon
        return icon

    # ---------------------------------------------------------------- actions
    def _resources_dir(self) -> Path:
        return Path(__file__).resolve().parent.parent / "resources"

    def _resource_icon(self, name: str) -> QIcon:
        icon_path = self._resources_dir() / name
        key = str(icon_path.resolve())
        cached = self._resource_icon_cache.get(key)
        if cached is not None:
            return cached
        if not icon_path.exists():
            return QIcon()
        icon = QIcon(str(icon_path))
        if icon.isNull():
            return QIcon()
        self._resource_icon_cache[key] = icon
        return icon

    def _register_icon(self, action: QAction, source: Callable[[], QIcon] | str) -> None:
        self._action_icons[action] = source
        if callable(source):
            action.setIcon(source())
        else:
            action.setIcon(self._resource_icon(source))

    def _refresh_icons(self) -> None:
        for action, source in self._action_icons.items():
            if callable(source):
                action.setIcon(source())
            else:
                action.setIcon(self._resource_icon(source))

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
        self.open_pdb_action.triggered.connect(self.open_pdb)
        self._register_icon(self.open_pdb_action, "open.png")

        self.open_image_action = QAction("Open Image…", self)
        self.open_image_action.triggered.connect(self.open_images)
        self._register_icon(self.open_image_action, self._make_open_image_icon)

        self.export_action = QAction("Export Selected…", self)
        self.export_action.triggered.connect(self.export_selected)
        self._register_icon(self.export_action, "export.png")

        self.delete_action = QAction("Delete Selected", self)
        self.delete_action.triggered.connect(self.delete_selected)
        self.delete_action.setEnabled(False)
        self._register_icon(self.delete_action, "delete.png")

        self.clear_action = QAction("Clear", self)
        self.clear_action.triggered.connect(self.clear_browser)
        self._register_icon(self.clear_action, "clear.png")

        self.exit_action = QAction("Exit", self)
        self.exit_action.triggered.connect(self.close)
        self._register_icon(self.exit_action, "exit.png")

        self.zoom_fit_action = QAction("Fit to Window", self)
        self.zoom_fit_action.setCheckable(True)
        self.zoom_fit_action.triggered.connect(self._activate_zoom_fit)
        self._register_icon(self.zoom_fit_action, lambda: self._make_zoom_icon("fit"))

        self.zoom_actual_action = QAction("Actual Size (1:1)", self)
        self.zoom_actual_action.setCheckable(True)
        self.zoom_actual_action.triggered.connect(self._activate_zoom_actual)
        self._register_icon(self.zoom_actual_action, lambda: self._make_zoom_icon("actual"))

        self.zoom_mode_group = QActionGroup(self)
        self.zoom_mode_group.setExclusive(True)
        self.zoom_mode_group.addAction(self.zoom_fit_action)
        self.zoom_mode_group.addAction(self.zoom_actual_action)

        self.zoom_in_action = QAction("Zoom In", self)
        self.zoom_in_action.setShortcuts(QKeySequence.keyBindings(QKeySequence.StandardKey.ZoomIn))
        self.zoom_in_action.triggered.connect(self._zoom_in)
        self._register_icon(self.zoom_in_action, lambda: self._make_zoom_icon("in"))

        self.zoom_out_action = QAction("Zoom Out", self)
        self.zoom_out_action.setShortcuts(QKeySequence.keyBindings(QKeySequence.StandardKey.ZoomOut))
        self.zoom_out_action.triggered.connect(self._zoom_out)
        self._register_icon(self.zoom_out_action, lambda: self._make_zoom_icon("out"))

        self.zoom_reset_action = QAction("Reset Zoom", self)
        self.zoom_reset_action.setShortcuts([QKeySequence("Ctrl+0")])
        self.zoom_reset_action.triggered.connect(self._zoom_reset)
        self._register_icon(self.zoom_reset_action, lambda: self._make_zoom_icon("reset"))

        self.about_action = QAction("About WQV Viewer", self)
        self.about_action.triggered.connect(self._show_about_dialog)

        self.diagnostics_action = QAction("View Load Diagnostics…", self)
        self.diagnostics_action.triggered.connect(self._show_diagnostics_dialog)
        self.diagnostics_action.setEnabled(False)

        self.save_pipeline_preset_action = QAction("Save Pipeline Preset…", self)
        self.save_pipeline_preset_action.triggered.connect(self._save_pipeline_preset)
        self._register_icon(self.save_pipeline_preset_action, lambda: self._make_preset_icon("save"))

        self.load_pipeline_preset_action = QAction("Load Pipeline Preset…", self)
        self.load_pipeline_preset_action.triggered.connect(self._load_pipeline_preset_dialog)
        self._register_icon(self.load_pipeline_preset_action, lambda: self._make_preset_icon("load"))

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(self.open_pdb_action)
        file_menu.addAction(self.open_image_action)
        self.recent_menu = file_menu.addMenu("Open Recent")
        file_menu.addSeparator()
        file_menu.addAction(self.export_action)
        file_menu.addAction(self.delete_action)
        file_menu.addSeparator()
        file_menu.addAction(self.clear_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)

        view_menu = self.menuBar().addMenu("View")
        view_menu.addAction(self.zoom_fit_action)
        view_menu.addAction(self.zoom_actual_action)
        view_menu.addSeparator()
        view_menu.addAction(self.zoom_in_action)
        view_menu.addAction(self.zoom_out_action)
        view_menu.addAction(self.zoom_reset_action)

        pipeline_menu = self.menuBar().addMenu("Pipeline")
        pipeline_menu.addAction(self.save_pipeline_preset_action)
        pipeline_menu.addAction(self.load_pipeline_preset_action)

        help_menu = self.menuBar().addMenu("Help")
        help_menu.addAction(self.about_action)
        help_menu.addAction(self.diagnostics_action)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Main")
        toolbar.addAction(self.open_pdb_action)
        toolbar.addAction(self.open_image_action)
        toolbar.addAction(self.export_action)
        toolbar.addAction(self.delete_action)
        toolbar.addAction(self.save_pipeline_preset_action)
        toolbar.addAction(self.load_pipeline_preset_action)
        toolbar.addSeparator()
        toolbar.addAction(self.clear_action)
        toolbar.addSeparator()
        toolbar.addAction(self.zoom_fit_action)
        toolbar.addAction(self.zoom_actual_action)
        toolbar.addAction(self.zoom_in_action)
        toolbar.addAction(self.zoom_out_action)
        toolbar.addAction(self.zoom_reset_action)
        toolbar.addSeparator()
        toolbar.addAction(self.exit_action)
        self.addToolBar(toolbar)

    def _create_progress_widget(self) -> None:
        self._progress_widget = QWidget()
        layout = QHBoxLayout(self._progress_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)
        layout.addWidget(self._progress_bar)

        self._progress_cancel = QPushButton("Cancel")
        self._progress_cancel.clicked.connect(self._cancel_upscale)
        layout.addWidget(self._progress_cancel)

        self._progress_widget.hide()
        self.status_bar.addPermanentWidget(self._progress_widget)

    def _cancel_upscale(self) -> None:
        if self.browser.cancel_active_upscale(user_requested=True):
            self._progress_cancel.setEnabled(False)

    def _activate_zoom_fit(self) -> None:
        self.browser.set_zoom_mode("fit")
        self._sync_zoom_actions()

    def _activate_zoom_actual(self) -> None:
        self.browser.set_zoom_mode("actual")
        self._sync_zoom_actions()

    def _zoom_in(self) -> None:
        self.browser.zoom_in()
        self._sync_zoom_actions()

    def _zoom_out(self) -> None:
        self.browser.zoom_out()
        self._sync_zoom_actions()

    def _zoom_reset(self) -> None:
        self.browser.reset_zoom()
        self._sync_zoom_actions()

    def _sync_zoom_actions(self) -> None:
        mode = self.browser.current_zoom_mode()
        self.zoom_fit_action.setChecked(mode == "fit")
        self.zoom_actual_action.setChecked(mode == "actual")

    def _on_upscale_started(self) -> None:
        self._progress_cancel.setEnabled(True)
        self._progress_widget.show()

    def _hide_progress_widget(self) -> None:
        self._progress_widget.hide()
        self._progress_cancel.setEnabled(True)

    def _on_upscale_completed(self) -> None:
        self._hide_progress_widget()

    def _on_upscale_failed(self, message: str) -> None:
        self._hide_progress_widget()
        self.status_bar.showMessage(message, 5000)

    def _on_upscale_cancelled(self) -> None:
        self._hide_progress_widget()
        self.status_bar.showMessage("Upscale cancelled", 4000)

    def _show_about_dialog(self) -> None:
        content = (
            "<h3>WQV Wristcam Viewer</h3>"
            "<p>Explore wrist camera captures, upscale them with conventional and AI "
            "pipelines, and manage Palm databases directly.</p>"
            "<p>Documentation and source are available on the "
            '<a href="https://github.com/RAHoebe/WQV-Viewer">project homepage</a>.</p>'
            "<p>Upscaling models are provided by the Real-ESRGAN project; see the "
            '<a href="https://github.com/xinntao/Real-ESRGAN">model licenses</a>.</p>'
            "<p>Application logs live alongside your session settings under the "
            "WQVViewer profile directory.</p>"
        )
        QMessageBox.about(self, "About WQV Viewer", content)

    def _make_zoom_icon(self, kind: str) -> QIcon:
        palette = self.palette()
        color = palette.color(QPalette.ColorRole.ButtonText)
        key = f"{kind}:{color.rgba()}"
        cached = self._zoom_icon_cache.get(key)
        if cached is not None:
            return cached

        base_size = 18
        pixmap = QPixmap(base_size, base_size)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(color)
        pen.setWidth(2 if base_size >= 20 else 1)
        painter.setPen(pen)

        rect = QRectF(3, 3, base_size - 6, base_size - 6)

        if kind == "fit":
            painter.drawRect(rect)
            arrow_pen = QPen(color)
            arrow_pen.setWidth(2)
            painter.setPen(arrow_pen)
            margin = 4
            top_left = rect.topLeft()
            bottom_right = rect.bottomRight()
            painter.drawLine(top_left, QPointF(top_left.x() + margin, top_left.y()))
            painter.drawLine(top_left, QPointF(top_left.x(), top_left.y() + margin))
            painter.drawLine(bottom_right, QPointF(bottom_right.x() - margin, bottom_right.y()))
            painter.drawLine(bottom_right, QPointF(bottom_right.x(), bottom_right.y() - margin))
        elif kind == "actual":
            font = painter.font()
            font.setBold(True)
            font.setPointSizeF(9)
            painter.setFont(font)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "1:1")
        else:
            radius = rect.width() * 0.4
            center = rect.center()
            ellipse_rect = QRectF(
                center.x() - radius,
                center.y() - radius,
                radius * 2,
                radius * 2,
            )
            painter.drawEllipse(ellipse_rect)
            handle_start = QPointF(
                center.x() + radius * 0.65,
                center.y() + radius * 0.65,
            )
            handle_end = QPointF(
                center.x() + radius * 1.15,
                center.y() + radius * 1.15,
            )
            painter.drawLine(handle_start, handle_end)
            if kind == "in":
                painter.drawLine(QPointF(center.x() - radius * 0.45, center.y()), QPointF(center.x() + radius * 0.45, center.y()))
                painter.drawLine(QPointF(center.x(), center.y() - radius * 0.45), QPointF(center.x(), center.y() + radius * 0.45))
            elif kind == "out":
                painter.drawLine(QPointF(center.x() - radius * 0.45, center.y()), QPointF(center.x() + radius * 0.45, center.y()))
            elif kind == "reset":
                painter.drawLine(QPointF(center.x() - radius * 0.45, center.y()), QPointF(center.x() + radius * 0.45, center.y()))
                font = painter.font()
                font.setPointSizeF(7)
                painter.setFont(font)
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "100")

        painter.end()

        icon = QIcon(pixmap)
        self._zoom_icon_cache[key] = icon
        return icon

    # --------------------------------------------------------------- file ops
    def open_pdb(self) -> None:
        if self._current_pdb_path is not None:
            start_dir = self._current_pdb_path.parent
        elif self._last_loaded_pdb_path is not None:
            start_dir = self._last_loaded_pdb_path.parent
        else:
            start_dir = Path.home()
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open WQV Palm database",
            str(start_dir),
            "WQV Palm database (*.pdb);;All files (*)",
        )
        if not path:
            return
        count = self._load_pdb(Path(path))
        self.status_bar.showMessage(f"Loaded {count} images from {Path(path).name}", 5000)

    def open_images(self) -> None:
        start_dir_value = self._settings.value("session/lastImageDirectory", "")
        if isinstance(start_dir_value, str) and start_dir_value:
            start_dir = Path(start_dir_value)
        elif self._current_pdb_path is not None:
            start_dir = self._current_pdb_path.parent
        elif self._last_loaded_pdb_path is not None:
            start_dir = self._last_loaded_pdb_path.parent
        else:
            start_dir = Path.home()

        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open PNG or JPEG",
            str(start_dir),
            "Images (*.png *.jpg *.jpeg);;All files (*)",
        )
        if not paths:
            return

        self._ingest_image_files([Path(entry) for entry in paths], allow_popups=True)

    def _ingest_image_files(self, paths: Sequence[Path], *, allow_popups: bool) -> bool:
        if not paths:
            return False

        images: List[WQVImage] = []
        errors: List[str] = []
        for candidate in paths:
            try:
                images.append(load_wqv_image(candidate, prefer_external=True))
            except Exception as exc:  # pragma: no cover - runtime guard
                logger.warning("Failed to load %s: %s", candidate, exc, exc_info=True)
                errors.append(f"{candidate.name}: {exc}")

        if not images:
            message = "None of the provided files could be opened."
            if allow_popups:
                QMessageBox.warning(self, "Load images", message)
            else:
                self.status_bar.showMessage(message, 5000)
            return False

        palm_fallbacks = [
            img.path.name
            for img in images
            if img.metadata.get("decoder") == "wqv_color" and img.metadata.get("preferred_external") == "true"
        ]

        self.browser.load_images(images)
        self._current_pdb_path = None
        self._set_delete_action_enabled(
            False,
            reason="Deletion is only available for Palm database records.",
        )
        self._update_diagnostics_state(None)
        self._record_selection_signature()

        self._settings.setValue("session/lastImageDirectory", str(paths[0].parent))
        base_message = f"Loaded {len(images)} standalone image{'s' if len(images) != 1 else ''}"
        if palm_fallbacks:
            preview = ", ".join(palm_fallbacks[:3])
            if len(palm_fallbacks) > 3:
                preview += f" …(+{len(palm_fallbacks) - 3})"
            base_message += f" (Palm decoder used for {preview})"
        self.status_bar.showMessage(base_message, 5000)

        if errors:
            summary = "\n".join(errors[:5])
            if len(errors) > 5:
                summary += f"\n…and {len(errors) - 5} more"
            if allow_popups:
                QMessageBox.warning(
                    self,
                    "Some files were skipped",
                    summary,
                )
            else:
                self.status_bar.showMessage(summary, 6000)
        return True

    def _paths_from_mime(self, mime_data: QMimeData | None) -> tuple[List[Path], List[Path]]:
        pdbs: List[Path] = []
        images: List[Path] = []
        if mime_data is None or not mime_data.hasUrls():
            return pdbs, images
        for url in mime_data.urls():
            if not isinstance(url, QUrl) or not url.isLocalFile():
                continue
            try:
                candidate = Path(url.toLocalFile()).resolve()
            except Exception:
                continue
            if not candidate.exists():
                continue
            suffix = candidate.suffix.lower()
            if suffix == ".pdb":
                pdbs.append(candidate)
            elif suffix in {".png", ".jpg", ".jpeg", ".jpe"}:
                images.append(candidate)
        return pdbs, images

    def _open_dropped_database(self, path: Path) -> None:
        try:
            count = self._load_pdb(path)
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.warning("Failed to open dropped database %s: %s", path, exc, exc_info=True)
            self.status_bar.showMessage(f"Failed to open {path.name}: {exc}", 6000)
            return
        suffix = "image" if count == 1 else "images"
        self.status_bar.showMessage(f"Loaded {count} {suffix} from {path.name}", 5000)

    def export_selected(self) -> None:
        selected = self.browser.selected_images()
        if not selected:
            self.status_bar.showMessage("Select one or more images to export", 4000)
            return

        start_dir = self._settings.value("session/lastExportDirectory", "")
        if isinstance(start_dir, str) and start_dir:
            initial_directory = Path(start_dir)
        else:
            initial_directory = Path.home()

        directory_str = QFileDialog.getExistingDirectory(
            self,
            "Select export folder",
            str(initial_directory),
        )
        if not directory_str:
            return

        destination_dir = Path(directory_str)
        try:
            destination_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self.status_bar.showMessage(f"Cannot use {destination_dir}: {exc}", 6000)
            return

        used_names: set[str] = set()
        successes = 0
        failures: List[str] = []

        for index, image in enumerate(selected, start=1):
            try:
                result = self.browser.run_pipeline_for_image(image)
            except Exception as exc:
                logger.error("Failed to upscale %s during export: %s", image.path, exc, exc_info=True)
                failures.append(f"{image.path.name}: {exc}")
                continue

            destination = self._build_export_destination(destination_dir, image, used_names, index=index)
            try:
                export_result(
                    destination,
                    image,
                    result,
                    extra_metadata={"pipeline_name": "primary"},
                )
            except Exception as exc:
                logger.error(
                    "Failed to export %s to %s: %s",
                    image.path,
                    destination,
                    exc,
                    exc_info=True,
                )
                failures.append(f"{image.path.name}: {exc}")
                continue

            successes += 1

        if successes:
            self._settings.setValue("session/lastExportDirectory", str(destination_dir))
            self._settings.sync()

        if successes and not failures:
            suffix = "s" if successes != 1 else ""
            self.status_bar.showMessage(f"Exported {successes} image{suffix} to {destination_dir}", 5000)
        elif successes and failures:
            suffix = "s" if successes != 1 else ""
            self.status_bar.showMessage(
                f"Exported {successes} image{suffix} to {destination_dir} ({len(failures)} failed)",
                6000,
            )
        else:
            message = failures[0] if failures else "Failed to export images"
            self.status_bar.showMessage(message, 6000)

    def _build_export_destination(
        self,
        directory: Path,
        image: WQVImage,
        used_names: set[str],
        *,
        index: int,
    ) -> Path:
        stem = image.path.stem or image.path.name or ""
        stem = stem.strip().strip(".")
        if not stem:
            stem = f"image_{index:03d}"
        candidate = directory / f"{stem}.png"
        counter = 1
        while (
            candidate.exists()
            or candidate.with_suffix(".json").exists()
            or candidate.name.lower() in used_names
            or candidate.with_suffix(".json").name.lower() in used_names
        ):
            candidate = directory / f"{stem}_{counter:03d}.png"
            counter += 1
        used_names.add(candidate.name.lower())
        used_names.add(candidate.with_suffix(".json").name.lower())
        return candidate

    def _set_delete_action_enabled(self, enabled: bool, *, reason: Optional[str] = None) -> None:
        self.delete_action.setEnabled(enabled)
        self._delete_disabled_reason = None if enabled else reason

    def _update_diagnostics_state(self, report: Optional[WQVLoadReport]) -> None:
        self._last_load_report = report
        warnings = report.warnings if report else []
        self.diagnostics_action.setEnabled(bool(warnings))
        if warnings:
            summary = warnings[0]
            if len(warnings) > 1:
                summary = f"{summary} (+{len(warnings) - 1} more warnings)"
            self.status_bar.showMessage(summary, 8000)

    def _show_diagnostics_dialog(self) -> None:
        report = self._last_load_report
        warnings = report.warnings if report else []
        if not warnings:
            QMessageBox.information(self, "Load diagnostics", "No warnings for the last load.")
            return
        text = "\n".join(warnings)
        QMessageBox.information(
            self,
            "Load diagnostics",
            text,
        )

    def _is_color_database(self, images: Sequence[WQVImage]) -> bool:
        for image in images:
            metadata = getattr(image, "metadata", {})
            if isinstance(metadata, dict) and metadata.get("source_color_db"):
                return True
        return False

    def _refresh_delete_action(self, images: Sequence[WQVImage]) -> None:
        if not images:
            self._set_delete_action_enabled(False)
            return
        if self._is_color_database(images):
            reason = "Deletion disabled for color databases; remove CAS archives manually."
            self._set_delete_action_enabled(False, reason=reason)
            self.status_bar.showMessage(reason, 6000)
        else:
            self._set_delete_action_enabled(True)

    # ------------------------------------------------------------ test hooks
    def _load_pdb(self, path: Path) -> int:
        resolved = path.resolve()
        report = WQVLoadReport()
        images = load_wqv_pdb(resolved, report=report)
        self.browser.load_images(images)
        self._current_pdb_path = resolved
        self._last_loaded_pdb_path = resolved
        self._refresh_delete_action(images)
        self._update_diagnostics_state(report)
        self._record_last_database(resolved)
        self._add_to_recent_list(resolved)
        self._apply_pending_selection()
        self._record_selection_signature()
        return len(images)

    def clear_browser(self) -> None:
        self.browser.clear()
        self._current_pdb_path = None
        self._set_delete_action_enabled(False)
        self._update_diagnostics_state(None)
        self._record_selection_signature()

    def _export_upscaled_image(self, path: Path) -> bool:
        results = self.browser.last_pipeline_results()
        result = results.get("primary")
        source = self.browser.current_image()
        if result is None or source is None:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        destination = path if path.suffix.lower() == ".png" else path.with_suffix(".png")
        export_result(destination, source, result, extra_metadata={"pipeline_name": "primary"})
        return True

    def _record_last_database(self, path: Path) -> None:
        self._settings.setValue("session/lastDatabase", str(path))
        self._settings.sync()

    def _add_to_recent_list(self, path: Path) -> None:
        entry = str(path)
        self._recent_files = [candidate for candidate in self._recent_files if candidate != entry]
        self._recent_files.insert(0, entry)
        if len(self._recent_files) > 10:
            self._recent_files = self._recent_files[:10]
        self._update_recent_menu()
        self._save_recent_files(commit=True)

    def _remove_from_recent(self, entry: str) -> None:
        previous = list(self._recent_files)
        self._recent_files = [candidate for candidate in self._recent_files if candidate != entry]
        if self._recent_files != previous:
            self._update_recent_menu()
            self._save_recent_files(commit=True)

    def _load_recent_files(self) -> List[str]:
        data = self._load_json_setting("session/recentFiles")
        if not isinstance(data, list):
            return []
        unique: List[str] = []
        for entry in data:
            if isinstance(entry, str) and entry and entry not in unique:
                unique.append(entry)
            if len(unique) >= 10:
                break
        return unique

    def _save_recent_files(self, *, commit: bool = False) -> None:
        self._settings.setValue("session/recentFiles", json.dumps(self._recent_files[:10]))
        if commit:
            self._settings.sync()

    def _load_pipeline_presets(self) -> List[Dict[str, Any]]:
        data = self._load_json_setting("session/pipelinePresets")
        if not isinstance(data, list):
            return []
        presets: List[Dict[str, Any]] = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            state = entry.get("state")
            if isinstance(name, str) and isinstance(state, dict):
                presets.append({"name": name, "state": state})
        return presets[:15]

    def _persist_pipeline_presets(self) -> None:
        self._settings.setValue("session/pipelinePresets", json.dumps(self._pipeline_presets))
        self._settings.sync()

    def _save_pipeline_preset(self) -> None:
        state = self.browser.save_state()
        suggestion = self.browser.suggest_pipeline_preset_name(state)
        name, accepted = QInputDialog.getText(
            self,
            "Save pipeline preset",
            "Preset name:",
            text=suggestion,
        )
        if not accepted:
            return
        preset_name = name.strip()
        if not preset_name:
            QMessageBox.warning(self, "Save pipeline preset", "Preset name cannot be empty.")
            return

        existing = next((entry for entry in self._pipeline_presets if entry["name"].lower() == preset_name.lower()), None)
        if existing:
            response = QMessageBox.question(
                self,
                "Overwrite preset",
                f"Preset '{preset_name}' already exists. Overwrite?",
            )
            if response != QMessageBox.StandardButton.Yes:
                return
            existing["state"] = state
        else:
            self._pipeline_presets.insert(0, {"name": preset_name, "state": state})
            if len(self._pipeline_presets) > 15:
                self._pipeline_presets = self._pipeline_presets[:15]
        self._persist_pipeline_presets()
        self.status_bar.showMessage(f"Saved preset '{preset_name}'", 4000)

    def _load_pipeline_preset_dialog(self) -> None:
        if not self._pipeline_presets:
            QMessageBox.information(self, "Load pipeline preset", "No presets available yet.")
            return
        names = [entry["name"] for entry in self._pipeline_presets]
        selection, accepted = QInputDialog.getItem(
            self,
            "Load pipeline preset",
            "Choose preset:",
            names,
            0,
            False,
        )
        if not accepted:
            return
        preset_name = selection
        target = next((entry for entry in self._pipeline_presets if entry["name"] == preset_name), None)
        if target is None:
            QMessageBox.warning(self, "Load pipeline preset", "Selected preset no longer exists.")
            return
        self.browser.restore_state(target["state"])
        self.status_bar.showMessage(f"Loaded preset '{preset_name}'", 4000)

    def _update_recent_menu(self) -> None:
        if not hasattr(self, "recent_menu"):
            return
        self.recent_menu.clear()
        if not self._recent_files:
            placeholder = self.recent_menu.addAction("No recent files")
            placeholder.setEnabled(False)
            return
        for entry in self._recent_files:
            action = self.recent_menu.addAction(entry)
            action.triggered.connect(partial(self._open_recent, entry))

    def _open_recent(self, entry: str) -> None:
        path = Path(entry)
        if not path.exists():
            self.status_bar.showMessage(f"Cannot open {path.name}: file missing", 5000)
            self._remove_from_recent(entry)
            return
        count = self._load_pdb(path)
        self.status_bar.showMessage(f"Loaded {count} images from {path.name}", 5000)

    def _apply_pending_selection(self) -> None:
        if not self._pending_selection_signature:
            return
        signature = self._pending_selection_signature
        self._pending_selection_signature = None
        if self.browser.select_by_signature(signature):
            self._record_selection_signature()

    def _record_selection_signature(self, *_args) -> None:
        signature = self.browser.current_selection_signature()
        self._last_selection_signature = signature
        self._settings.setValue("session/selectedImage", json.dumps(signature))

    def _connect_pipeline_signals(self) -> None:
        signals = [
            self.browser.conventional_checkbox.toggled,
            self.browser.conventional_combo.currentIndexChanged,
            self.browser.conventional_scale_combo.currentIndexChanged,
            self.browser.ai_checkbox.toggled,
            self.browser.ai_combo.currentIndexChanged,
            self.browser.ai_scale_combo.currentIndexChanged,
            self.browser.device_combo.currentIndexChanged,
        ]
        for signal in signals:
            signal.connect(self._store_pipeline_state)
        self._store_pipeline_state()

        self.browser.upscale_started.connect(self._on_upscale_started)
        self.browser.upscale_completed.connect(self._on_upscale_completed)
        self.browser.upscale_failed.connect(self._on_upscale_failed)
        self.browser.upscale_cancelled.connect(self._on_upscale_cancelled)

    def _store_pipeline_state(self, *_args) -> None:
        state = self.browser.save_state()
        self._settings.setValue("session/pipeline", json.dumps(state))

    def _load_json_setting(self, key: str) -> Any:
        raw = self._settings.value(key)
        if raw in (None, ""):
            return None
        if isinstance(raw, bytes):
            try:
                raw = raw.decode("utf-8")
            except Exception:
                return None
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return None
        return None

    def _restore_session(self) -> None:
        geometry_value = self._settings.value("session/windowGeometry", "")
        if isinstance(geometry_value, (bytes, bytearray)):
            try:
                geometry_value = geometry_value.decode("ascii")
            except Exception:
                geometry_value = ""
        if isinstance(geometry_value, str) and geometry_value:
            try:
                geometry = QByteArray.fromBase64(geometry_value.encode("ascii"))
                if not geometry.isEmpty():
                    self.restoreGeometry(geometry)
            except Exception:  # pragma: no cover - defensive guard
                logger.debug("Failed to restore window geometry", exc_info=True)

        layout_state = self._load_json_setting("session/browserLayout")

        pipeline_state = self._load_json_setting("session/pipeline")
        if isinstance(pipeline_state, dict):
            self.browser.restore_state(pipeline_state)

        selection_state = self._load_json_setting("session/selectedImage")
        if isinstance(selection_state, dict):
            self._pending_selection_signature = {
                key: str(value)
                for key, value in selection_state.items()
                if isinstance(key, str)
            }
            self._last_selection_signature = dict(self._pending_selection_signature)
        else:
            self._pending_selection_signature = None
            self._last_selection_signature = {}

        last_database = self._settings.value("session/lastDatabase", "")
        if isinstance(last_database, str) and last_database:
            path = Path(last_database)
            if path.exists():
                try:
                    count = self._load_pdb(path)
                    if count == 0:
                        self.status_bar.showMessage(
                            f"Loaded 0 images from {path.name}",
                            5000,
                        )
                except Exception as exc:  # pragma: no cover - defensive guard
                    logger.warning("Failed to restore last database %s: %s", path, exc)
            else:
                self.status_bar.showMessage(
                    f"Last database {path.name} not found",
                    5000,
                )
                self._remove_from_recent(str(path))

        if isinstance(layout_state, dict):
            self.browser.restore_layout_state(layout_state)
        self._sync_zoom_actions()

    def _save_session_state(self) -> None:
        self._store_pipeline_state()
        self._settings.setValue(
            "session/selectedImage",
            json.dumps(self.browser.current_selection_signature()),
        )
        last_database = str(self._last_loaded_pdb_path) if self._last_loaded_pdb_path else ""
        self._settings.setValue("session/lastDatabase", last_database)
        geometry = self.saveGeometry().toBase64().data().decode("ascii")
        self._settings.setValue("session/windowGeometry", geometry)
        self._settings.setValue(
            "session/browserLayout",
            json.dumps(self.browser.save_layout_state()),
        )
        self._save_recent_files()
        self._settings.sync()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # type: ignore[override]
        pdb_paths, image_paths = self._paths_from_mime(event.mimeData())
        if pdb_paths or image_paths:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:  # type: ignore[override]
        pdb_paths, image_paths = self._paths_from_mime(event.mimeData())
        if pdb_paths or image_paths:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:  # type: ignore[override]
        pdb_paths, image_paths = self._paths_from_mime(event.mimeData())
        if pdb_paths:
            event.acceptProposedAction()
            self._open_dropped_database(pdb_paths[0])
            return
        if image_paths:
            event.acceptProposedAction()
            self._ingest_image_files(image_paths, allow_popups=False)
            return
        event.ignore()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._save_session_state()
        super().closeEvent(event)

    def changeEvent(self, event: QEvent) -> None:  # type: ignore[override]
        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange:
            self._zoom_icon_cache.clear()
            self._refresh_icons()

    def delete_selected(self) -> None:
        pdb_path = self._current_pdb_path
        if pdb_path is None:
            self.status_bar.showMessage("No Palm database loaded", 4000)
            return

        if not self.delete_action.isEnabled():
            reason = self._delete_disabled_reason or "Deletion is currently disabled"
            self.status_bar.showMessage(reason, 5000)
            return

        selected = self.browser.selected_images()
        if not selected:
            self.status_bar.showMessage("Select one or more images to delete", 4000)
            return

        metadata_paths = {
            Path(image.metadata.get("source_pdb", "")).resolve()
            for image in selected
            if image.metadata.get("source_pdb")
        }
        if not metadata_paths:
            self.status_bar.showMessage("Selected images are not backed by a Palm database", 5000)
            return

        if len(metadata_paths) > 1 or next(iter(metadata_paths)) != pdb_path:
            self.status_bar.showMessage("Selections span multiple databases; delete them separately", 6000)
            return

        selectors = []
        for image in selected:
            meta = image.metadata
            unique_raw = meta.get("record_unique_id")
            index_raw = meta.get("record_index")
            unique_id = None
            index = None
            try:
                if unique_raw is not None:
                    unique_id = int(unique_raw)
            except ValueError:
                unique_id = None
            try:
                if index_raw is not None:
                    index = int(index_raw)
            except ValueError:
                index = None
            selector_key = (unique_id if unique_id else None, index)
            if selector_key == (None, None):
                continue
            selectors.append(selector_key)

        if not selectors:
            self.status_bar.showMessage("Unable to determine Palm records for deletion", 6000)
            return

        count = len(selectors)
        response = QMessageBox.question(
            self,
            "Delete images",
            f"Delete {count} selected image{'s' if count != 1 else ''} from the database?",
        )
        if response != QMessageBox.StandardButton.Yes:
            return

        removed = delete_wqv_pdb_records(pdb_path, selectors)
        if removed == 0:
            self.status_bar.showMessage("No records were deleted", 4000)
            return

        report = WQVLoadReport()
        images = load_wqv_pdb(pdb_path, report=report)
        self.browser.load_images(images)
        self._refresh_delete_action(images)
        self._update_diagnostics_state(report)
        self.status_bar.showMessage(f"Deleted {removed} record{'s' if removed != 1 else ''}", 5000)


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
