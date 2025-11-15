"""PyQt GUI for browsing and upscaling WQV wrist camera captures."""

from __future__ import annotations

import json
import logging
import sys
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from PIL import Image
from PyQt6.QtCore import Qt, QSize, QSettings
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
    QMessageBox,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .exporter import export_result
from .parser import WQVImage, delete_wqv_pdb_records, load_wqv_pdb
from .pipeline import PipelineConfig, PipelineResult, PipelineStage, build_pipeline, run_pipeline
from .upscaling import ALLOWED_CONVENTIONAL_SCALES, Upscaler, ai_upscalers, conventional_upscalers


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
        self._last_results: Dict[str, PipelineResult] = {}
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

        self.primary_status = QLabel("Pipeline disabled")
        self.primary_status.setFrameShape(QFrame.Shape.StyledPanel)
        self.primary_status.setMinimumHeight(32)
        right_layout.addWidget(self.primary_status)

        controls_group = QGroupBox("Upscaling controls")
        controls_layout = QVBoxLayout(controls_group)

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
        for item in self.list_widget.selectedItems():
            candidate = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(candidate, WQVImage):
                images.append(candidate)
        return images

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
            self.primary_status.setText("Select an image from the list")
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

    def _device_policy_label(self, policy: str) -> str:
        mapping = {"auto": "Auto", "gpu": "GPU", "cpu": "CPU"}
        return mapping.get(policy, policy.upper())

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
        )

    def _apply_upscale(self) -> None:
        if self._current_image is None:
            self.upscaled_label.clear()
            self.primary_status.setText("Select an image from the list")
            self._last_results.clear()
            self._last_upscaled_image = None
            return

        self._last_results.clear()
        self._last_upscaled_image = None
        source_image = self._current_image.image.copy()
        device_policy = self._selected_device_policy()

        pipeline = build_pipeline(
            self._pipeline_config(),
            conventional_map=self._conventional_map,
            ai_map=self._ai_map,
        )

        result, error = self._run_pipeline_with_feedback(
            pipeline,
            source_image,
            device_policy,
        )
        if result is None:
            self.upscaled_label.clear()
            self.primary_status.setText(error or "Upscale failed")
            return

        self.upscaled_label.set_image(result.image)
        self.primary_status.setText(self._format_pipeline_summary(result))
        self._last_results["primary"] = result
        self._last_upscaled_image = result.image.copy()

    def _run_pipeline_with_feedback(
        self,
        pipeline: Sequence[PipelineStage],
        source_image: Image.Image,
        device_policy: str,
        *,
        show_cursor: bool = True,
    ) -> tuple[Optional[PipelineResult], Optional[str]]:
        show_wait_cursor = show_cursor and bool(pipeline)
        if show_wait_cursor:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            result = run_pipeline(pipeline, source_image.copy(), original_policy=device_policy)
        except Exception as exc:  # pragma: no cover - GUI feedback path
            logger.debug("Pipeline execution failed: %s", exc, exc_info=True)
            return None, f"Upscale failed: {exc}"
        finally:
            if show_wait_cursor:
                QApplication.restoreOverrideCursor()

        return result, None

    def _format_pipeline_summary(self, result: PipelineResult) -> str:
        summary = result.summary
        if result.fallback_used and result.applied_policy.lower() != result.original_policy.lower():
            summary += f" (fallback to {result.applied_policy.upper()})"
        return summary

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

    def __init__(self, *, settings: Optional[QSettings] = None) -> None:
        super().__init__()
        self.setWindowTitle("WQV Wristcam Viewer")
        self.resize(1100, 720)
        self._apply_window_icon()
        self._settings = settings or QSettings("RAHoebe", "WQVViewer")
        self._current_pdb_path: Optional[Path] = None
        self._last_loaded_pdb_path: Optional[Path] = None
        self._pending_selection_signature: Optional[Dict[str, str]] = None
        self._last_selection_signature: Dict[str, str] = {}
        self._recent_files: List[str] = []

        self.browser = ImageBrowser()
        self.setCentralWidget(self.browser)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self._create_actions()
        self._build_menus()
        self._build_toolbar()
        self.browser.set_context_actions([self.delete_action])
        self.browser.list_widget.currentItemChanged.connect(self._record_selection_signature)

        self._recent_files = self._load_recent_files()
        self._update_recent_menu()
        self._restore_session()

        self._connect_pipeline_signals()

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

        self.delete_action = QAction("Delete Selected", self)
        self.delete_action.setIcon(self._resource_icon("delete.png"))
        self.delete_action.triggered.connect(self.delete_selected)
        self.delete_action.setEnabled(False)

        self.clear_action = QAction("Clear", self)
        self.clear_action.setIcon(self._resource_icon("clear.png"))
        self.clear_action.triggered.connect(self.clear_browser)

        self.exit_action = QAction("Exit", self)
        self.exit_action.setIcon(self._resource_icon("exit.png"))
        self.exit_action.triggered.connect(self.close)

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(self.open_pdb_action)
        self.recent_menu = file_menu.addMenu("Open Recent")
        file_menu.addSeparator()
        file_menu.addAction(self.export_action)
        file_menu.addAction(self.delete_action)
        file_menu.addSeparator()
        file_menu.addAction(self.clear_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Main")
        toolbar.addAction(self.open_pdb_action)
        toolbar.addAction(self.export_action)
        toolbar.addAction(self.delete_action)
        toolbar.addSeparator()
        toolbar.addAction(self.clear_action)
        toolbar.addSeparator()
        toolbar.addAction(self.exit_action)
        self.addToolBar(toolbar)

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

    def export_selected(self) -> None:
        results = self.browser.last_pipeline_results()
        if not results:
            self.status_bar.showMessage("No upscaled image to export", 3000)
            return

        source = self.browser.current_image()
        if source is None:
            self.status_bar.showMessage("No source image available", 3000)
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

        result = results.get("primary")
        if result is None:
            self.status_bar.showMessage("No upscaled preview available", 3000)
            return

        try:
            export_result(
                path,
                source,
                result,
                extra_metadata={"pipeline_name": "primary"},
            )
        except Exception as exc:
            self.status_bar.showMessage(f"Failed to save image: {exc}", 5000)
            return

        self.status_bar.showMessage(f"Saved {path.name}", 5000)

    # ------------------------------------------------------------ test hooks
    def _load_pdb(self, path: Path) -> int:
        resolved = path.resolve()
        images = load_wqv_pdb(resolved)
        self.browser.load_images(images)
        self._current_pdb_path = resolved
        self._last_loaded_pdb_path = resolved
        self.delete_action.setEnabled(bool(images))
        self._record_last_database(resolved)
        self._add_to_recent_list(resolved)
        self._apply_pending_selection()
        self._record_selection_signature()
        return len(images)

    def clear_browser(self) -> None:
        self.browser.clear()
        self._current_pdb_path = None
        self.delete_action.setEnabled(False)
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

    def _save_session_state(self) -> None:
        self._store_pipeline_state()
        self._settings.setValue(
            "session/selectedImage",
            json.dumps(self.browser.current_selection_signature()),
        )
        last_database = str(self._last_loaded_pdb_path) if self._last_loaded_pdb_path else ""
        self._settings.setValue("session/lastDatabase", last_database)
        self._save_recent_files()
        self._settings.sync()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._save_session_state()
        super().closeEvent(event)

    def delete_selected(self) -> None:
        pdb_path = self._current_pdb_path
        if pdb_path is None:
            self.status_bar.showMessage("No Palm database loaded", 4000)
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

        images = load_wqv_pdb(pdb_path)
        self.browser.load_images(images)
        self.delete_action.setEnabled(bool(images))
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
