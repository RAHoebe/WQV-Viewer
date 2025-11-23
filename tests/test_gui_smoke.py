import os
import shutil
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QMimeData, QSettings, QUrl
from PyQt6.QtWidgets import QApplication, QFileDialog, QListView, QListWidget, QInputDialog

import pytest

from wqv_viewer.gui import MainWindow


def test_main_window_constructs(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(Path(tmp_path) / "settings.ini"), QSettings.Format.IniFormat)
    window = MainWindow(settings=settings, enable_async_upscale=False)
    try:
        assert window.browser.list_widget.count() == 0
        assert window.browser.conventional_scale_combo.itemText(0) == "2×"
        assert window.browser.ai_scale_combo.currentText() == "2×"
        assert window.browser.list_widget.viewMode() == QListView.ViewMode.IconMode
        assert window.browser.list_widget.selectionMode() == QListWidget.SelectionMode.ExtendedSelection
        assert window.browser.order_combo.currentData() == "ai-first"
        assert all(value == "—" for value in window.browser.metadata_snapshot().values())
    finally:
        window.deleteLater()
        if QApplication.instance() is app:
            app.quit()


def _copy_pdb_fixture(
    tmp_path: Path,
    *,
    name: str = "WQVLinkDB.PDB",
    source_name: str = "WQVLinkDB.PDB",
) -> Path:
    source = Path(__file__).parent / "data" / source_name
    destination = tmp_path / name
    shutil.copyfile(source, destination)
    return destination


def test_upscale_controls_produce_preview(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(Path(tmp_path) / "settings.ini"), QSettings.Format.IniFormat)
    window = MainWindow(settings=settings, enable_async_upscale=False)
    try:
        pdb_path = _copy_pdb_fixture(Path(tmp_path))
        window._load_pdb(pdb_path)

        assert window.browser.list_widget.count() > 0
        metadata = window.browser.metadata_snapshot()
        assert metadata["filename"].upper().endswith(".PDR")
        assert metadata["resolution"] != "—"
        assert metadata["captured"] != "—"

        window.browser.conventional_checkbox.setChecked(True)
        window.browser.conventional_combo.setCurrentIndex(0)
        window.browser.conventional_scale_combo.setCurrentIndex(0)
        window.browser.ai_checkbox.setChecked(False)
        window.browser._apply_upscale()

        pixmap = window.browser.upscaled_view.current_pixmap()
        assert pixmap is not None and not pixmap.isNull()

        status = window.browser.primary_status.text()
        assert status
        assert "×" in status
    finally:
        window.deleteLater()
        if QApplication.instance() is app:
            app.quit()


def test_export_upscaled_image(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(Path(tmp_path) / "settings.ini"), QSettings.Format.IniFormat)
    window = MainWindow(settings=settings, enable_async_upscale=False)
    try:
        pdb_path = _copy_pdb_fixture(Path(tmp_path))
        window._load_pdb(pdb_path)

        target = tmp_path / "upscaled.png"
        assert window._export_upscaled_image(target)
        assert target.exists()
    finally:
        window.deleteLater()
        if QApplication.instance() is app:
            app.quit()


def test_multi_selection_uses_first_item(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(Path(tmp_path) / "settings.ini"), QSettings.Format.IniFormat)
    window = MainWindow(settings=settings, enable_async_upscale=False)
    try:
        pdb_path = _copy_pdb_fixture(Path(tmp_path))
        window._load_pdb(pdb_path)

        list_widget = window.browser.list_widget
        if list_widget.count() < 2:
            pytest.skip("Fixture does not provide multiple images")

        list_widget.clearSelection()
        list_widget.item(0).setSelected(True)
        list_widget.item(1).setSelected(True)
        app.processEvents()

        selected = window.browser.selected_images()
        assert selected
        assert window.browser.current_image() == selected[0]
        assert list_widget.currentRow() == 0
    finally:
        window.deleteLater()
        if QApplication.instance() is app:
            app.quit()


def test_delete_disabled_for_color_pdb(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(Path(tmp_path) / "settings.ini"), QSettings.Format.IniFormat)
    window = MainWindow(settings=settings, enable_async_upscale=False)
    try:
        pdb_path = _copy_pdb_fixture(
            Path(tmp_path),
            name="WQVColorDB.PDB",
            source_name="WQVColorDB.PDB",
        )
        window._load_pdb(pdb_path)
        assert window.delete_action.isEnabled() is False
        window.delete_selected()
        message = window.status_bar.currentMessage()
        assert message and "color" in message.lower()
    finally:
        window.deleteLater()
        if QApplication.instance() is app:
            app.quit()


def test_color_pdb_missing_cas_emits_warning(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(Path(tmp_path) / "settings.ini"), QSettings.Format.IniFormat)
    window = MainWindow(settings=settings, enable_async_upscale=False)
    try:
        pdb_path = _copy_pdb_fixture(
            Path(tmp_path),
            name="WQVColorDB.PDB",
            source_name="WQVColorDB.PDB",
        )
        window._load_pdb(pdb_path)
        assert window.diagnostics_action.isEnabled() is True
        assert window._last_load_report is not None
        assert window._last_load_report.warnings
        status_message = window.status_bar.currentMessage()
        assert status_message and "placeholder" in status_message.lower()
    finally:
        window.deleteLater()
        if QApplication.instance() is app:
            app.quit()


def test_export_selected_directory(tmp_path, monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(Path(tmp_path) / "settings.ini"), QSettings.Format.IniFormat)
    window = MainWindow(settings=settings, enable_async_upscale=False)
    try:
        pdb_path = _copy_pdb_fixture(Path(tmp_path))
        window._load_pdb(pdb_path)

        list_widget = window.browser.list_widget
        if list_widget.count() < 2:
            pytest.skip("Fixture does not provide enough images")

        list_widget.clearSelection()
        list_widget.item(0).setSelected(True)
        list_widget.item(1).setSelected(True)
        app.processEvents()

        export_dir = Path(tmp_path) / "exports"
        export_dir.mkdir()
        monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_, **__: str(export_dir))

        window.export_selected()

        png_exports = sorted(export_dir.glob("*.png"))
        json_exports = sorted(export_dir.glob("*.json"))
        assert len(png_exports) == 2
        assert len(json_exports) == 2
        assert window._settings.value("session/lastExportDirectory") == str(export_dir)
    finally:
        window.deleteLater()
        if QApplication.instance() is app:
            app.quit()


def test_pipeline_preset_roundtrip(tmp_path, monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(Path(tmp_path) / "settings.ini"), QSettings.Format.IniFormat)
    window = MainWindow(settings=settings, enable_async_upscale=False)
    try:
        pdb_path = _copy_pdb_fixture(Path(tmp_path))
        window._load_pdb(pdb_path)

        window.browser.conventional_checkbox.setChecked(False)
        window.browser.ai_checkbox.setChecked(True)

        monkeypatch.setattr(QInputDialog, "getText", lambda *_, **__: ("PresetA", True))
        window._save_pipeline_preset()
        assert window._pipeline_presets and window._pipeline_presets[0]["name"] == "PresetA"

        window.browser.conventional_checkbox.setChecked(True)
        window.browser.ai_checkbox.setChecked(False)

        monkeypatch.setattr(QInputDialog, "getItem", lambda *_, **__: ("PresetA", True))
        window._load_pipeline_preset_dialog()
        assert window.browser.conventional_checkbox.isChecked() is False
        assert window.browser.ai_checkbox.isChecked() is True
    finally:
        window.deleteLater()
        if QApplication.instance() is app:
            app.quit()


def test_pdb_path_from_mime_filters_non_pdb(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(Path(tmp_path) / "settings.ini"), QSettings.Format.IniFormat)
    window = MainWindow(settings=settings, enable_async_upscale=False)
    try:
        valid_path = tmp_path / "database.pdb"
        valid_path.write_bytes(b"")
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(valid_path))])
        pdbs, images = window._paths_from_mime(mime)
        assert pdbs == [valid_path.resolve()]
        assert images == []

        invalid = tmp_path / "image.pdr"
        invalid.write_bytes(b"")
        mime_invalid = QMimeData()
        mime_invalid.setUrls([QUrl.fromLocalFile(str(invalid))])
        pdbs, images = window._paths_from_mime(mime_invalid)
        assert pdbs == []
        assert images == []

        remote_mime = QMimeData()
        remote_mime.setUrls([QUrl("https://example.com/file.pdb")])
        pdbs, images = window._paths_from_mime(remote_mime)
        assert pdbs == []
        assert images == []
    finally:
        window.deleteLater()
        if QApplication.instance() is app:
            app.quit()


def test_session_persistence_restores_state(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    settings_path = Path(tmp_path) / "settings.ini"
    settings = QSettings(str(settings_path), QSettings.Format.IniFormat)

    pdb_path = _copy_pdb_fixture(Path(tmp_path))

    window = MainWindow(settings=settings, enable_async_upscale=False)
    try:
        window._load_pdb(pdb_path)
        if window.browser.list_widget.count() > 1:
            window.browser.list_widget.setCurrentRow(1)
        window.browser.conventional_checkbox.setChecked(False)
        window.browser.ai_checkbox.setChecked(True)
        if window.browser.ai_combo.count() > 0:
            window.browser.ai_combo.setCurrentIndex(window.browser.ai_combo.count() - 1)
        if window.browser.ai_scale_combo.count() > 0:
            window.browser.ai_scale_combo.setCurrentIndex(window.browser.ai_scale_combo.count() - 1)
        window.browser.device_combo.setCurrentIndex(2)
        window.browser.set_zoom_mode("actual")
        window.browser.splitter.setSizes([150, 950])
        window.resize(1024, 768)
        window._save_session_state()
    finally:
        window.deleteLater()

    window_restored = MainWindow(settings=settings, enable_async_upscale=False)
    try:
        assert window_restored._current_pdb_path == pdb_path.resolve()
        assert window_restored.browser.conventional_checkbox.isChecked() is False
        assert window_restored.browser.ai_checkbox.isChecked() is True
        assert window_restored.browser.device_combo.currentData() == "cpu"
        if window_restored.browser.list_widget.count() > 1:
            assert window_restored.browser.list_widget.currentRow() == 1
        assert window_restored.browser.current_zoom_mode() == "actual"
    finally:
        window_restored.deleteLater()
        if QApplication.instance() is app:
            app.quit()


def test_recent_files_capped_at_ten(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    settings_path = Path(tmp_path) / "settings.ini"
    settings = QSettings(str(settings_path), QSettings.Format.IniFormat)

    window = MainWindow(settings=settings, enable_async_upscale=False)
    try:
        for index in range(12):
            pdb_path = _copy_pdb_fixture(
                Path(tmp_path),
                name=f"WQVLinkDB_{index}.PDB",
                source_name="WQVLinkDB.PDB",
            )
            window._load_pdb(pdb_path)
        window._save_session_state()
    finally:
        window.deleteLater()

    window_reloaded = MainWindow(settings=settings, enable_async_upscale=False)
    try:
        assert len(window_reloaded._recent_files) == 10
        latest_path = Path(tmp_path) / "WQVLinkDB_11.PDB"
        assert window_reloaded._recent_files[0] == str(latest_path.resolve())
    finally:
        window_reloaded.deleteLater()
        if QApplication.instance() is app:
            app.quit()


def test_open_standalone_jpeg_shows_preview(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    settings_path = Path(tmp_path) / "settings.ini"
    settings = QSettings(str(settings_path), QSettings.Format.IniFormat)

    window = MainWindow(settings=settings, enable_async_upscale=False)
    try:
        image_path = Path(__file__).parent / "data" / "test.jpg"
        assert image_path.exists()
        assert window._ingest_image_files([image_path], allow_popups=False)
        app.processEvents()

        pixmap = window.browser.original_view.current_pixmap()
        assert pixmap is not None and not pixmap.isNull()

        # Re-select the same row to mimic the user clicking the entry again.
        window.browser.list_widget.setCurrentRow(0)
        app.processEvents()
        pixmap = window.browser.original_view.current_pixmap()
        assert pixmap is not None and not pixmap.isNull()
    finally:
        window.deleteLater()
        if QApplication.instance() is app:
            app.quit()
