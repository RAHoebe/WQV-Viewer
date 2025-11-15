import os
import shutil
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication, QListView

from wqv_viewer.gui import MainWindow


def test_main_window_constructs(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(Path(tmp_path) / "settings.ini"), QSettings.Format.IniFormat)
    window = MainWindow(settings=settings)
    try:
        assert window.browser.list_widget.count() == 0
        assert window.browser.conventional_scale_combo.itemText(0) == "2×"
        assert window.browser.ai_scale_combo.currentText() == "2×"
        assert window.browser.list_widget.viewMode() == QListView.ViewMode.IconMode
    finally:
        window.deleteLater()
        if QApplication.instance() is app:
            app.quit()


def _copy_pdb_fixture(tmp_path: Path, *, name: str = "WQVLinkDB.PDB") -> Path:
    source = Path(__file__).parent / "data" / "WQVLinkDB.PDB"
    destination = tmp_path / name
    shutil.copyfile(source, destination)
    return destination


def test_upscale_controls_produce_preview(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(Path(tmp_path) / "settings.ini"), QSettings.Format.IniFormat)
    window = MainWindow(settings=settings)
    try:
        pdb_path = _copy_pdb_fixture(Path(tmp_path))
        window._load_pdb(pdb_path)

        assert window.browser.list_widget.count() > 0

        window.browser.conventional_checkbox.setChecked(True)
        window.browser.conventional_combo.setCurrentIndex(0)
        window.browser.conventional_scale_combo.setCurrentIndex(0)
        window.browser.ai_checkbox.setChecked(False)
        window.browser._apply_upscale()

        pixmap = window.browser.upscaled_label.pixmap()
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
    window = MainWindow(settings=settings)
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


def test_session_persistence_restores_state(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    settings_path = Path(tmp_path) / "settings.ini"
    settings = QSettings(str(settings_path), QSettings.Format.IniFormat)

    pdb_path = _copy_pdb_fixture(Path(tmp_path))

    window = MainWindow(settings=settings)
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
        window._save_session_state()
    finally:
        window.deleteLater()

    window_restored = MainWindow(settings=settings)
    try:
        assert window_restored._current_pdb_path == pdb_path.resolve()
        assert window_restored.browser.conventional_checkbox.isChecked() is False
        assert window_restored.browser.ai_checkbox.isChecked() is True
        assert window_restored.browser.device_combo.currentData() == "cpu"
        if window_restored.browser.list_widget.count() > 1:
            assert window_restored.browser.list_widget.currentRow() == 1
    finally:
        window_restored.deleteLater()
        if QApplication.instance() is app:
            app.quit()


def test_recent_files_capped_at_ten(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    settings_path = Path(tmp_path) / "settings.ini"
    settings = QSettings(str(settings_path), QSettings.Format.IniFormat)

    window = MainWindow(settings=settings)
    try:
        for index in range(12):
            pdb_path = _copy_pdb_fixture(Path(tmp_path), name=f"WQVLinkDB_{index}.PDB")
            window._load_pdb(pdb_path)
        window._save_session_state()
    finally:
        window.deleteLater()

    window_reloaded = MainWindow(settings=settings)
    try:
        assert len(window_reloaded._recent_files) == 10
        latest_path = Path(tmp_path) / "WQVLinkDB_11.PDB"
        assert window_reloaded._recent_files[0] == str(latest_path.resolve())
    finally:
        window_reloaded.deleteLater()
        if QApplication.instance() is app:
            app.quit()
