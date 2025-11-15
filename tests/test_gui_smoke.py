import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QListView

from wqv_viewer.gui import MainWindow


def test_main_window_constructs() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        assert window.browser.list_widget.count() == 0
        assert window.browser.conventional_scale_combo.itemText(0) == "2×"
        assert window.browser.ai_scale_combo.currentText() == "2×"
        assert window.browser.list_widget.viewMode() == QListView.ViewMode.IconMode
    finally:
        window.deleteLater()
        if QApplication.instance() is app:
            app.quit()


def test_upscale_controls_produce_preview() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        pdb_path = Path(__file__).parent / "data" / "WQVLinkDB.PDB"
        window._load_pdb(pdb_path)

        assert window.browser.list_widget.count() > 0

        window.browser.conventional_checkbox.setChecked(True)
        window.browser.conventional_combo.setCurrentIndex(0)
        window.browser.conventional_scale_combo.setCurrentIndex(0)
        window.browser.ai_checkbox.setChecked(False)
        window.browser._apply_upscale()

        pixmap = window.browser.upscaled_label.pixmap()
        assert pixmap is not None and not pixmap.isNull()

        status = window.browser.upscale_status.text()
        assert "Conventional" in status
    finally:
        window.deleteLater()
        if QApplication.instance() is app:
            app.quit()


def test_export_upscaled_image(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        pdb_path = Path(__file__).parent / "data" / "WQVLinkDB.PDB"
        window._load_pdb(pdb_path)

        target = tmp_path / "upscaled.png"
        assert window._export_upscaled_image(target)
        assert target.exists()
    finally:
        window.deleteLater()
        if QApplication.instance() is app:
            app.quit()
