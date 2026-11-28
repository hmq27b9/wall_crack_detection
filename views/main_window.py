"""Main window (PyQt6). Contains no detection logic: it only emits signals."""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                             QFormLayout, QGroupBox, QLabel, QMainWindow,
                             QMessageBox, QPlainTextEdit, QProgressBar,
                             QPushButton, QSpinBox, QSplitter, QVBoxLayout,
                             QWidget)

from models.visualization import MODES
from .image_viewer import ImageViewer


class MainWindow(QMainWindow):
    open_requested = pyqtSignal(str)
    detect_requested = pyqtSignal()
    mode_changed = pyqtSignal(str)
    export_overlay_requested = pyqtSignal(str)
    export_csv_requested = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Wall Crack Detector - MAD adaptive threshold")
        self.resize(1300, 800)

        self.viewer = ImageViewer()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_side_panel())
        splitter.addWidget(self.viewer)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)
        self.statusBar().showMessage("Open an image to begin.")
        self.set_actions_enabled(False, False)

    # ------------------------------------------------------------ building
    def _build_side_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMaximumWidth(360)
        panel.setMinimumWidth(300)
        lay = QVBoxLayout(panel)

        self.btn_open = QPushButton("Open image…")
        self.btn_open.clicked.connect(self._on_open)
        self.lbl_info = QLabel("No image loaded")
        self.lbl_info.setWordWrap(True)
        lay.addWidget(self.btn_open)
        lay.addWidget(self.lbl_info)

        # --- MAD parameters
        box = QGroupBox("MAD segmentation")
        form = QFormLayout(box)
        self.sp_k = QDoubleSpinBox()
        self.sp_k.setRange(1.0, 6.0)
        self.sp_k.setSingleStep(0.1)
        self.sp_k.setValue(2.5)
        self.sp_k.setToolTip("T = median - k·MAD. Low k: more sensitive (k≤2.5 detects 0.1 mm),\n"
                             "high k: less noise and better width accuracy.")
        self.sp_tile = QSpinBox()
        self.sp_tile.setRange(32, 1024)
        self.sp_tile.setValue(224)
        self.chk_blobs = QCheckBox("Remove short blobs (noise)")
        self.chk_blobs.setChecked(True)
        self.sp_minlen = QSpinBox()
        self.sp_minlen.setRange(2, 500)
        self.sp_minlen.setValue(20)
        self.sp_minlen.setSuffix(" px")
        form.addRow("k (scale factor)", self.sp_k)
        form.addRow("Sub-image size", self.sp_tile)
        form.addRow(self.chk_blobs)
        form.addRow("Min. length", self.sp_minlen)
        lay.addWidget(box)

        # --- camera / scale
        cam = QGroupBox("Scale (pinhole camera model)")
        cform = QFormLayout(cam)
        self.sp_dist = QDoubleSpinBox()
        self.sp_dist.setRange(0.05, 50.0)
        self.sp_dist.setDecimals(3)
        self.sp_dist.setValue(1.0)
        self.sp_dist.setSuffix(" m")
        self.sp_aov = QDoubleSpinBox()
        self.sp_aov.setRange(1.0, 180.0)
        self.sp_aov.setValue(40.4)
        self.sp_aov.setSuffix(" °")
        self.sp_aov.setToolTip("Horizontal angle of view of the lens.")
        self.chk_manual = QCheckBox("Enter mm/pixel manually")
        self.sp_mm = QDoubleSpinBox()
        self.sp_mm.setRange(0.001, 50.0)
        self.sp_mm.setDecimals(4)
        self.sp_mm.setValue(0.1)
        self.sp_mm.setSuffix(" mm/px")
        self.chk_manual.toggled.connect(self._toggle_manual)
        cform.addRow("Distance u", self.sp_dist)
        cform.addRow("Angle of view α", self.sp_aov)
        cform.addRow(self.chk_manual)
        cform.addRow(self.sp_mm)
        lay.addWidget(cam)
        self._toggle_manual(False)

        self.btn_detect = QPushButton("Detect cracks")
        self.btn_detect.clicked.connect(self.detect_requested.emit)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        lay.addWidget(self.btn_detect)
        lay.addWidget(self.progress)

        # --- display
        lay.addWidget(QLabel("Display"))
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(MODES)
        self.cmb_mode.currentTextChanged.connect(self.mode_changed.emit)
        self.btn_fit = QPushButton("Fit to window")
        self.btn_fit.clicked.connect(self.viewer.fit)
        lay.addWidget(self.cmb_mode)
        lay.addWidget(self.btn_fit)

        # --- results
        self.txt_results = QPlainTextEdit()
        self.txt_results.setReadOnly(True)
        self.txt_results.setPlaceholderText("Results will appear here.")
        lay.addWidget(self.txt_results, 1)

        self.btn_png = QPushButton("Export overlay image…")
        self.btn_png.clicked.connect(self._on_export_png)
        self.btn_csv = QPushButton("Export widths (CSV)…")
        self.btn_csv.clicked.connect(self._on_export_csv)
        lay.addWidget(self.btn_png)
        lay.addWidget(self.btn_csv)
        return panel

    # ------------------------------------------------------------- slots
    def _toggle_manual(self, manual: bool) -> None:
        self.sp_mm.setEnabled(manual)
        self.sp_dist.setEnabled(not manual)
        self.sp_aov.setEnabled(not manual)

    def _on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if path:
            self.open_requested.emit(path)

    def _on_export_png(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save overlay", "overlay.png", "PNG (*.png);;JPEG (*.jpg)")
        if path:
            self.export_overlay_requested.emit(path)

    def _on_export_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save widths", "widths.csv", "CSV (*.csv)")
        if path:
            self.export_csv_requested.emit(path)

    # ------------------------------------------------------- API for controller
    def get_params(self) -> dict:
        return dict(
            k=self.sp_k.value(), tile_size=self.sp_tile.value(),
            remove_blobs=self.chk_blobs.isChecked(), min_length_px=self.sp_minlen.value(),
            use_manual_pixel_size=self.chk_manual.isChecked(),
            manual_pixel_size_mm=self.sp_mm.value(),
            distance_m=self.sp_dist.value(), angle_of_view_deg=self.sp_aov.value())

    def current_mode(self) -> str:
        return self.cmb_mode.currentText()

    def set_mode(self, mode: str) -> None:
        self.cmb_mode.blockSignals(True)
        self.cmb_mode.setCurrentText(mode)
        self.cmb_mode.blockSignals(False)

    def show_image(self, rgb, fit: bool = False) -> None:
        self.viewer.set_rgb(rgb, fit=fit)

    def set_image_info(self, text: str) -> None:
        self.lbl_info.setText(text)
        self.statusBar().showMessage("Image loaded.")

    def set_results_text(self, text: str) -> None:
        self.txt_results.setPlainText(text)

    def set_busy(self, busy: bool) -> None:
        self.progress.setVisible(busy)
        self.btn_detect.setEnabled(not busy)
        self.statusBar().showMessage("Detecting cracks…" if busy else "Done.")

    def set_actions_enabled(self, has_image: bool, has_result: bool) -> None:
        self.btn_detect.setEnabled(has_image)
        self.cmb_mode.setEnabled(has_image)
        self.btn_fit.setEnabled(has_image)
        self.btn_png.setEnabled(has_image)
        self.btn_csv.setEnabled(has_result)

    def show_error(self, msg: str) -> None:
        QMessageBox.critical(self, "Error", msg)
