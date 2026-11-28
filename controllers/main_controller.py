"""Connects the view to the models; detection runs in a worker thread."""
from __future__ import annotations

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from models.crack_detector import CrackDetector, DetectionParams
from models.image_model import ImageModel
from views.main_window import MainWindow


class DetectionWorker(QThread):
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, image, params: DetectionParams) -> None:
        super().__init__()
        self._image, self._params = image, params

    def run(self) -> None:
        try:
            self.succeeded.emit(CrackDetector().detect(self._image, self._params))
        except Exception as exc:  # report to UI instead of killing the thread
            self.failed.emit(str(exc))


class MainController(QObject):
    def __init__(self, model: ImageModel, view: MainWindow) -> None:
        super().__init__()
        self.model, self.view = model, view
        self._worker: DetectionWorker | None = None

        view.open_requested.connect(self.open_image)
        view.detect_requested.connect(self.run_detection)
        view.mode_changed.connect(self.refresh_view)
        view.export_overlay_requested.connect(self.export_overlay)
        view.export_csv_requested.connect(self.export_csv)

    # ------------------------------------------------------------- actions
    def open_image(self, path: str) -> None:
        try:
            self.model.load(path)
        except Exception as exc:
            self.view.show_error(str(exc))
            return
        h, w = self.model.image_bgr.shape[:2]
        self.view.set_image_info(f"{path}\n{w} x {h} px")
        self.view.set_results_text("")
        self.view.set_mode("Original")
        self.view.show_image(self.model.render("Original"), fit=True)
        self.view.set_actions_enabled(has_image=True, has_result=False)

    def run_detection(self) -> None:
        if not self.model.has_image or self._worker is not None:
            return
        params = DetectionParams(**self.view.get_params())
        self.view.set_busy(True)
        self._worker = DetectionWorker(self.model.image_bgr, params)
        self._worker.succeeded.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._cleanup)
        self._worker.start()

    def _on_done(self, result) -> None:
        self.model.result = result
        self.view.set_results_text(self._format(result))
        self.view.set_mode("Edges + skeleton")
        self.refresh_view("Edges + skeleton")
        self.view.set_actions_enabled(has_image=True, has_result=True)

    def _on_failed(self, msg: str) -> None:
        self.view.show_error(f"Detection failed:\n{msg}")

    def _cleanup(self) -> None:
        self._worker = None
        self.view.set_busy(False)

    def refresh_view(self, mode: str) -> None:
        if self.model.has_image:
            self.view.show_image(self.model.render(mode))

    def export_overlay(self, path: str) -> None:
        self._safe(lambda: self.model.save_overlay(path, self.view.current_mode()))

    def export_csv(self, path: str) -> None:
        self._safe(lambda: self.model.export_widths_csv(path))

    # ------------------------------------------------------------- helpers
    def _safe(self, fn) -> None:
        try:
            fn()
        except Exception as exc:
            self.view.show_error(str(exc))

    @staticmethod
    def _format(result) -> str:
        s = result.summary()
        if s["crack_pixels"] == 0:
            return "No crack pixels detected."
        return (
            f"Pixel size:       {s['pixel_size_mm']:.4f} mm/px\n"
            f"Crack pixels:     {s['crack_pixels']}\n"
            f"Crack area:       {s['crack_area_mm2']:.1f} mm²\n"
            f"Approx. length:   {s['approx_length_mm']:.1f} mm\n"
            f"Mean width:       {s['mean_width_mm']:.2f} mm\n"
            f"Median width:     {s['median_width_mm']:.2f} mm\n"
            f"Max width:        {s['max_width_mm']:.2f} mm\n"
            f"Min width:        {s['min_width_mm']:.2f} mm\n\n"
            "Note: the paper reports that widths tend to be\n"
            "overestimated at low k (a conservative bias)."
        )
