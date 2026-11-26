# controllers/app_controller.py
"""Glue layer between the Tkinter view and the domain modules."""
from __future__ import annotations

import threading
from pathlib import Path

from config import DetectorConfig
from .pipeline import process_image


class AppController:
    """Runs detection on a background thread and reports back via callbacks."""

    def __init__(self, output_dir: Path, detector_config: DetectorConfig | None = None):
        self.output_dir = Path(output_dir)
        self.detector_config = detector_config or DetectorConfig()
        self._thread: threading.Thread | None = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start_detection(
        self,
        image_path: str | Path,
        wall_id: str,
        pixel_per_mm: float | None,
        on_success,
        on_error,
        on_status,
    ) -> None:
        if self.is_running():
            on_error("A detection is already running.")
            return

        self._thread = threading.Thread(
            target=self._worker,
            args=(Path(image_path), wall_id, pixel_per_mm,
                  on_success, on_error, on_status),
            daemon=True,
        )
        self._thread.start()

    def _worker(self, image_path, wall_id, pixel_per_mm,
                on_success, on_error, on_status):
        try:
            on_status("Connecting to MongoDB...")
            on_status("Detecting cracks and saving...")
            result = process_image(
                image_path=image_path,
                output_dir=self.output_dir,
                wall_id=wall_id,
                pixel_per_mm=pixel_per_mm,
                config=self.detector_config,
            )
            on_success(result)
        except Exception as exc:  # noqa: BLE001
            on_error(str(exc))