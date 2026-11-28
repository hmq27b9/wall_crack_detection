"""Application state: the loaded image, the latest result and file I/O."""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from .crack_detector import DetectionResult
from .visualization import render


class ImageModel:
    def __init__(self) -> None:
        self.path: str | None = None
        self.image_bgr: np.ndarray | None = None
        self.result: DetectionResult | None = None

    @property
    def has_image(self) -> bool:
        return self.image_bgr is not None

    def load(self, path: str) -> None:
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image: {path}")
        self.path, self.image_bgr, self.result = path, img, None

    def render(self, mode: str) -> np.ndarray:
        return render(self.image_bgr, self.result, mode)

    def save_overlay(self, path: str, mode: str) -> None:
        rgb = self.render(mode)
        ok, buf = cv2.imencode(Path(path).suffix or ".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise ValueError("Could not encode image")
        buf.tofile(path)

    def export_widths_csv(self, path: str) -> None:
        if self.result is None:
            raise ValueError("No result to export")
        r = self.result
        with open(path, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["x_px", "y_px", "width_px", "width_mm"])
            for (x, y), wpx, wmm in zip(r.points_xy, r.widths_px, r.widths_mm):
                wr.writerow([int(x), int(y), f"{wpx:.2f}", f"{wmm:.4f}"])
