"""High-level pipeline: grey -> MAD segmentation -> edges -> skeleton -> widths."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .crack_analysis import (compute_skeleton, laplacian_edges, measure_widths,
                             pixel_size_mm)
from .mad_segmentation import remove_small_blobs, segment_cracks, to_greyscale


@dataclass
class DetectionParams:
    k: float = 2.5                 # MAD scale factor (paper: 2.0 - 3.5)
    tile_size: int = 224           # sub-image size from the paper
    remove_blobs: bool = True      # optional noise filter (extension)
    min_length_px: int = 20
    use_manual_pixel_size: bool = False
    manual_pixel_size_mm: float = 0.1
    distance_m: float = 1.0        # camera-to-surface distance u
    angle_of_view_deg: float = 40.4  # horizontal lens angle of view alpha


@dataclass
class DetectionResult:
    mask: np.ndarray
    edges: np.ndarray
    skeleton: np.ndarray
    points_xy: np.ndarray
    widths_px: np.ndarray
    pixel_size_mm: float
    params: DetectionParams = field(default_factory=DetectionParams)

    @property
    def widths_mm(self) -> np.ndarray:
        return self.widths_px * self.pixel_size_mm

    def summary(self) -> dict:
        w = self.widths_mm
        has = len(w) > 0
        return {
            "pixel_size_mm": self.pixel_size_mm,
            "crack_pixels": int(self.mask.sum()),
            "crack_area_mm2": float(self.mask.sum() * self.pixel_size_mm ** 2),
            "approx_length_mm": float(len(w) * self.pixel_size_mm),
            "mean_width_mm": float(w.mean()) if has else 0.0,
            "median_width_mm": float(np.median(w)) if has else 0.0,
            "max_width_mm": float(w.max()) if has else 0.0,
            "min_width_mm": float(w.min()) if has else 0.0,
        }


class CrackDetector:
    def detect(self, bgr: np.ndarray, params: DetectionParams) -> DetectionResult:
        grey = to_greyscale(bgr)
        mask = segment_cracks(grey, params.k, params.tile_size)
        if params.remove_blobs:
            mask = remove_small_blobs(mask, params.min_length_px)
        edges = laplacian_edges(mask)
        skeleton = compute_skeleton(mask)
        points, widths_px = measure_widths(mask, skeleton)

        if params.use_manual_pixel_size:
            pw = params.manual_pixel_size_mm
        else:
            pw = pixel_size_mm(params.distance_m * 1000.0, params.angle_of_view_deg,
                               bgr.shape[1])
        return DetectionResult(mask, edges, skeleton, points, widths_px, pw, params)
