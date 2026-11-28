"""Builds RGB overlay images for the different display modes (pure numpy/OpenCV)."""
from __future__ import annotations

import cv2
import numpy as np

from .crack_detector import DetectionResult

MODES = ["Original", "Crack mask", "Edges + skeleton", "Width map"]


def render(bgr: np.ndarray, result: DetectionResult | None, mode: str) -> np.ndarray:
    """Return an RGB uint8 image for the requested mode."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if result is None or mode == "Original":
        return rgb
    out = rgb.copy()
    if mode == "Crack mask":
        out[result.mask] = (255, 215, 0)
    elif mode == "Edges + skeleton":
        out[result.mask] = (out[result.mask] * 0.5 + np.array([255, 215, 0]) * 0.5).astype(np.uint8)
        out[result.edges] = (0, 120, 255)
        out[result.skeleton] = (255, 0, 0)
    elif mode == "Width map":
        out = _width_map(out, result)
    return out


def _width_map(rgb: np.ndarray, result: DetectionResult) -> np.ndarray:
    if len(result.points_xy) == 0:
        return rgb
    w = result.widths_mm
    norm = np.clip((w - w.min()) / max(w.max() - w.min(), 1e-6) * 255, 0, 255).astype(np.uint8)
    colours = cv2.applyColorMap(norm.reshape(-1, 1), cv2.COLORMAP_JET).reshape(-1, 3)[:, ::-1]
    layer = np.zeros_like(rgb)
    layer[result.points_xy[:, 1], result.points_xy[:, 0]] = colours
    drawn = layer.any(axis=2).astype(np.uint8)
    layer = cv2.dilate(layer, np.ones((3, 3), np.uint8))  # thicken for visibility
    drawn = cv2.dilate(drawn, np.ones((3, 3), np.uint8)).astype(bool)
    rgb[drawn] = layer[drawn]
    return rgb
