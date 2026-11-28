"""Crack segmentation with a per-sub-image Median Absolute Deviation threshold.

Paper, Sec. 2.2-2.3:
  * image is split into 224x224 sub-images
  * greyscale:  Y = 0.299 R + 0.587 G + 0.114 B            (Eq. 1)
  * MAD = median(|Xi - median(X)|)                          (Eq. 2)
  * T   = median(X) - k * MAD                               (Eq. 3)
  * pixels darker than T are outliers -> crack pixels       (Eq. 4)
"""
from __future__ import annotations

import cv2
import numpy as np

NTSC_WEIGHTS = np.array([0.114, 0.587, 0.299], dtype=np.float32)  # B, G, R order (OpenCV)


def to_greyscale(bgr: np.ndarray) -> np.ndarray:
    """Eq. (1) with NTSC coefficients. Input is an OpenCV BGR image."""
    if bgr.ndim == 2:
        return bgr
    return np.clip(bgr.astype(np.float32) @ NTSC_WEIGHTS, 0, 255).astype(np.uint8)


def mad_threshold(tile: np.ndarray, k: float) -> tuple[float, float, float]:
    """Return (median, MAD, T) for one sub-image."""
    values = tile.astype(np.float32).ravel()
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median, mad, median - k * mad


def segment_cracks(grey: np.ndarray, k: float = 2.5, tile_size: int = 224) -> np.ndarray:
    """Boolean crack mask obtained with one adaptive MAD threshold per tile."""
    h, w = grey.shape
    mask = np.zeros((h, w), dtype=bool)
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = grey[y:y + tile_size, x:x + tile_size]
            _, mad, thr = mad_threshold(tile, k)
            if mad == 0:  # perfectly flat tile: no intensity discontinuity
                continue
            mask[y:y + tile_size, x:x + tile_size] = tile < thr
    return mask


def remove_small_blobs(mask: np.ndarray, min_length_px: int) -> np.ndarray:
    """Optional post-processing (the paper lists this as future work):
    drop dot-like / rounded components whose longest bounding-box side is
    shorter than ``min_length_px``."""
    if min_length_px <= 1:
        return mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    longest = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    keep = longest >= min_length_px
    keep[0] = False  # background label
    return keep[labels]
