"""Edge detection, skeletonisation and crack-width measurement.

Paper, Sec. 2.4-2.5:
  * Laplacian edge detector, edge pixels are the negative responses   (Eq. 5)
  * centreline via skimage.morphology.skeletonize
  * at each skeleton pixel the perpendicular line is followed in both
    directions until the crack edge is reached; the distance between the two
    edge pixels is the width in pixels
  * pixel size from the pinhole model: pw = 2 * tan(alpha/2) * u / nw (Eq. 6)
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize


def laplacian_edges(mask: np.ndarray) -> np.ndarray:
    """Edge pixels of the crack region = negative Laplacian response inside it."""
    lap = cv2.Laplacian(mask.astype(np.float32), cv2.CV_32F, ksize=1)
    return (lap < 0) & mask


def compute_skeleton(mask: np.ndarray) -> np.ndarray:
    return skeletonize(mask)


def pixel_size_mm(distance_mm: float, angle_of_view_deg: float, sensor_width_px: int) -> float:
    """Eq. (6): physical size of one pixel on the surface."""
    return 2.0 * math.tan(math.radians(angle_of_view_deg) / 2.0) * distance_mm / sensor_width_px


def _local_normals(points_xy: np.ndarray, neighbours: int = 15) -> np.ndarray:
    """Unit normals to the skeleton direction at every skeleton point (PCA of
    the nearest skeleton neighbours)."""
    n = len(points_xy)
    k = min(neighbours, n)
    tree = cKDTree(points_xy)
    _, idx = tree.query(points_xy, k=k)
    idx = idx.reshape(n, -1)
    local = points_xy[idx].astype(np.float64)            # (n, k, 2)
    local -= local.mean(axis=1, keepdims=True)
    cxx = (local[..., 0] ** 2).mean(axis=1)
    cyy = (local[..., 1] ** 2).mean(axis=1)
    cxy = (local[..., 0] * local[..., 1]).mean(axis=1)
    theta = 0.5 * np.arctan2(2 * cxy, cxx - cyy)         # tangent angle
    return np.stack([-np.sin(theta), np.cos(theta)], axis=1)


def measure_widths(mask: np.ndarray, skeleton: np.ndarray,
                   max_half_width_px: int = 80, step: float = 0.5):
    """Return (points_xy [N,2] int, widths_px [N] float)."""
    ys, xs = np.nonzero(skeleton)
    if len(xs) == 0:
        return np.empty((0, 2), int), np.empty(0, float)
    pts = np.stack([xs, ys], axis=1)
    if len(pts) < 3:
        return pts, np.ones(len(pts))

    normals = _local_normals(pts)
    h, w = mask.shape
    dist = {+1: np.full(len(pts), float(max_half_width_px)),
            -1: np.full(len(pts), float(max_half_width_px))}
    done = {+1: np.zeros(len(pts), bool), -1: np.zeros(len(pts), bool)}

    for t in np.arange(step, max_half_width_px + step, step):
        for sign in (+1, -1):
            todo = ~done[sign]
            if not todo.any():
                continue
            p = pts[todo] + sign * t * normals[todo]
            xi = np.rint(p[:, 0]).astype(int)
            yi = np.rint(p[:, 1]).astype(int)
            inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
            hit = np.ones(len(p), bool)               # leaving the image counts as a hit
            hit[inside] = ~mask[yi[inside], xi[inside]]  # first pixel outside the crack
            todo_idx = np.nonzero(todo)[0]
            dist[sign][todo_idx[hit]] = t
            done[sign][todo_idx[hit]] = True
        if done[+1].all() and done[-1].all():
            break

    widths = np.clip(dist[+1] + dist[-1] - 1.0, 1.0, None)
    return pts, widths
