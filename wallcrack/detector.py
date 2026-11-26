# wallcrack/detector.py
from __future__ import annotations

import cv2
import numpy as np
from skimage.morphology import skeletonize
from .config import DetectorConfig


def _odd(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 else value + 1


def segment_cracks(image_bgr: np.ndarray, cfg: DetectorConfig):
    """Return a binary crack mask and useful intermediate images.

    This is a classical image-processing baseline, not a trained AI model.
    It works best for dark cracks on relatively light concrete surfaces.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=cfg.clahe_clip_limit,
        tileGridSize=(cfg.clahe_grid_size, cfg.clahe_grid_size),
    )
    enhanced = clahe.apply(gray)

    k = _odd(cfg.blackhat_kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)

    # Otsu thresholding adapts to the contrast in each image.
    _, blackhat_mask = cv2.threshold(
        blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    if cfg.combine_adaptive_threshold:
        block = _odd(cfg.adaptive_block_size)
        adaptive = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block, cfg.adaptive_c
        )
        mask = cv2.bitwise_and(blackhat_mask, adaptive)
    else:
        mask = blackhat_mask

    close_k = max(1, int(cfg.close_kernel))
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_k, close_k)
    )
    if cfg.close_iterations > 0:
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, close_kernel,
            iterations=cfg.close_iterations
        )

    # Remove tiny specks and implausibly large regions.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    cleaned = np.zeros_like(mask)
    max_area = int(mask.shape[0] * mask.shape[1] *
                   cfg.max_component_area_fraction)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if cfg.min_component_area_px <= area <= max_area:
            cleaned[labels == label] = 255

    return cleaned, {
        "gray": gray,
        "enhanced": enhanced,
        "blackhat": blackhat,
    }


def measure_components(mask: np.ndarray, pixel_per_mm: float | None = None):
    """Measure each connected crack region in image pixel coordinates.

    Length is approximated by skeleton pixel count. Width is estimated from
    the distance transform sampled along the skeleton. These are image-plane
    estimates; perspective and camera angle can distort physical dimensions.
    """
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    height, width = mask.shape
    results = []

    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area_px = int(stats[label, cv2.CC_STAT_AREA])
        component = labels == label

        skeleton = skeletonize(component)
        skeleton_count = int(skeleton.sum())
        length_px = float(skeleton_count)
        widths_px = 2.0 * distance[skeleton]
        mean_width_px = float(np.mean(widths_px)) if widths_px.size else 0.0
        max_width_px = float(np.max(widths_px)) if widths_px.size else 0.0

        # PCA orientation of all foreground pixels, in degrees from +x.
        ys, xs = np.where(component)
        if len(xs) >= 2:
            coords = np.column_stack((xs, ys)).astype(np.float64)
            cov = np.cov(coords, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov)
            major = eigvecs[:, int(np.argmax(eigvals))]
            angle = float(np.degrees(np.arctan2(major[1], major[0])))
        else:
            angle = 0.0

        record = {
            "coordinates_px": {
                "centroid": [float(centroids[label][0]),
                             float(centroids[label][1])],
                "bounding_box": {"x": x, "y": y, "width": w, "height": h},
            },
            "length_px": round(length_px, 2),
            "mean_width_px": round(mean_width_px, 2),
            "max_width_px": round(max_width_px, 2),
            "area_px": area_px,
            "orientation_degrees": round(angle, 2),
            "depth_mm": None,
            "depth_note": "Not measurable from a single ordinary RGB image",
        }

        if pixel_per_mm is not None and pixel_per_mm > 0:
            record["coordinates_mm"] = {
                "centroid": [
                    round(float(centroids[label][0]) / pixel_per_mm, 3),
                    round(float(centroids[label][1]) / pixel_per_mm, 3),
                ],
                "bounding_box": {
                    "x": round(x / pixel_per_mm, 3),
                    "y": round(y / pixel_per_mm, 3),
                    "width": round(w / pixel_per_mm, 3),
                    "height": round(h / pixel_per_mm, 3),
                },
            }
            record["length_mm"] = round(length_px / pixel_per_mm, 3)
            record["mean_width_mm"] = round(mean_width_px / pixel_per_mm, 3)
            record["max_width_mm"] = round(max_width_px / pixel_per_mm, 3)
            record["area_mm2"] = round(area_px / (pixel_per_mm ** 2), 3)

        results.append(record)

    return results


def annotate_image(image_bgr: np.ndarray, cracks: list[dict]) -> np.ndarray:
    output = image_bgr.copy()
    for index, crack in enumerate(cracks, start=1):
        box = crack["coordinates_px"]["bounding_box"]
        x, y, w, h = box["x"], box["y"], box["width"], box["height"]
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(
            output, f"Crack {index}", (x, max(18, y - 7)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2,
            cv2.LINE_AA
        )
    return output
