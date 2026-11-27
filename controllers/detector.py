# controllers/detector.py
from __future__ import annotations

import cv2
import numpy as np
from skimage.morphology import skeletonize

from config.detector_config import DetectorConfig


def _odd(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 else value + 1


def _scaled_kernel(
    image_shape: tuple[int, int],
    cfg: DetectorConfig,
    pixel_per_mm: float | None = None,
) -> int:
    """Scale the black-hat kernel to the image resolution (or physical scale)."""
    if cfg.blackhat_kernel_mm is not None and pixel_per_mm:
        return _odd(int(round(cfg.blackhat_kernel_mm * pixel_per_mm)))
    min_dim = min(image_shape[:2])
    reference = max(1, int(cfg.reference_min_dim))
    scaled = int(round(cfg.blackhat_kernel * min_dim / reference))
    return _odd(scaled)


def _hysteresis(blackhat: np.ndarray, low: float, high: float) -> np.ndarray:
    """Keep weak pixels only if connected to a strong seed."""
    strong = (blackhat >= high).astype(np.uint8)
    weak = (blackhat >= low).astype(np.uint8)
    if not strong.any():
        return np.zeros_like(blackhat, dtype=np.uint8)
    count, labels = cv2.connectedComponents(weak, connectivity=8)
    keep = np.zeros(count, dtype=bool)
    keep[np.unique(labels[strong > 0])] = True
    keep[0] = False
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def _looks_like_crack(
    component: np.ndarray,
    stats_row: np.ndarray,
    min_elongation: float,
    max_fill_ratio: float,
) -> bool:
    """Shape filter: cracks are long, thin, and sparse in their bounding box."""
    area = int(stats_row[cv2.CC_STAT_AREA])
    w = int(stats_row[cv2.CC_STAT_WIDTH])
    h = int(stats_row[cv2.CC_STAT_HEIGHT])
    if area <= 0 or w <= 0 or h <= 0:
        return False
    fill = area / float(w * h)
    if fill > max_fill_ratio:
        return False
    skel = skeletonize(component)
    length = float(skel.sum())
    if length <= 0:
        return False
    elongation = (length * length) / float(area)
    return elongation >= min_elongation


def segment_cracks(
    image_bgr: np.ndarray,
    cfg: DetectorConfig,
    pixel_per_mm: float | None = None,
):
    """Return a binary crack mask and useful intermediate images."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=cfg.clahe_clip_limit,
        tileGridSize=(cfg.clahe_grid_size, cfg.clahe_grid_size),
    )
    enhanced = clahe.apply(gray)

    k = _scaled_kernel(enhanced.shape, cfg, pixel_per_mm)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)

    # Large-scale background subtraction removes low-frequency shading.
    if cfg.flatten_background:
        sigma = max(1.0, k / 2.0)
        background = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=sigma)
        flattened = cv2.subtract(background, enhanced)
        response = cv2.max(blackhat, flattened)
    else:
        response = blackhat

    # Hysteresis thresholding from response percentiles (robust to lighting).
    p_low = float(np.percentile(response, cfg.low_percentile))
    p_high = float(np.percentile(response, cfg.high_percentile))
    if p_high <= p_low:
        p_high = p_low + 1.0
    mask = _hysteresis(response, p_low, p_high)

    # Adaptive threshold in parallel: catches cracks under uneven lighting.
    if cfg.combine_adaptive_threshold:
        block = _odd(max(cfg.adaptive_block_size, 2 * k + 1))
        adaptive = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block, cfg.adaptive_c,
        )
        mask = cv2.bitwise_and(mask, adaptive)

    # Per-component shape filtering (before morphology, so closing does not
    # merge blobs into crack-shaped structures).
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    filtered = np.zeros_like(mask)
    max_area = int(mask.shape[0] * mask.shape[1] *
                   cfg.max_component_area_fraction)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < cfg.min_component_area_px or area > max_area:
            continue
        component = labels == label
        if not _looks_like_crack(
            component, stats[label],
            cfg.min_elongation, cfg.max_fill_ratio,
        ):
            continue
        filtered[component] = 255

    # Conditional closing: reconnect breaks in surviving components only.
    close_k = _odd(max(3, k // 4))
    if cfg.close_iterations > 0 and filtered.any():
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_k, close_k)
        )
        filtered = cv2.morphologyEx(
            filtered, cv2.MORPH_CLOSE, close_kernel,
            iterations=cfg.close_iterations,
        )
        # Drop anything closing may have merged into a blob.
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (filtered > 0).astype(np.uint8), connectivity=8
        )
        cleaned = np.zeros_like(filtered)
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < cfg.min_component_area_px or area > max_area:
                continue
            component = labels == label
            if not _looks_like_crack(
                component, stats[label],
                cfg.min_elongation, cfg.max_fill_ratio,
            ):
                continue
            cleaned[component] = 255
        filtered = cleaned

    return filtered, {
        "gray": gray,
        "enhanced": enhanced,
        "blackhat": blackhat,
        "response": response,
        "low_threshold": p_low,
        "high_threshold": p_high,
        "blackhat_kernel": k,
    }


def measure_components(mask: np.ndarray, pixel_per_mm: float | None = None):
    """Measure each connected crack region in image pixel coordinates."""
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