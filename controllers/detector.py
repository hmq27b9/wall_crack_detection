from __future__ import annotations

import cv2
import numpy as np
from skimage.morphology import skeletonize

from config.detector_config import DetectorConfig


def _looks_like_crack(
    component: np.ndarray,
    stats_row: np.ndarray,
    min_elongation: float,
    max_fill_ratio: float,
) -> bool:
    """
    Shape filter: cracks are long, thin, and sparse in their bounding box.
    Implements the paper's conclusion recommendation to remove round noise elements.
    """
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
    """
    Return a binary crack mask and useful intermediate images using the 
    Median Absolute Deviation (MAD) method as described in the paper.
    """
    # NTSC grayscale conversion (OpenCV's default matches the paper's Eq 1: 0.299R + 0.587G + 0.114B)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    full_mask = np.zeros_like(gray, dtype=np.uint8)
    sub_h, sub_w = cfg.sub_image_size, cfg.sub_image_size
    
    # 1. Divide into sub-images and apply MAD thresholding (Section 2.3)
    for y in range(0, h, sub_h):
        for x in range(0, w, sub_w):
            sub_gray = gray[y:y+sub_h, x:x+sub_w]
            if sub_gray.size == 0:
                continue
            
            pixels = sub_gray.flatten().astype(np.float32)
            med = np.median(pixels)
            mad = np.median(np.abs(pixels - med))
            
            # Threshold T = median - k * MAD
            T = med - cfg.mad_k * mad
            
            # Crack pixels are outliers with intensity < T (darker than threshold)
            sub_mask = (sub_gray < T).astype(np.uint8) * 255
            full_mask[y:y+sub_h, x:x+sub_w] = sub_mask

    # 2. Morphological cleaning to reduce noise and connect small gaps
    # (The paper notes higher k reduces noise but might create gaps; closing/opening balances this)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned_mask = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
    
    # 3. Component filtering (shape-based noise elimination as suggested in the paper's conclusion)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned_mask, connectivity=8)
    final_mask = np.zeros_like(cleaned_mask)
    max_area = int(cleaned_mask.shape[0] * cleaned_mask.shape[1] * cfg.max_component_area_fraction)
    
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if cfg.min_component_area_px <= area <= max_area:
            component = labels == label
            if _looks_like_crack(
                component, stats[label],
                cfg.min_elongation, cfg.max_fill_ratio,
            ):
                final_mask[component] = 255

    return final_mask, {
        "gray": gray,
        "mad_thresholded": full_mask,
        "mad_k": cfg.mad_k,
        "sub_image_size": cfg.sub_image_size,
    }


def measure_components(mask: np.ndarray, pixel_per_mm: float | None = None):
    """
    Measure each connected crack region in image pixel coordinates.
    Implements Section 2.5: Width Calculation via skeletonize and distance transform.
    """
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    
    # Distance transform gives distance to nearest edge (background pixel)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    results = []

    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area_px = int(stats[label, cv2.CC_STAT_AREA])
        component = labels == label

        # Centreline obtained through skeletonize function
        skeleton = skeletonize(component)
        skeleton_count = int(skeleton.sum())
        length_px = float(skeleton_count)
        
        # Width calculated as 2 * distance from skeleton to nearest edge
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
                "centroid": [float(centroids[label][0]), float(centroids[label][1])],
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