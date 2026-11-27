# config/detector_config.py
"""Tunable parameters for the classical crack detector."""
from dataclasses import dataclass


@dataclass
class DetectorConfig:
    # ------------------------------------------------------------------
    # Black-hat / kernel sizing
    # ------------------------------------------------------------------
    # Reference black-hat kernel size in pixels, measured on an image whose
    # shorter side is `reference_min_dim`. The actual kernel is scaled
    # proportionally to the input resolution so behaviour is stable across
    # image sizes.
    blackhat_kernel: int = 17
    reference_min_dim: int = 1500

    # If a pixel-per-mm calibration is supplied, the kernel can instead be
    # sized physically. Set to None to disable physical sizing.
    blackhat_kernel_mm: float | None = None

    # ------------------------------------------------------------------
    # Adaptive threshold (run in parallel with black-hat when enabled)
    # ------------------------------------------------------------------
    adaptive_block_size: int = 31
    adaptive_c: int = 2
    combine_adaptive_threshold: bool = True

    # ------------------------------------------------------------------
    # Hysteresis thresholds (percentiles of the black-hat response)
    # ------------------------------------------------------------------
    low_percentile: float = 95.0
    high_percentile: float = 99.0

    # ------------------------------------------------------------------
    # Morphology
    # ------------------------------------------------------------------
    close_kernel: int = 3          # scaled from the black-hat kernel at runtime
    close_iterations: int = 1

    # ------------------------------------------------------------------
    # Component filtering
    # ------------------------------------------------------------------
    min_component_area_px: int = 8
    max_component_area_fraction: float = 0.25
    min_elongation: float = 3.0    # (skeleton_length^2) / area
    max_fill_ratio: float = 0.5    # area / bounding_box_area

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8
    flatten_background: bool = True  # large-scale background subtraction