# wallcrack/config.py
from dataclasses import dataclass

@dataclass
class DetectorConfig:
    # Black-hat kernel should be larger than the expected crack width.
    blackhat_kernel: int = 17
    adaptive_block_size: int = 31
    adaptive_c: int = 7
    min_component_area_px: int = 12
    max_component_area_fraction: float = 0.25
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8
    close_kernel: int = 3
    close_iterations: int = 1
    combine_adaptive_threshold: bool = False
