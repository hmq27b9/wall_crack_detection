"""Tunable parameters for the MAD-based crack detector."""
from dataclasses import dataclass


@dataclass
class DetectorConfig:
    # ------------------------------------------------------------------
    # MAD (Median Absolute Deviation) Method Parameters
    # ------------------------------------------------------------------
    # Reference: "Image-Based Concrete Crack Detection Method Using the Median Absolute Deviation"
    # Section 2.2: Sub-images of 224x224 pixels
    sub_image_size: int = 224
    
    # Section 2.3: Scale factor for MAD threshold. 
    # Paper sensitivity analysis suggests k=2.5 provides excellent POD (97% for 0.1mm) 
    # while keeping noise manageable.
    mad_k: float = 2.5

    # ------------------------------------------------------------------
    # Component filtering (Shape-based noise elimination)
    # ------------------------------------------------------------------
    # As recommended in the paper's Conclusion: "a post-processing step will be 
    # introduced after applying the MAD procedure to eliminate noise based on the 
    # shape of detected elements... to identify and remove round elements"
    min_component_area_px: int = 8
    max_component_area_fraction: float = 0.25
    min_elongation: float = 3.0    # (skeleton_length^2) / area
    max_fill_ratio: float = 0.5    # area / bounding_box_area