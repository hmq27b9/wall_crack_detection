from __future__ import annotations

import json
from pathlib import Path

import cv2

from config.detector_config import DetectorConfig
from config.settings import SUPPORTED_EXTENSIONS
from models import build_inspection_document, save_inspection
from .detector import segment_cracks, measure_components, annotate_image


SUPPORTED = SUPPORTED_EXTENSIONS


def process_image(
    image_path: Path,
    output_dir: Path,
    wall_id: str,
    pixel_per_mm: float | None,
    config: DetectorConfig | None = None,
):
    config = config or DetectorConfig()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    mask, intermediate = segment_cracks(image, config, pixel_per_mm)
    cracks = measure_components(mask, pixel_per_mm)
    annotated = annotate_image(image, cracks)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    mask_path = output_dir / f"{stem}_mask.png"
    annotated_path = output_dir / f"{stem}_annotated.jpg"
    cv2.imwrite(str(mask_path), mask)
    cv2.imwrite(str(annotated_path), annotated)

    diagnostics = {
        "mad_k": intermediate.get("mad_k"),
        "sub_image_size": intermediate.get("sub_image_size"),
        "config": config.__dict__,
    }

    document = build_inspection_document(
        wall_id=wall_id,
        image_path=image_path,
        image_shape=image.shape,
        mask_path=mask_path,
        annotated_path=annotated_path,
        cracks=cracks,
        pixel_per_mm=pixel_per_mm,
        detector_config=config.__dict__,
        diagnostics=diagnostics,
    )

    json_path = output_dir / f"{stem}_results.json"
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, ensure_ascii=False)

    inserted_id = save_inspection(document)

    return {
        "source": str(image_path),
        "crack_count": len(cracks),
        "mask": str(mask_path),
        "annotated": str(annotated_path),
        "json": str(json_path),
        "mongodb_id": str(inserted_id),
    }


def find_images(input_path: Path):
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED:
            raise ValueError(f"Unsupported image format: {input_path.suffix}")
        return [input_path]
    if input_path.is_dir():
        return sorted(
            p for p in input_path.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED
        )
    raise FileNotFoundError(input_path)