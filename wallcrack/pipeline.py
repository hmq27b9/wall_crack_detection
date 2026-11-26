from __future__ import annotations

import hashlib
import json
from pathlib import Path
import cv2

from .config import DetectorConfig
from .detector import segment_cracks, measure_components, annotate_image


SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def process_image(
    image_path: Path,
    output_dir: Path,
    wall_id: str,
    pixel_per_mm: float | None,
    repository=None,
    config: DetectorConfig | None = None,
):
    config = config or DetectorConfig()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    mask, intermediate = segment_cracks(image, config)
    cracks = measure_components(mask, pixel_per_mm)
    annotated = annotate_image(image, cracks)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    mask_path = output_dir / f"{stem}_mask.png"
    annotated_path = output_dir / f"{stem}_annotated.jpg"
    cv2.imwrite(str(mask_path), mask)
    cv2.imwrite(str(annotated_path), annotated)

    image_height, image_width = image.shape[:2]
    document = {
        "wall_id": wall_id,
        "image": {
            "filename": image_path.name,
            "sha256": sha256_file(image_path),
            "width_px": image_width,
            "height_px": image_height,
            "source_path": str(image_path.resolve()),
        },
        "artifacts": {
            "mask_path": str(mask_path.resolve()),
            "annotated_path": str(annotated_path.resolve()),
        },
        "calibration": {
            "pixel_per_mm": pixel_per_mm,
            "note": (
                "Physical dimensions are approximate and require a scale "
                "reference in the same plane as the wall."
                if pixel_per_mm else
                "No pixel-to-mm calibration supplied; measurements are pixels."
            ),
        },
        "detector": {
            "method": "OpenCV black-hat + Otsu threshold + connected components",
            "config": config.__dict__,
        },
        "crack_count": len(cracks),
        "cracks": [
            {"crack_id": f"{wall_id}_{index:04d}", **crack}
            for index, crack in enumerate(cracks, start=1)
        ],
    }

    json_path = output_dir / f"{stem}_results.json"
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, ensure_ascii=False)

    inserted_id = None
    if repository is not None:
        inserted_id = repository.save_inspection(document)

    return {
        "source": str(image_path),
        "crack_count": len(cracks),
        "mask": str(mask_path),
        "annotated": str(annotated_path),
        "json": str(json_path),
        "mongodb_id": str(inserted_id) if inserted_id is not None else None,
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
