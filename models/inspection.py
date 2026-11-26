# models/inspection.py
"""Inspection document model: schema + persistence."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from pymongo import ASCENDING
from pymongo.collection import Collection

from config.database import get_database


COLLECTION_NAME = "cracks"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collection() -> Collection:
    """Return the cracks collection, ensuring indexes exist."""
    collection = get_database()[COLLECTION_NAME]
    collection.create_index(
        [("wall_id", ASCENDING), ("inspected_at", ASCENDING)]
    )
    collection.create_index("image.sha256")
    return collection


def build_inspection_document(
    wall_id: str,
    image_path: Path,
    image_shape: tuple[int, int, int],
    mask_path: Path,
    annotated_path: Path,
    cracks: list[dict],
    pixel_per_mm: float | None,
    detector_config: dict,
    image_hash: str | None = None,
) -> dict:
    """Assemble the full inspection document."""
    height, width = image_shape[:2]
    image_hash = image_hash or sha256_file(image_path)

    return {
        "wall_id": wall_id,
        "image": {
            "filename": image_path.name,
            "sha256": image_hash,
            "width_px": width,
            "height_px": height,
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
            "method": (
                "OpenCV black-hat + Otsu threshold + connected components"
            ),
            "config": detector_config,
        },
        "crack_count": len(cracks),
        "cracks": [
            {"crack_id": f"{wall_id}_{i:04d}", **c}
            for i, c in enumerate(cracks, start=1)
        ],
    }


def save_inspection(document: dict):
    """Insert an inspection document. Returns the inserted id."""
    document["inspected_at"] = datetime.now(timezone.utc)
    return _collection().insert_one(document).inserted_id