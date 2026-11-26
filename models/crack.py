# models/crack.py
"""Crack domain model: normalises raw detector output into DB records."""
from __future__ import annotations


def build_crack_record(crack_index: int, wall_id: str, raw: dict) -> dict:
    """Convert a raw measurement dict from the detector into the
    MongoDB-ready crack record."""
    return {
        "crack_id": f"{wall_id}_{crack_index:04d}",
        **raw,
    }