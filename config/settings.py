# config/settings.py
"""Application-wide settings loaded from environment variables."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ------------------------------------------------------------------
# MongoDB
# ------------------------------------------------------------------
# The database name is part of the URI, e.g.
#   mongodb://host:27017/document
MONGODB_URI: str = os.getenv("MONGODB_URI", "")

# ------------------------------------------------------------------
# Filesystem
# ------------------------------------------------------------------
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "detection_results"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
}

# ------------------------------------------------------------------
# Detection constants (legacy detector path only)
# ------------------------------------------------------------------
MIN_CRACK_AREA = 15
MAX_CRACK_AREA = 100_000
BLACKHAT_KERNEL_SIZE = 15
MORPH_KERNEL_SIZE = 3