# config/__init__.py
from .settings import MONGODB_URI, OUTPUT_DIR, SUPPORTED_EXTENSIONS
from .database import get_client, get_database
from .detector_config import DetectorConfig

__all__ = [
    "MONGODB_URI",
    "OUTPUT_DIR",
    "SUPPORTED_EXTENSIONS",
    "get_client",
    "get_database",
    "DetectorConfig",
]