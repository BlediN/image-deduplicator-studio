"""Constants, optional dependencies and scan options."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from PIL import ImageFile

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIF_EXTENSIONS = {".heic", ".heif"}
except ImportError:
    HEIF_EXTENSIONS = set()


ImageFile.LOAD_TRUNCATED_IMAGES = True

PROJECT_DIR = Path(__file__).resolve().parent.parent
APP_HOME = Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".local" / "share") / "PhotoReview"
SETTINGS_FILE = APP_HOME / "settings.json"
MODEL_DIR = APP_HOME / "models"
LOG_FILE = PROJECT_DIR / "logs" / "photo_review.log"
FACE_MODEL_FILE = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
FACE_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
MOVE_DIR_NAME = "_marked_for_delete"
INDEX_VERSION = 2
HASH_BITS = 192  # three 64-bit hashes
ANALYSIS_SIZE = 2000  # JPEGs are decoded at reduced scale, at least this big
FACE_DETECT_SIZE = 1280
MIN_FACE_PX = 24
MIN_FACE_FRACTION = 0.025
MAIN_FACE_RATIO = 0.45  # faces smaller than this share of the largest one are background people

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
} | HEIF_EXTENSIONS
BROWSER_FORMATS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
SKIP_DIR_NAMES = {
    "_photo_review",
    MOVE_DIR_NAME,
    "__pycache__",
    "venv",
    "node_modules",
    "$RECYCLE.BIN",
    "System Volume Information",
}

if cv2 is not None:
    try:
        cv2.setNumThreads(1)  # the scan already runs one image per worker thread
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:
        pass


@dataclass
class ScanOptions:
    strong_threshold: int = 24
    burst_threshold: int = 34
    burst_seconds: int = 120
    session_threshold: int = 100
    session_seconds: int = 15
    color_threshold: int = 30
    match_rotations: bool = True
    prefer_faces: bool = True
    detect_faces: bool = True
    thumb_size: int = 480
    workers: int = min(8, os.cpu_count() or 4)


GROUPING_FIELDS = (
    "strong_threshold",
    "burst_threshold",
    "burst_seconds",
    "session_threshold",
    "session_seconds",
    "color_threshold",
    "match_rotations",
    "prefer_faces",
)
OPTION_LIMITS = {
    "strong_threshold": (0, HASH_BITS),
    "burst_threshold": (0, HASH_BITS),
    "burst_seconds": (0, 86400),
    "session_threshold": (0, HASH_BITS),
    "session_seconds": (0, 3600),
    "color_threshold": (0, 255),
    "thumb_size": (160, 1024),
    "workers": (1, 32),
}


def options_from_dict(data: dict[str, Any], base: ScanOptions) -> ScanOptions:
    values = asdict(base)
    for field in fields(ScanOptions):
        if field.name not in data:
            continue
        raw = data[field.name]
        try:
            if isinstance(values[field.name], bool):
                values[field.name] = bool(raw)
            else:
                low, high = OPTION_LIMITS[field.name]
                values[field.name] = min(high, max(low, int(raw)))
        except (TypeError, ValueError):
            pass
    return ScanOptions(**values)


def grouping_signature(options: ScanOptions) -> dict[str, Any]:
    return {name: getattr(options, name) for name in GROUPING_FIELDS}
