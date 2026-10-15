"""Per-image analysis (hashes, sharpness, faces, thumbnails) and the folder scan."""

from __future__ import annotations

import hashlib
import io
import logging
import math
import os
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import ExifTags, Image, ImageFilter, ImageOps

from .config import (
    cv2,
    MODEL_DIR,
    FACE_MODEL_FILE,
    FACE_MODEL_URL,
    INDEX_VERSION,
    ANALYSIS_SIZE,
    FACE_DETECT_SIZE,
    MIN_FACE_PX,
    MIN_FACE_FRACTION,
    MAIN_FACE_RATIO,
    ScanOptions,
)
from .storage import file_id, INDEX_LOCK, Library, find_images
from .grouping import build_groups, keep_best_shots

logger = logging.getLogger(__name__)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def bits_to_hex(bits: np.ndarray) -> str:
    bit_string = "".join("1" if bool(bit) else "0" for bit in bits.flatten())
    return f"{int(bit_string, 2):0{len(bit_string) // 4}x}"


def average_hash(gray: Image.Image) -> str:
    small = gray.resize((8, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(small, dtype=np.float32)
    return bits_to_hex(pixels >= pixels.mean())


def difference_hash(gray: Image.Image) -> str:
    small = gray.resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(small, dtype=np.float32)
    return bits_to_hex(pixels[:, 1:] > pixels[:, :-1])


def perceptual_hash(gray: Image.Image) -> str:
    small = gray.resize((32, 32), Image.Resampling.LANCZOS)
    pixels = np.asarray(small, dtype=np.float32)
    pixels -= pixels.mean()
    dct = dct_2d(pixels)
    low = dct[:8, :8]
    flattened = low.flatten()
    median = np.median(flattened[1:])
    return bits_to_hex(flattened >= median)


def dct_2d(values: np.ndarray) -> np.ndarray:
    size = values.shape[0]
    coords = np.arange(size)
    matrix = np.cos((np.pi / size) * (coords[:, None] + 0.5) * coords[None, :])
    matrix[:, 0] *= math.sqrt(1 / size)
    matrix[:, 1:] *= math.sqrt(2 / size)
    return matrix.T @ values @ matrix


ROTATIONS = (None, Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270)


def image_hashes(gray: Image.Image) -> list[list[str]]:
    """aHash/dHash/pHash of the image at 0, 90, 180 and 270 degrees (counter-clockwise)."""
    base = gray.resize((64, 64), Image.Resampling.LANCZOS)
    hashes = []
    for rotation in ROTATIONS:
        rotated = base if rotation is None else base.transpose(rotation)
        hashes.append([average_hash(rotated), difference_hash(rotated), perceptual_hash(rotated)])
    return hashes


def color_signature(rgb: Image.Image) -> list[int]:
    """Average colour of a 3x3 grid, used to tell apart scenes that only match in grayscale."""
    grid = np.asarray(rgb.resize((3, 3), Image.Resampling.BOX), dtype=np.uint8)
    return [int(value) for value in grid.flatten()]


def laplacian_variance(arr: np.ndarray) -> float:
    arr = arr.astype(np.float32, copy=False)
    if arr.shape[0] < 3 or arr.shape[1] < 3:
        return 0.0
    lap = arr[:-2, 1:-1] + arr[2:, 1:-1] + arr[1:-1, :-2] + arr[1:-1, 2:] - 4 * arr[1:-1, 1:-1]
    return float(lap.var())


def sharpness_of(gray: Image.Image) -> float:
    sample = gray.copy()
    sample.thumbnail((360, 360), Image.Resampling.LANCZOS)
    # A light blur stops sensor noise from being scored as fine detail.
    sample = sample.filter(ImageFilter.GaussianBlur(0.7))
    return laplacian_variance(np.asarray(sample, dtype=np.float32))


def photo_score(
    width: int, height: int, size: int, sharpness: float, brightness: float, contrast: float, clipped: float
) -> float:
    megapixels = (width * height) / 1_000_000
    resolution_score = min(30.0, math.log1p(megapixels) * 14.0)
    sharpness_score = min(42.0, math.log1p(max(sharpness, 0.0)) * 6.0)
    # Mild penalty for overall darkness/brightness, stronger for blown highlights and crushed shadows.
    exposure_score = clamp(1.0 - abs(brightness - 122.0) / 200.0 - clipped * 3.0) * 18.0
    contrast_score = min(10.0, contrast / 7.0)
    size_score = min(6.0, math.log1p(max(size, 1)) / 3.0)
    return round(resolution_score + sharpness_score + exposure_score + contrast_score + size_score, 2)


def exif_capture_time(exif: Image.Exif) -> float | None:
    try:
        sub_ifd = exif.get_ifd(ExifTags.IFD.Exif)
    except Exception:
        sub_ifd = {}
    subsec = str(sub_ifd.get(37521) or "").strip().strip("\x00")
    fraction = float(f"0.{subsec}") if subsec.isdigit() else 0.0
    # DateTimeOriginal and DateTimeDigitized live in the Exif sub-IFD, DateTime in IFD0.
    for value in (sub_ifd.get(36867), sub_ifd.get(36868), exif.get(36867), exif.get(306)):
        if isinstance(value, bytes):
            value = value.decode(errors="ignore")
        text = str(value or "").strip().strip("\x00")[:19]
        if not text:
            continue
        for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(text, fmt).timestamp() + fraction
            except (ValueError, OSError, OverflowError):
                pass
    return None


def to_rgb(image: Image.Image) -> Image.Image:
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (245, 247, 244))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return image.convert("RGB")


def build_thumbnail(rgb: Image.Image, destination: Path, size: int) -> None:
    thumb = rgb.copy()
    thumb.thumbnail((size, size), Image.Resampling.LANCZOS)
    thumb.save(destination, "JPEG", quality=82, optimize=True)


_face_model_lock = threading.Lock()
_face_model_error: str | None = None
_thread_local = threading.local()


def face_model_path() -> tuple[Path | None, str]:
    """Return the YuNet face model, downloading it once. The string explains why it is missing."""
    global _face_model_error
    if cv2 is None:
        return None, "OpenCV is not installed. Run: pip install opencv-python-headless, then rescan."
    if not hasattr(cv2, "FaceDetectorYN"):
        return None, "This OpenCV build has no face detector. Run: pip install -U opencv-python-headless."
    with _face_model_lock:
        if FACE_MODEL_FILE.exists():
            return FACE_MODEL_FILE, ""
        if _face_model_error:
            return None, _face_model_error
        try:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            temp = FACE_MODEL_FILE.with_suffix(".download")
            urllib.request.urlretrieve(FACE_MODEL_URL, temp)
            os.replace(temp, FACE_MODEL_FILE)
            return FACE_MODEL_FILE, ""
        except OSError as exc:
            _face_model_error = f"Could not download the face model ({exc}). Check the internet connection and rescan."
            logger.warning("Face model download from %s failed: %s", FACE_MODEL_URL, exc)
            return None, _face_model_error


def face_detector(model: Path) -> Any:
    detector = getattr(_thread_local, "detector", None)
    if detector is None:
        detector = cv2.FaceDetectorYN.create(str(model), "", (320, 320), 0.75, 0.3, 50)
        _thread_local.detector = detector
    return detector


def patch_contrast(gray: np.ndarray, center: np.ndarray, eye_dist: float) -> float:
    cx, cy = float(center[0]), float(center[1])
    half_w, half_h = max(2.0, 0.22 * eye_dist), max(2.0, 0.13 * eye_dist)
    patch = gray[int(max(0, cy - half_h)) : int(cy + half_h) + 1, int(max(0, cx - half_w)) : int(cx + half_w) + 1]
    if patch.size < 9:
        return 0.0
    low, high = np.percentile(patch, (5, 95))
    return float(high - low)


def detect_faces(rgb: Image.Image, gray: np.ndarray, model: Path) -> list[dict[str, float]]:
    """Find the main faces and describe how each one is posed.

    YuNet returns a box plus five landmarks per face (both eyes, nose tip, mouth
    corners). From those we estimate:
      frontal - 1 when looking straight at the camera, towards 0 when turned away
      eyes    - contrast inside the eye regions; open eyes (iris + white) score higher
      mouth   - mouth width / eye distance; wider usually means smiling
      sharp   - focus of the face itself, independent of the background
    Eyes and mouth are only meaningful relative to other shots of the same people,
    so the ranking compares them within a group.
    """
    small = rgb.copy()
    small.thumbnail((FACE_DETECT_SIZE, FACE_DETECT_SIZE), Image.Resampling.BILINEAR)
    det_w, det_h = small.size
    bgr = np.ascontiguousarray(np.asarray(small)[:, :, ::-1])
    detector = face_detector(model)
    detector.setInputSize((det_w, det_h))
    _, raw = detector.detect(bgr)
    if raw is None or len(raw) == 0:
        return []

    scale = gray.shape[1] / det_w
    longest = max(det_w, det_h)
    found: list[dict[str, float]] = []
    for row in raw:
        x, y, w, h = (float(value) for value in row[:4])
        if w < max(MIN_FACE_PX, MIN_FACE_FRACTION * longest):
            continue
        right_eye, left_eye, nose, mouth_right, mouth_left = (
            row[i : i + 2].astype(np.float64) for i in range(4, 14, 2)
        )
        eye_dist = float(np.hypot(*(left_eye - right_eye)))
        if eye_dist < 4:
            continue
        eye_mid = (right_eye + left_eye) / 2
        mouth_mid = (mouth_right + mouth_left) / 2
        # Head turned sideways: nose moves off the eye midline and the eyes get closer together.
        turn = clamp(1 - abs(nose[0] - eye_mid[0]) / eye_dist / 0.45) * clamp((eye_dist / w - 0.15) / 0.22)
        # Head tilted down/up: nose moves towards the mouth or the eyes.
        span = mouth_mid[1] - eye_mid[1]
        tilt = clamp(1 - abs((nose[1] - eye_mid[1]) / span - 0.55) / 0.35) if span > 1 else 0.0

        x0, y0 = int(max(0, x * scale)), int(max(0, y * scale))
        x1, y1 = int(min(gray.shape[1], (x + w) * scale)), int(min(gray.shape[0], (y + h) * scale))
        face = gray[y0:y1, x0:x1]
        if face.shape[0] < 8 or face.shape[1] < 8:
            continue
        low, high = np.percentile(face, (5, 95))
        face_range = float(high - low) + 1.0
        eyes = (
            patch_contrast(gray, right_eye * scale, eye_dist * scale)
            + patch_contrast(gray, left_eye * scale, eye_dist * scale)
        ) / 2 / face_range
        face_img = Image.fromarray(np.ascontiguousarray(face)).resize((128, 128), Image.Resampling.BILINEAR)
        found.append(
            {
                "size": round(w / longest, 4),
                "conf": round(float(row[14]), 3),
                "frontal": round(0.8 * turn + 0.2 * tilt, 3),
                "eyes": round(float(eyes), 4),
                "mouth": round(float(np.hypot(*(mouth_left - mouth_right))) / eye_dist, 4),
                "sharp": round(laplacian_variance(np.asarray(face_img, dtype=np.float32)), 2),
            }
        )
    if not found:
        return []
    largest = max(face["size"] for face in found)
    main = [face for face in found if face["size"] >= MAIN_FACE_RATIO * largest]
    main.sort(key=lambda face: face["size"], reverse=True)
    return main[:8]


def analyze_image(library: Library, path: Path, options: ScanOptions, model: Path | None) -> dict[str, Any]:
    relative = library.rel(path)
    stat = path.stat()
    ident = file_id(relative)
    data = path.read_bytes()

    with Image.open(io.BytesIO(data)) as raw:
        exif = raw.getexif()
        capture_ts = exif_capture_time(exif)
        width, height = raw.size
        if exif.get(0x0112) in (5, 6, 7, 8):
            width, height = height, width
        raw.draft("RGB", (ANALYSIS_SIZE, ANALYSIS_SIZE))  # fast reduced-scale JPEG decode
        rgb = to_rgb(ImageOps.exif_transpose(raw))

    gray_image = rgb.convert("L")
    gray = np.asarray(gray_image)
    sample = gray_image.copy()
    sample.thumbnail((512, 512), Image.Resampling.LANCZOS)
    arr = np.asarray(sample, dtype=np.float32)
    brightness = float(arr.mean())
    contrast = float(arr.std())
    clipped = float(((arr < 6) | (arr > 249)).mean())
    sharpness = sharpness_of(gray_image)
    faces = detect_faces(rgb, gray, model) if model else None
    build_thumbnail(rgb, library.thumb_dir / f"{ident}.jpg", options.thumb_size)

    return {
        "version": INDEX_VERSION,
        "id": ident,
        "rel_path": relative,
        "name": path.name,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "capture_ts": capture_ts if capture_ts is not None else stat.st_mtime,
        "ts_source": "exif" if capture_ts is not None else "mtime",
        "width": width,
        "height": height,
        "sha256": hashlib.sha256(data).hexdigest(),
        "hashes": image_hashes(gray_image),
        "colors": color_signature(rgb),
        "brightness": round(brightness, 3),
        "contrast": round(contrast, 3),
        "clipped": round(clipped, 4),
        "sharpness": round(sharpness, 3),
        "faces": faces,
        "score": photo_score(width, height, stat.st_size, sharpness, brightness, contrast, clipped),
    }


ProgressFn = Callable[..., None]


def scan(library: Library, options: ScanOptions, force: bool = False, progress: ProgressFn | None = None) -> dict[str, Any]:
    report = progress or (lambda *args, **kwargs: None)
    if not library.root.is_dir():
        raise FileNotFoundError(f"Folder not found: {library.root}")
    library.ensure_dirs()
    report("Finding photos")
    previous = library.load_index()
    previous_by_path = {item["rel_path"]: item for item in previous.get("images", [])}
    paths = find_images(library.root)

    model, faces_reason = (None, "Face detection is turned off in Settings.")
    if options.detect_faces:
        report("Preparing face detection")
        model, faces_reason = face_model_path()

    images: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    todo: list[Path] = []
    for path in paths:
        relative = library.rel(path)
        try:
            stat = path.stat()
        except OSError as exc:
            errors.append({"rel_path": relative, "error": str(exc)})
            logger.warning("Cannot read %s: %s", relative, exc)
            continue
        cached = previous_by_path.get(relative)
        if (
            not force
            and cached
            and cached.get("version") == INDEX_VERSION
            and cached.get("size") == stat.st_size
            and cached.get("mtime_ns") == stat.st_mtime_ns
            and (library.thumb_dir / f"{cached['id']}.jpg").exists()
            and not (model and cached.get("faces") is None)
        ):
            images.append(cached)
        else:
            todo.append(path)

    logger.debug(
        "Found %d photos: %d cached, %d to analyze | faces: %s",
        len(paths), len(images), len(todo), "on" if model else faces_reason,
    )
    report("Analyzing photos", 0, len(todo))
    if todo:
        with ThreadPoolExecutor(max_workers=options.workers) as pool:
            futures = {pool.submit(analyze_image, library, path, options, model): path for path in todo}
            for done, future in enumerate(as_completed(futures), start=1):
                try:
                    images.append(future.result())
                except Exception as exc:  # one unreadable file must not stop the scan
                    relative = library.rel(futures[future])
                    errors.append({"rel_path": relative, "error": f"{type(exc).__name__}: {exc}"})
                    logger.warning("Cannot analyze %s: %s: %s", relative, type(exc).__name__, exc,
                                exc_info=logger.isEnabledFor(logging.DEBUG))
                report("Analyzing photos", done, len(todo))
    images.sort(key=lambda item: item["rel_path"].lower())

    # Drop thumbnails of files that are gone, but keep those needed to undo a move.
    wanted = {f"{item['id']}.jpg" for item in images}
    wanted |= {f"{entry['id']}.jpg" for batch in library.load_moves() for entry in batch.get("items", [])}
    for thumb in library.thumb_dir.glob("*.jpg"):
        if thumb.name not in wanted:
            thumb.unlink(missing_ok=True)

    report("Grouping similar photos")
    payload = {
        "version": INDEX_VERSION,
        "root": str(library.root),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "faces_available": model is not None,
        "faces_reason": faces_reason,
        "images": images,
        "errors": errors,
    }
    payload["grouping"] = build_groups(images, options, payload)
    with INDEX_LOCK:
        library.save_index(payload)
        keep_best_shots(library, payload["grouping"]["groups"])
    return payload
