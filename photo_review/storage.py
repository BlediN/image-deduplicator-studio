"""JSON files, app settings and the per-folder Library (index, thumbnails, marks, move log)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import asdict
from datetime import datetime
from http import HTTPStatus
from pathlib import Path
from typing import Any

from .config import APP_HOME, SETTINGS_FILE, MOVE_DIR_NAME, IMAGE_EXTENSIONS, SKIP_DIR_NAMES, ScanOptions

logger = logging.getLogger(__name__)


class ApiError(Exception):
    def __init__(self, message: str, status: HTTPStatus = HTTPStatus.BAD_REQUEST) -> None:
        super().__init__(message)
        self.status = status


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        logger.error("Cannot read %s (%s: %s). Using defaults instead.", path, type(exc).__name__, exc)
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    for attempt in range(5):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            # Sync clients and virus scanners can hold the file for a moment.
            if attempt == 4:
                logger.error("Cannot write %s: the file is locked by another program", path)
                raise
            time.sleep(0.2)


def is_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def file_id(relative_path: str) -> str:
    return hashlib.sha1(relative_path.encode("utf-8", errors="surrogatepass")).hexdigest()[:16]


def load_settings() -> dict[str, Any]:
    data = read_json(SETTINGS_FILE, {})
    return data if isinstance(data, dict) else {}


def save_settings(folder: Path | None, options: ScanOptions) -> None:
    write_json(SETTINGS_FILE, {"folder": str(folder) if folder else "", "options": asdict(options)})


# Index lookups for /image/ requests are cached by index file mtime.
_lookup_cache: dict[Path, tuple[int, dict[str, dict[str, Any]]]] = {}
INDEX_LOCK = threading.RLock()


class Library:
    """Everything the app stores for one scanned folder."""

    def __init__(self, folder: Path | str) -> None:
        self.root = Path(folder).expanduser().resolve()
        key = hashlib.sha1(str(self.root).lower().encode("utf-8", errors="surrogatepass")).hexdigest()[:12]
        self.data_dir = APP_HOME / "libraries" / key
        self.thumb_dir = self.data_dir / "thumbs"
        self.index_file = self.data_dir / "index.json"
        self.decisions_file = self.data_dir / "decisions.json"
        self.moves_file = self.data_dir / "moves.json"
        self.move_dir = self.root / MOVE_DIR_NAME

    def ensure_dirs(self) -> None:
        self.thumb_dir.mkdir(parents=True, exist_ok=True)

    def rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def load_index(self) -> dict[str, Any]:
        data = read_json(self.index_file, {})
        return data if isinstance(data, dict) else {}

    def save_index(self, payload: dict[str, Any]) -> None:
        write_json(self.index_file, payload)

    def load_decisions(self) -> dict[str, str]:
        status = read_json(self.decisions_file, {}).get("status", {})
        return {str(key): value for key, value in status.items() if value in ("keep", "move")}

    def save_decisions(self, status: dict[str, Any]) -> None:
        cleaned = {str(key): value for key, value in status.items() if value in ("keep", "move")}
        write_json(self.decisions_file, {"status": cleaned, "updated_at": datetime.now().isoformat(timespec="seconds")})

    def load_moves(self) -> list[dict[str, Any]]:
        moves = read_json(self.moves_file, [])
        return moves if isinstance(moves, list) else []

    def save_moves(self, moves: list[dict[str, Any]]) -> None:
        write_json(self.moves_file, moves[-20:])

    def image_lookup(self, ident: str) -> dict[str, Any] | None:
        try:
            mtime = self.index_file.stat().st_mtime_ns
        except OSError:
            return None
        cached = _lookup_cache.get(self.index_file)
        if not cached or cached[0] != mtime:
            images = self.load_index().get("images", [])
            cached = (mtime, {item["id"]: item for item in images})
            _lookup_cache[self.index_file] = cached
        return cached[1].get(ident)


def find_images(root: Path) -> list[Path]:
    images: list[Path] = []
    for current, dirs, files in os.walk(root):
        dirs[:] = [name for name in dirs if name not in SKIP_DIR_NAMES and not name.startswith(".")]
        current_path = Path(current)
        for filename in files:
            path = current_path / filename
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(path)
    return sorted(images, key=lambda item: item.relative_to(root).as_posix().lower())
