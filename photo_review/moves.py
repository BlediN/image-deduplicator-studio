"""Move marked photos into _marked_for_delete and undo the last move."""

from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from .storage import ApiError, is_inside, INDEX_LOCK, Library

logger = logging.getLogger(__name__)


def safe_destination(library: Library, relative: str) -> Path:
    destination = library.move_dir / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        return destination
    counter = 2
    while True:
        candidate = destination.parent / f"{destination.stem}_{counter}{destination.suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def log_outcome(action: str, count: int, errors: list[dict[str, str]]) -> None:
    logger.info("%s %d photos, %d errors", action, count, len(errors))
    for error in errors:
        logger.warning("%s: %s (id %s)", action, error["error"], error["id"])


def move_files(library: Library, ids: list[str]) -> dict[str, Any]:
    moved: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    batch: list[dict[str, Any]] = []
    with INDEX_LOCK:
        index = library.load_index()
        lookup = {item["id"]: item for item in index.get("images", [])}
        library.move_dir.mkdir(exist_ok=True)
        for ident in dict.fromkeys(ids):
            item = lookup.get(ident)
            if not item:
                errors.append({"id": ident, "error": "Not found in the index"})
                continue
            source = (library.root / item["rel_path"]).resolve()
            if not is_inside(source, library.root) or is_inside(source, library.move_dir):
                errors.append({"id": ident, "error": "Unsafe source path"})
                continue
            if not source.exists():
                errors.append({"id": ident, "error": f"{item['rel_path']} is missing"})
                continue
            destination = safe_destination(library, item["rel_path"]).resolve()
            if not is_inside(destination, library.move_dir):
                errors.append({"id": ident, "error": "Unsafe destination path"})
                continue
            try:
                shutil.move(str(source), str(destination))
            except OSError as exc:
                errors.append({"id": ident, "error": f"{item['rel_path']}: {exc}"})
                continue
            target = destination.relative_to(library.root).as_posix()
            moved.append({"id": ident, "from": item["rel_path"], "to": target})
            logger.debug("Moved %s -> %s", item["rel_path"], target)
            batch.append({"id": ident, "from": item["rel_path"], "to": target, "entry": item})

        if batch:
            gone = {entry["id"] for entry in batch}
            index["images"] = [item for item in index.get("images", []) if item["id"] not in gone]
            index.pop("grouping", None)
            library.save_index(index)
            moves = library.load_moves()
            moves.append({"at": datetime.now().isoformat(timespec="seconds"), "items": batch})
            library.save_moves(moves)
            status = library.load_decisions()
            for ident in gone:
                status.pop(ident, None)
            library.save_decisions(status)
    log_outcome("Moved", len(moved), errors)
    return {"moved": moved, "errors": errors}


def undo_last_move(library: Library) -> dict[str, Any]:
    restored: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    with INDEX_LOCK:
        moves = library.load_moves()
        if not moves:
            raise ApiError("There is no move to undo.")
        batch = moves.pop()
        index = library.load_index()
        images = index.setdefault("images", [])
        status = library.load_decisions()
        failed: list[dict[str, Any]] = []
        for entry in batch.get("items", []):
            source = (library.root / entry["to"]).resolve()
            destination = (library.root / entry["from"]).resolve()
            if not is_inside(source, library.move_dir) or not is_inside(destination, library.root):
                errors.append({"id": entry["id"], "error": "Unsafe path"})
                continue
            if not source.exists():
                errors.append({"id": entry["id"], "error": f"{entry['to']} is missing"})
                continue
            if destination.exists():
                errors.append({"id": entry["id"], "error": f"{entry['from']} already exists"})
                failed.append(entry)
                continue
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(destination))
            except OSError as exc:
                errors.append({"id": entry["id"], "error": str(exc)})
                failed.append(entry)
                continue
            item = dict(entry["entry"])
            stat = destination.stat()
            item["size"], item["mtime_ns"] = stat.st_size, stat.st_mtime_ns
            images.append(item)
            status[item["id"]] = "move"  # restore the mark so a single mistake is easy to fix
            restored.append({"id": item["id"], "path": entry["from"]})
            logger.debug("Restored %s", entry["from"])
        if failed:
            moves.append({"at": batch.get("at"), "items": failed})
        images.sort(key=lambda item: item["rel_path"].lower())
        index.pop("grouping", None)
        library.save_index(index)
        library.save_moves(moves)
        library.save_decisions(status)
    log_outcome("Undo restored", len(restored), errors)
    return {"restored": restored, "errors": errors}
