"""App state and the local HTTP server behind the browser UI."""

from __future__ import annotations

import io
import json
import logging
import mimetypes
import posixpath
import re
import secrets
import threading
import time
import urllib.parse
import webbrowser
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from .config import INDEX_VERSION, BROWSER_FORMATS, ScanOptions, options_from_dict, grouping_signature
from .storage import ApiError, is_inside, save_settings, INDEX_LOCK, Library
from .analysis import scan, to_rgb
from .grouping import build_groups, keep_best_shots
from .moves import move_files, undo_last_move

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_FILES = {
    "/static/app.css": "text/css; charset=utf-8",
    "/static/app.js": "text/javascript; charset=utf-8",
}


def pick_folder(initial: str) -> str | None:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
        root.update()
        start = initial if initial and Path(initial).is_dir() else str(Path.home())
        chosen = filedialog.askdirectory(parent=root, initialdir=start, mustexist=True, title="Choose a photo folder to scan")
    finally:
        root.destroy()
    return str(Path(chosen)) if chosen else None


class App:
    def __init__(self, folder: Path | None, options: ScanOptions, host: str) -> None:
        self.options = options
        self.library = Library(folder) if folder and folder.is_dir() else None
        self.host = host
        self.token = secrets.token_urlsafe(24)
        self.job_lock = threading.Lock()
        self.dialog_lock = threading.Lock()
        self.job: dict[str, Any] = {"running": False, "phase": "", "done": 0, "total": 0, "message": "", "error": None}

    def job_snapshot(self) -> dict[str, Any]:
        with self.job_lock:
            return dict(self.job)

    def require_library(self) -> Library:
        if self.library is None:
            raise ApiError("Choose a folder and scan it first.")
        return self.library

    def require_idle(self) -> None:
        if self.job_snapshot()["running"]:
            raise ApiError("Wait for the scan to finish first.", HTTPStatus.CONFLICT)

    def start_scan(self, folder: Path, force: bool) -> dict[str, Any]:
        with self.job_lock:
            if self.job["running"]:
                raise ApiError("A scan is already running.", HTTPStatus.CONFLICT)
            library = Library(folder)
            self.library = library
            self.job = {"running": True, "phase": "Starting", "done": 0, "total": 0, "message": "", "error": None}
            snapshot = dict(self.job)
        save_settings(library.root, self.options)
        logger.info("Scan started: %s (force=%s)", library.root, force)
        threading.Thread(target=self._run_scan, args=(library, force), daemon=True).start()
        return snapshot

    def _progress(self, phase: str, done: int = 0, total: int = 0) -> None:
        with self.job_lock:
            self.job.update(phase=phase, done=done, total=total)

    def _run_scan(self, library: Library, force: bool) -> None:
        message, error = "", None
        start = time.perf_counter()
        try:
            payload = scan(library, self.options, force=force, progress=self._progress)
            summary = payload["grouping"]["summary"]
            message = f"Scan complete: {summary['image_count']} photos, {summary['group_count']} groups to review."
            if payload["errors"]:
                message += f" {len(payload['errors'])} files could not be read."
            logger.info("%s Took %.1fs.", message, time.perf_counter() - start)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.exception("Scan of %s failed", library.root)
        with self.job_lock:
            self.job.update(running=False, phase="Finished", message=message, error=error)

    def update_options(self, data: dict[str, Any]) -> None:
        self.options = options_from_dict(data, self.options)
        logger.info("Settings changed: %s", asdict(self.options))
        save_settings(self.library.root if self.library else None, self.options)

    def current_state(self) -> dict[str, Any]:
        library = self.library
        state: dict[str, Any] = {
            "folder": str(library.root) if library else "",
            "move_dir": str(library.move_dir) if library else "",
            "options": asdict(self.options),
            "job": self.job_snapshot(),
            "scanned": False,
            "groups": [],
            "summary": {},
            "errors": [],
            "decisions": {},
            "can_undo": False,
        }
        if library is None or not library.index_file.exists():
            return state
        with INDEX_LOCK:
            index = library.load_index()
            if index.get("version") != INDEX_VERSION:
                return state
            grouping = index.get("grouping")
            if not grouping or grouping.get("signature") != grouping_signature(self.options):
                grouping = build_groups(index.get("images", []), self.options, index)
                index["grouping"] = grouping
                library.save_index(index)
                keep_best_shots(library, grouping["groups"])
            state.update(
                scanned=True,
                groups=grouping["groups"],
                summary=grouping["summary"],
                errors=index.get("errors", []),
                created_at=index.get("created_at"),
                decisions=library.load_decisions(),
                can_undo=bool(library.load_moves()),
            )
        return state


class ReviewHandler(BaseHTTPRequestHandler):
    app: App
    server_version = "PhotoReview/2"

    def handle_one_request(self) -> None:
        self.started = time.perf_counter()
        super().handle_one_request()

    def log_request(self, code: Any = "-", size: Any = "-") -> None:
        code = getattr(code, "value", code)
        level = logging.WARNING if str(code).startswith("5") else logging.INFO if str(code).startswith("4") else logging.DEBUG
        elapsed = (time.perf_counter() - getattr(self, "started", time.perf_counter())) * 1000
        logger.log(level, "%s %s -> %s (%.0f ms)", self.command, self.path.split("?")[0], code, elapsed)

    def log_error(self, format: str, *args: Any) -> None:
        logger.debug("%s: %s", self.address_string(), format % args)  # log_request already reports the status

    def log_message(self, format: str, *args: Any) -> None:
        logger.info("%s: %s", self.address_string(), format % args)

    def host_ok(self) -> bool:
        # Blocks DNS-rebinding: only answer requests addressed to this machine.
        host = (self.headers.get("Host") or "").lower()
        name = host[: host.rindex(":")] if ":" in host and not host.endswith("]") else host
        return name in {"127.0.0.1", "localhost", "[::1]", self.app.host.lower()}

    def token_ok(self) -> bool:
        return secrets.compare_digest(self.headers.get("X-Review-Token", ""), self.app.token)

    def do_GET(self) -> None:
        if not self.host_ok():
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        path = urllib.parse.urlparse(self.path).path
        if path == "/":
            html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
            self.send_bytes(html.replace("__TOKEN__", self.app.token).encode("utf-8"), "text/html; charset=utf-8")
        elif path in STATIC_FILES:
            self.send_bytes((STATIC_DIR / posixpath.basename(path)).read_bytes(), STATIC_FILES[path])
        elif path.startswith("/api/"):
            if not self.token_ok():
                self.send_json({"error": "Session expired. Reload the page."}, HTTPStatus.FORBIDDEN)
            elif path == "/api/state":
                self.send_json(self.app.current_state())
            elif path == "/api/job":
                self.send_json(self.app.job_snapshot())
            else:
                self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
        elif path.startswith("/thumb/"):
            self.serve_thumb(path)
        elif path.startswith("/image/"):
            self.serve_image(path)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if not self.host_ok() or not self.token_ok():
            self.send_json({"error": "Session expired. Reload the page."}, HTTPStatus.FORBIDDEN)
            return
        path = urllib.parse.urlparse(self.path).path
        try:
            data = self.read_body()
            self.send_json(self.route_post(path, data))
        except ApiError as exc:
            logger.info("API error on %s: %s", path, exc)
            self.send_json({"error": str(exc)}, exc.status)

    def route_post(self, path: str, data: dict[str, Any]) -> Any:
        app = self.app
        if path == "/api/pick-folder":
            if not app.dialog_lock.acquire(blocking=False):
                raise ApiError("A folder picker is already open.", HTTPStatus.CONFLICT)
            try:
                return {"path": pick_folder(str(data.get("initial") or ""))}
            except Exception as exc:
                raise ApiError(f"Could not open the folder picker ({exc}). Type the path instead.") from exc
            finally:
                app.dialog_lock.release()
        if path == "/api/scan":
            folder = Path(str(data.get("folder") or "")).expanduser()
            if not str(folder) or not folder.is_dir():
                raise ApiError(f"Folder not found: {folder}")
            return app.start_scan(folder, bool(data.get("force")))
        if path == "/api/settings":
            options = data.get("options")
            if not isinstance(options, dict):
                raise ApiError("options must be an object")
            app.update_options(options)
            return app.current_state()
        if path == "/api/decisions":
            library = app.require_library()
            if data.get("folder") != str(library.root):
                raise ApiError("The folder changed. Reload the page.", HTTPStatus.CONFLICT)
            status = data.get("status")
            if not isinstance(status, dict):
                raise ApiError("status must be an object")
            library.save_decisions(status)
            return {"ok": True}
        if path == "/api/move":
            app.require_idle()
            ids = data.get("ids")
            if not isinstance(ids, list):
                raise ApiError("ids must be a list")
            return move_files(app.require_library(), [str(item) for item in ids])
        if path == "/api/undo":
            app.require_idle()
            return undo_last_move(app.require_library())
        raise ApiError("Not found", HTTPStatus.NOT_FOUND)

    def read_body(self) -> dict[str, Any]:
        if not (self.headers.get("Content-Type") or "").startswith("application/json"):
            raise ApiError("Expected a JSON body", HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ApiError("Invalid JSON") from exc
        return data if isinstance(data, dict) else {}

    def send_bytes(self, data: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK, cache: bool = False) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "private, max-age=86400" if cache else "no-store")
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # the browser navigated away mid-response

    def send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_bytes(json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8", status)

    def serve_thumb(self, path: str) -> None:
        library = self.app.library
        ident = posixpath.basename(path).removesuffix(".jpg")
        if library is None or not re.fullmatch(r"[0-9a-f]{16}", ident):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            data = (library.thumb_dir / f"{ident}.jpg").read_bytes()
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_bytes(data, "image/jpeg", cache=True)

    def serve_image(self, path: str) -> None:
        library = self.app.library
        item = library.image_lookup(posixpath.basename(path)) if library else None
        if not item:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        source = (library.root / item["rel_path"]).resolve()
        if not is_inside(source, library.root) or not source.exists():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            if source.suffix.lower() in BROWSER_FORMATS:
                data = source.read_bytes()
                mime = mimetypes.guess_type(source.name)[0] or "application/octet-stream"
            else:
                # HEIC/TIFF/BMP: convert so every browser can show it.
                with Image.open(source) as raw:
                    image = to_rgb(ImageOps.exif_transpose(raw))
                image.thumbnail((2560, 2560), Image.Resampling.LANCZOS)
                buffer = io.BytesIO()
                image.save(buffer, "JPEG", quality=90)
                data, mime = buffer.getvalue(), "image/jpeg"
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_bytes(data, mime, cache=True)


class ReviewServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request: Any, client_address: Any) -> None:
        logger.exception("Unhandled error while serving %s", client_address[0])


def serve(app: App, port: int, open_browser: bool) -> None:
    ReviewHandler.app = app
    server = ReviewServer((app.host, port), ReviewHandler)
    url = f"http://{'127.0.0.1' if app.host in ('0.0.0.0', '') else app.host}:{port}"
    print(f"Photo review UI is running at {url}")
    logger.info("Serving %s on %s", app.library.root if app.library else "(no folder)", url)
    print("Press Ctrl+C to stop it.")
    if open_browser:
        threading.Timer(0.6, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
