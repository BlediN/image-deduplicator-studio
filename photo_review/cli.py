"""Command-line entry point."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .config import PROJECT_DIR, ScanOptions, options_from_dict
from .storage import load_settings, save_settings, Library
from .analysis import ProgressFn, scan
from .logs import setup_logging
from .server import App, serve

logger = logging.getLogger(__name__)


def cli_progress() -> ProgressFn:
    last_phase = [""]

    def report(phase: str, done: int = 0, total: int = 0) -> None:
        if phase != last_phase[0]:
            print(phase + ("" if total else "..."))
            last_phase[0] = phase
        if total and (done == total or done % 50 == 0):
            print(f"  {done}/{total}")

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find and review exact/similar photos.")
    parser.add_argument("--folder", help="Folder to scan. Defaults to the last folder used, or the project folder.")
    parser.add_argument("--scan", action="store_true", help="Scan from the command line (add --serve to open the UI afterwards).")
    parser.add_argument("--serve", action="store_true", help="Start the browser UI (the default when --scan is not given).")
    parser.add_argument("--force", action="store_true", help="Re-analyze every image even when cached.")
    parser.add_argument("--host", default="127.0.0.1", help="Host for the local UI.")
    parser.add_argument("--port", type=int, default=8765, help="Port for the local UI.")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the browser automatically.")
    parser.add_argument("--strong-threshold", type=int, help="Visual distance for strong matches (0-192, default 24).")
    parser.add_argument("--burst-threshold", type=int, help="Visual distance for close-time burst matches (default 34).")
    parser.add_argument("--burst-seconds", type=int, help="Time window for looser burst grouping (default 120).")
    parser.add_argument("--session-seconds", type=int, help="Group shots this many seconds apart as the same moment (0 = off, default 15).")
    parser.add_argument("--session-threshold", type=int, help="Loose visual distance for same-moment shots (default 100).")
    parser.add_argument("--color-threshold", type=int, help="Maximum colour difference within a group (0-255, default 30).")
    parser.add_argument("--no-faces", action="store_true", help="Skip face detection and face-aware ranking.")
    parser.add_argument("--workers", type=int, help="Parallel workers for scanning.")
    parser.add_argument("--debug", action="store_true", help="Print detailed logs to the terminal and the log file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.debug)
    try:
        run(args)
    except Exception:
        logger.exception("Crashed")
        raise


def run(args: argparse.Namespace) -> None:
    settings = load_settings()
    options = options_from_dict(settings.get("options", {}), ScanOptions())
    overrides = {
        "strong_threshold": args.strong_threshold,
        "burst_threshold": args.burst_threshold,
        "burst_seconds": args.burst_seconds,
        "session_seconds": args.session_seconds,
        "session_threshold": args.session_threshold,
        "color_threshold": args.color_threshold,
        "workers": args.workers,
    }
    if args.no_faces:
        overrides.update(detect_faces=False, prefer_faces=False)
    options = options_from_dict({key: value for key, value in overrides.items() if value is not None}, options)

    folder_text = args.folder or settings.get("folder") or str(PROJECT_DIR)
    folder = Path(folder_text).expanduser()
    if args.folder and not folder.is_dir():
        sys.exit(f"Folder not found: {folder}")
    logger.debug("Folder %s | options %s", folder, asdict(options))

    if args.scan:
        library = Library(folder)
        save_settings(library.root, options)
        start = time.time()
        payload = scan(library, options, force=args.force, progress=cli_progress())
        summary = payload["grouping"]["summary"]
        print(
            f"Done. Found {summary['group_count']} review groups "
            f"({summary['exact_group_count']} exact, {summary['similar_group_count']} similar, "
            f"{summary['moment_group_count']} same moment, "
            f"{summary['people_group_count']} with people)."
        )
        if not payload["faces_available"]:
            print(f"Face ranking off: {payload['faces_reason']}")
        if payload["errors"]:
            print(f"{len(payload['errors'])} files could not be read.")
        print(f"Elapsed: {time.time() - start:.1f}s")
        logger.info(
            "Command-line scan of %s: %d photos, %d groups, %d unreadable, %.1fs",
            library.root, summary["image_count"], summary["group_count"], len(payload["errors"]), time.time() - start,
        )
        if not args.serve:
            return

    serve(App(folder, options, args.host), args.port, open_browser=not args.no_browser)
