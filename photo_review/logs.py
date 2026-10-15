"""Logging setup: a rotating log file, plus the terminal for warnings (or everything with --debug)."""

from __future__ import annotations

import logging
import os
import platform
import sys
import threading
from logging.handlers import RotatingFileHandler
from typing import TextIO

from .config import LOG_FILE, cv2

FORMAT = "%(asctime)s %(levelname)-7s %(name)s [%(threadName)s] %(message)s"
RUN_SEPARATOR = "-" * 100

RESET = "\033[0m"
DIM = "\033[2m"
LEVEL_STYLES = {
    logging.DEBUG: ("DEBUG", "\033[90m", DIM),  # grey label, dim message
    logging.INFO: ("INFO", "\033[36m", ""),  # cyan label
    logging.WARNING: ("WARN", "\033[33;1m", "\033[33m"),  # yellow
    logging.ERROR: ("ERROR", "\033[31;1m", "\033[31m"),  # red
    logging.CRITICAL: ("CRIT", "\033[97;41;1m", "\033[31;1m"),  # white on red
}


class ConsoleFormatter(logging.Formatter):
    """`21:42:49  WARN   moves     message`, with extra lines (tracebacks) indented under the message."""

    def __init__(self, color: bool) -> None:
        super().__init__()
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        label, label_style, message_style = LEVEL_STYLES.get(record.levelno, (record.levelname, "", ""))
        module = record.name.removeprefix("photo_review").lstrip(".") or "app"
        time = self.formatTime(record, "%H:%M:%S")
        lines = record.getMessage().split("\n")
        if record.exc_info:
            lines += self.formatException(record.exc_info).split("\n")
        if record.stack_info:
            lines += self.formatStack(record.stack_info).split("\n")

        prefix_width = len(f"{time}  {label:<5}  {module:<9} ")
        if not self.color:
            prefix = f"{time}  {label:<5}  {module:<9} "
            return "\n".join([prefix + lines[0]] + [" " * prefix_width + line for line in lines[1:]])

        prefix = f"{DIM}{time}{RESET}  {label_style}{label:<5}{RESET}  {DIM}{module:<9}{RESET} "
        first = f"{prefix}{message_style}{lines[0]}{RESET}"
        rest = [f"{' ' * (prefix_width - 2)}{DIM}|{RESET} {message_style}{line}{RESET}" for line in lines[1:]]
        return "\n".join([first] + rest)


def use_color(stream: TextIO) -> bool:
    """Color when writing to a real terminal, unless NO_COLOR is set. FORCE_COLOR turns it on regardless."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if not hasattr(stream, "isatty") or not stream.isatty():
        return False
    if sys.platform != "win32":
        return True
    # Windows 10+ consoles understand ANSI colors once virtual terminal processing is switched on.
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-12)  # STD_ERROR_HANDLE
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return bool(os.environ.get("TERM") or os.environ.get("WT_SESSION"))
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    except Exception:
        return False


def setup_logging(debug: bool = False) -> None:
    """Log INFO and up to the file (DEBUG with --debug), and WARNING and up to the terminal (DEBUG with --debug)."""
    root = logging.getLogger("photo_review")
    root.setLevel(logging.DEBUG if debug else logging.INFO)
    root.propagate = False
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.DEBUG if debug else logging.WARNING)
    console.setFormatter(ConsoleFormatter(color=use_color(sys.stderr)))
    root.addHandler(console)

    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    except OSError as exc:
        root.warning("Could not open the log file %s (%s). Logging to the terminal only.", LOG_FILE, exc)
    else:
        file_handler.setFormatter(logging.Formatter(FORMAT))
        if file_handler.stream.tell():
            file_handler.stream.write("\n" + RUN_SEPARATOR + "\n")  # mark where each run starts
        root.addHandler(file_handler)

    def log_thread_crash(args: threading.ExceptHookArgs) -> None:
        if issubclass(args.exc_type, SystemExit):
            return
        thread = args.thread.name if args.thread else "unknown"
        root.critical("Uncaught error in thread %s", thread, exc_info=(args.exc_type, args.exc_value, args.exc_traceback))

    threading.excepthook = log_thread_crash

    root.info(
        "Starting (debug=%s) | Python %s | %s | OpenCV %s | log file %s",
        debug,
        platform.python_version(),
        platform.platform(),
        getattr(cv2, "__version__", "not installed"),
        LOG_FILE,
    )
