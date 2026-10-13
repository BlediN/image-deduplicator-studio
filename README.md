# Image Deduplicator Studio

A local tool for cleaning up photo folders. It finds exact copies, near-duplicates and burst shots of the same moment, suggests the best shot in each group, and lets you review everything in your browser before anything is touched.

- **Never deletes anything.** Photos you mark are moved into `<folder>/_marked_for_delete`, and the last move can be undone from the UI.
- **Runs entirely on your computer.** The UI is served on `127.0.0.1`; photos are never uploaded anywhere.
- **Picks the best shot for you.** Ranking uses sharpness, exposure and resolution, and, when OpenCV is installed, faces: open eyes, smiles and whether everyone is in frame. The best shot in each group is marked **Keep** automatically.

## Requirements

- Python 3.10 or newer
- `numpy` and `pillow`
- Optional: `opencv-python-headless` for face-aware ranking (the face model is downloaded once on first use)
- Optional: `pillow-heif` for HEIC/HEIF photos

## Install

```bash
git clone https://github.com/BlediN/image-deduplicator-studio.git
cd image-deduplicator-studio
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

To skip the optional extras, install just `pip install numpy pillow`.

## Usage

Start the app. It opens in your browser:

```bash
python -m photo_review
```

(`python photo_review_app.py` does the same.)

1. **Choose folder...** (or type a path) and press **Scan**. Rescans are fast: unchanged photos are cached.
2. Go through the groups. Each one shows the photos side by side with the **★ Best shot** already marked **Keep**.
   - **Keep best, move rest**: keep the suggested shot, mark the others Move.
   - **Move all not kept**: mark everything you haven't kept as Move.
   - **Accept all exact copies**: one click for every exact-copy group you haven't touched.
   - Click a photo to open it full size. Use the arrow keys to browse, **K** to toggle Keep, **M** to toggle Move, and **Esc** to close.
3. Press **Move marked (N)**. The files go to `_marked_for_delete` inside the scanned folder. Check that folder, then delete it yourself when you're sure.

Your marks are saved as you go, so you can close the app and continue later.

### Command line

```bash
python -m photo_review --folder "D:\Photos" --scan            # scan only, print a summary
python -m photo_review --folder "D:\Photos" --scan --serve    # scan, then open the UI
python -m photo_review --help                                 # all options
```

Useful options:

| Option | What it does |
| --- | --- |
| `--force` | Re-analyze every photo, ignoring the cache |
| `--no-faces` | Skip face detection (faster) |
| `--session-seconds N` | Group shots taken within N seconds as the same moment (0 = off, default 15) |
| `--strong-threshold N` | How similar photos must look to be grouped (lower = stricter, default 24) |
| `--workers N` | Parallel scan workers |
| `--port N` / `--no-browser` | Server port (default 8765) / don't open a browser |
| `--debug` | Print detailed logs to the terminal and the log file |

The grouping thresholds can also be changed in the UI under **Settings**.

### Reviewing from a phone or tablet

Start the app with your PC's local IP address, then open `http://<that-ip>:8765` on a device on the same Wi-Fi:

```bash
python -m photo_review --host 192.168.1.20 --no-browser
```

Only do this on a network you trust. Anyone on it who has the address can open the page.

## Where data is stored

The scan index, thumbnails and your Keep/Move marks are stored per folder outside the photo folder:

- Windows: `%LOCALAPPDATA%\PhotoReview`
- macOS / Linux: `~/.local/share/PhotoReview`

The only thing written inside a photo folder is `_marked_for_delete`, and only when you move photos.

## Logs

The app writes a log file to `logs/photo_review.log` in the project folder, next to `README.md`. The folder is created on first run and is ignored by git, so logs are never pushed. The file rotates at 1 MB and keeps 3 old copies (`photo_review.log.1` to `.3`).

To watch the log live while the app runs (PowerShell):

```powershell
Get-Content logs\photo_review.log -Wait -Tail 20
```

| | Terminal | Log file |
| --- | --- | --- |
| Normal | Warnings and errors only | Startup info, scans, settings changes, moves and undos, unreadable files, failed requests, crashes with tracebacks |
| `--debug` | Everything | Everything, plus every request with its timing, cache hits, each moved file and tracebacks for unreadable files |

In the terminal, log lines are colored by level (grey debug, cyan info, yellow warnings, red errors), and tracebacks are indented under the entry they belong to. Colors switch off automatically when output is redirected. Set `NO_COLOR=1` to turn them off, or `FORCE_COLOR=1` to force them on. In the log file, a dashed line marks the start of each run.

If something goes wrong, run the app again with `--debug`, reproduce the problem, and check the log file.

## Project structure

```
photo_review/
├── __main__.py      # python -m photo_review
├── cli.py           # command-line arguments and entry point
├── config.py        # constants, optional dependencies, scan options
├── logs.py          # log file and terminal logging setup
├── storage.py       # JSON files, settings, per-folder Library
├── analysis.py      # per-image hashing, sharpness, faces, thumbnails; the scan
├── grouping.py      # matching photos into groups and ranking the best shot
├── moves.py         # moving marked photos and undo
├── server.py        # local HTTP server and JSON API
└── static/
    ├── index.html   # UI markup
    ├── app.css      # styles
    └── app.js       # UI logic (talks to the /api/* endpoints)
logs/                # log files (created on first run, git-ignored)
photo_review_app.py  # launcher kept for backwards compatibility
```

The backend is plain Python with no web framework. The front end is a single page with no build step: edit the files in `static/` and reload the browser.
