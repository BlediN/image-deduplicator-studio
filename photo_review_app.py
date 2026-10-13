#!/usr/bin/env python3
"""
Local duplicate/similar photo review tool.

Run:
    python photo_review_app.py                              # open the browser UI
    python photo_review_app.py --folder "D:\\Photos" --scan  # scan from the command line only

Pick a folder and start scans from the UI. The app never permanently deletes
photos: files you mark are moved into <folder>/_marked_for_delete, and the
last move can be undone from the UI.

The index, thumbnails and your keep/move marks are stored per folder under
%LOCALAPPDATA%\\PhotoReview, so nothing extra is written into the photo folder.

Optional extras:
    pip install opencv-python-headless   # face/posture-aware best-shot ranking
    pip install pillow-heif              # HEIC/HEIF photos
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import mimetypes
import os
import posixpath
import re
import secrets
import shutil
import sys
import threading
import time
import urllib.parse
import urllib.request
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import ExifTags, Image, ImageFile, ImageFilter, ImageOps

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

SCRIPT_DIR = Path(__file__).resolve().parent
APP_HOME = Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".local" / "share") / "PhotoReview"
SETTINGS_FILE = APP_HOME / "settings.json"
MODEL_DIR = APP_HOME / "models"
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


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="review-token" content="__TOKEN__">
  <title>Photo Review</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7f4;
      --panel: #ffffff;
      --ink: #1d2520;
      --muted: #647069;
      --line: #dbe2dc;
      --accent: #176b65;
      --accent-2: #8a4b12;
      --danger: #b42318;
      --danger-soft: #fff1ef;
      --keep: #11603a;
      --keep-soft: #eaf7f0;
      --warn-soft: #fff8e6;
      --shadow: 0 8px 24px rgba(27, 40, 32, 0.08);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    * { box-sizing: border-box; }
    [hidden] { display: none !important; }

    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
    }

    button,
    input,
    select {
      font: inherit;
    }

    button {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      min-height: 38px;
      border-radius: 6px;
      padding: 0 12px;
      cursor: pointer;
    }

    button:hover { border-color: #9fb2aa; }
    button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
    button.danger { background: var(--danger); border-color: var(--danger); color: #fff; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }

    header {
      position: sticky;
      top: 0;
      z-index: 20;
      background: rgba(245, 247, 244, 0.96);
      backdrop-filter: blur(10px);
      border-bottom: 1px solid var(--line);
    }

    .topbar {
      display: grid;
      gap: 12px;
      grid-template-columns: minmax(240px, 1fr) auto;
      align-items: center;
      max-width: 1500px;
      margin: 0 auto;
      padding: 14px 18px 10px;
    }

    h1 {
      margin: 0;
      font-size: 22px;
      letter-spacing: 0;
    }

    .sub {
      margin-top: 3px;
      color: var(--muted);
      font-size: 13px;
      overflow-wrap: anywhere;
    }

    .actions {
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
      justify-content: flex-end;
    }

    .folder-bar {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      max-width: 1500px;
      margin: 0 auto;
      padding: 0 18px 12px;
    }

    .folder-bar input[type="text"] {
      flex: 1 1 360px;
      min-width: 0;
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 10px;
      background: #fff;
      color: var(--ink);
    }

    .inline-check {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 13px;
      color: var(--muted);
      cursor: pointer;
    }

    .progress {
      max-width: 1500px;
      margin: 0 auto;
      padding: 0 18px 12px;
    }

    .progress-track {
      height: 8px;
      background: #e1e8e3;
      border-radius: 999px;
      overflow: hidden;
    }

    .progress-fill {
      height: 100%;
      width: 0;
      background: var(--accent);
      transition: width 200ms ease;
    }

    .progress-fill.indeterminate {
      width: 30%;
      animation: slide 1.2s infinite ease-in-out;
    }

    @keyframes slide {
      from { margin-left: -30%; }
      to { margin-left: 100%; }
    }

    .progress-text {
      margin-top: 6px;
      color: var(--muted);
      font-size: 13px;
    }

    .settings {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
      gap: 12px;
      max-width: 1500px;
      margin: 0 auto 12px;
      padding: 14px 18px;
      background: #fff;
      border-top: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
    }

    .field {
      display: grid;
      gap: 4px;
      font-size: 13px;
      font-weight: 600;
    }

    .field input[type="number"] {
      min-height: 36px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 10px;
    }

    .field small {
      color: var(--muted);
      font-weight: 400;
      line-height: 1.35;
    }

    .field.check-field {
      grid-template-columns: auto 1fr;
      align-items: start;
      column-gap: 8px;
    }

    .field.check-field small { grid-column: 2; }

    .settings-actions {
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
      grid-column: 1 / -1;
      color: var(--muted);
      font-size: 13px;
    }

    .stats {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      max-width: 1500px;
      margin: 0 auto;
      padding: 0 18px 14px;
    }

    .pill {
      display: inline-flex;
      align-items: center;
      min-height: 30px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #fff;
      padding: 0 10px;
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }

    main {
      max-width: 1500px;
      margin: 0 auto;
      padding: 18px;
    }

    .notice {
      border: 1px solid #f1d9a6;
      background: var(--warn-soft);
      border-radius: 8px;
      padding: 10px 14px;
      margin-bottom: 12px;
      font-size: 13px;
      line-height: 1.45;
    }

    .notice summary { cursor: pointer; font-weight: 600; }
    .notice ul { margin: 8px 0 0; padding-left: 18px; }

    .controls {
      display: grid;
      grid-template-columns: minmax(180px, 1fr) 190px 210px auto;
      gap: 10px;
      align-items: center;
      margin-bottom: 12px;
    }

    .controls input,
    .controls select {
      width: 100%;
      min-height: 40px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 10px;
      background: #fff;
      color: var(--ink);
    }

    .pager {
      display: flex;
      gap: 10px;
      align-items: center;
      justify-content: center;
      margin: 6px 0 14px;
      color: var(--muted);
      font-size: 13px;
    }

    .group {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      margin-bottom: 16px;
      overflow: clip;
    }

    .group-head {
      display: grid;
      grid-template-columns: minmax(240px, 1fr) auto;
      gap: 12px;
      align-items: center;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfb;
    }

    .group-title,
    .tag-row {
      display: flex;
      gap: 6px;
      align-items: center;
      flex-wrap: wrap;
      font-weight: 700;
    }

    .tag-row { min-height: 24px; }

    .tag {
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      border-radius: 999px;
      padding: 0 8px;
      font-size: 12px;
      font-weight: 700;
      background: #edf4f3;
      color: var(--accent);
      white-space: nowrap;
    }

    .tag.exact { background: #fff3e6; color: var(--accent-2); }
    .tag.delete { background: var(--danger-soft); color: var(--danger); }
    .tag.keep { background: var(--keep-soft); color: var(--keep); }
    .tag.best { background: #fff6d6; color: #7a5a00; }
    .tag.people { background: #eef0ff; color: #3b3f9e; }
    .tag.moment { background: #f3ecfb; color: #6a3d9a; }

    .group-meta {
      margin-top: 4px;
      color: var(--muted);
      font-size: 13px;
    }

    .group-tools {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      justify-content: flex-end;
    }

    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 12px;
      padding: 14px;
    }

    .photo {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      overflow: clip;
      min-width: 0;
    }

    .photo.marked {
      border-color: #f0a39b;
      background: var(--danger-soft);
    }

    .photo.keep {
      border-color: #a8d8bd;
      background: var(--keep-soft);
    }

    .thumb {
      display: block;
      width: 100%;
      aspect-ratio: 4 / 3;
      object-fit: cover;
      background: #e7ece8;
      cursor: zoom-in;
    }

    .photo-body {
      display: grid;
      gap: 7px;
      padding: 10px;
    }

    .name {
      font-size: 13px;
      font-weight: 700;
      overflow-wrap: anywhere;
      line-height: 1.25;
    }

    .detail {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
      overflow-wrap: anywhere;
    }

    .notes {
      display: flex;
      flex-wrap: wrap;
      gap: 4px;
    }

    .note {
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 11px;
      font-weight: 600;
    }

    .note.good { background: var(--keep-soft); color: var(--keep); }
    .note.bad { background: var(--danger-soft); color: var(--danger); }

    .choice-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }

    .toggle { font-size: 13px; min-height: 34px; }
    .toggle.on-keep { background: var(--keep); border-color: var(--keep); color: #fff; font-weight: 700; }
    .toggle.on-move { background: var(--danger); border-color: var(--danger); color: #fff; font-weight: 700; }

    .open-link {
      color: var(--accent);
      font-size: 12px;
      text-decoration: none;
    }

    .empty {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 22px;
      color: var(--muted);
      line-height: 1.5;
    }

    .lightbox {
      position: fixed;
      inset: 0;
      z-index: 50;
      display: flex;
      flex-direction: column;
      background: rgba(12, 16, 14, 0.95);
    }

    .lb-bar {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      padding: 10px 14px;
      color: #fff;
    }

    .lb-text { flex: 1 1 280px; min-width: 0; }
    .lb-title { font-weight: 700; overflow-wrap: anywhere; }
    .lb-info { margin-top: 2px; color: #c9d3cd; font-size: 12px; }
    .lb-bar .notes { margin-top: 4px; }

    .lb-stage {
      flex: 1;
      min-height: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 0 12px 12px;
    }

    .lb-stage img {
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
    }

    .toast {
      position: fixed;
      right: 18px;
      bottom: 18px;
      max-width: min(460px, calc(100vw - 36px));
      background: #1d2520;
      color: #fff;
      border-radius: 8px;
      padding: 12px 14px;
      box-shadow: var(--shadow);
      opacity: 0;
      transform: translateY(12px);
      pointer-events: none;
      transition: opacity 160ms ease, transform 160ms ease;
      z-index: 60;
    }

    .toast.show {
      opacity: 1;
      transform: translateY(0);
    }

    @media (max-width: 860px) {
      .topbar,
      .group-head,
      .controls {
        grid-template-columns: 1fr;
      }

      .actions,
      .group-tools {
        justify-content: stretch;
      }

      .actions button,
      .group-tools button {
        flex: 1 1 150px;
      }
    }
  </style>
</head>
<body>
  <header>
    <div class="topbar">
      <div>
        <h1>Photo Review</h1>
        <div class="sub" id="subtitle">Loading...</div>
      </div>
      <div class="actions">
        <button type="button" id="undoBtn" disabled title="Put the files from the last move back where they were">Undo last move</button>
        <button type="button" id="moveBtn" class="danger" disabled title="Move files marked Move into _marked_for_delete">Move marked (0)</button>
      </div>
    </div>
    <div class="folder-bar">
      <input id="folderInput" type="text" placeholder="Folder to scan, e.g. C:\Users\you\Pictures" spellcheck="false" aria-label="Folder to scan">
      <button type="button" id="browseBtn">Choose folder...</button>
      <button type="button" id="scanBtn" class="primary">Scan</button>
      <label class="inline-check"><input type="checkbox" id="forceScan"> Re-analyze every photo</label>
      <button type="button" id="settingsBtn" aria-expanded="false">Settings</button>
    </div>
    <div class="progress" id="progress" hidden>
      <div class="progress-track"><div class="progress-fill" id="progressFill"></div></div>
      <div class="progress-text" id="progressText"></div>
    </div>
    <div class="settings" id="settings" hidden>
      <label class="field">
        <span>Strong match distance</span>
        <input type="number" data-option="strong_threshold" min="0" max="192">
        <small>0-192. Photos this close are grouped regardless of time. Higher = looser. Default 24.</small>
      </label>
      <label class="field">
        <span>Burst match distance</span>
        <input type="number" data-option="burst_threshold" min="0" max="192">
        <small>Looser limit used only for photos taken within the burst window. Default 34.</small>
      </label>
      <label class="field">
        <span>Burst window (seconds)</span>
        <input type="number" data-option="burst_seconds" min="0" max="86400">
        <small>Uses the camera's EXIF capture time; photos without one never use the burst rule. Default 120.</small>
      </label>
      <label class="field">
        <span>Same-moment window (seconds)</span>
        <input type="number" data-option="session_seconds" min="0" max="3600">
        <small>Groups shots taken this close together even when pose, framing or lights changed, so you can pick the best pose. 0 turns it off. Default 15.</small>
      </label>
      <label class="field">
        <span>Same-moment distance</span>
        <input type="number" data-option="session_threshold" min="0" max="192">
        <small>Loose visual limit for same-moment shots (unrelated photos average about 96). Default 100.</small>
      </label>
      <label class="field">
        <span>Colour difference limit</span>
        <input type="number" data-option="color_threshold" min="0" max="255">
        <small>0-255. Lower splits groups whose colours differ. 255 turns the check off. Default 30.</small>
      </label>
      <label class="field check-field">
        <input type="checkbox" data-option="prefer_faces">
        <span>Rank by faces and posture</span>
        <small>In groups with people, prefer shots where everyone is visible, facing the camera, eyes open, smiling and in focus.</small>
      </label>
      <label class="field check-field">
        <input type="checkbox" data-option="match_rotations">
        <span>Match rotated copies</span>
        <small>Also groups copies that were rotated by 90/180/270 degrees.</small>
      </label>
      <label class="field check-field">
        <input type="checkbox" data-option="detect_faces">
        <span>Detect faces while scanning</span>
        <small>Needed for face/posture ranking. Turning it back on requires a rescan.</small>
      </label>
      <div class="settings-actions">
        <button type="button" id="applySettingsBtn" class="primary">Apply and regroup</button>
        <span>Regrouping uses the existing scan, so no rescan is needed.</span>
      </div>
    </div>
    <div class="stats" id="stats"></div>
  </header>

  <main>
    <div id="notices"></div>
    <div class="controls">
      <input id="searchBox" type="search" placeholder="Filter by filename or path">
      <select id="typeFilter" aria-label="Group type">
        <option value="all">All groups</option>
        <option value="exact">Exact copies only</option>
        <option value="similar">Similar only</option>
        <option value="moment">Same moment only</option>
        <option value="people">Groups with people</option>
      </select>
      <select id="decisionFilter" aria-label="Review status">
        <option value="all">Any review status</option>
        <option value="undecided">Not reviewed yet</option>
        <option value="progress">Partly reviewed</option>
        <option value="done">Fully reviewed</option>
        <option value="nokeep">Every photo marked Move</option>
      </select>
      <button type="button" id="acceptExactBtn" title="For exact-copy groups you have not touched: keep one, move the identical copies">Accept all exact copies</button>
    </div>
    <div class="pager" id="pagerTop"></div>
    <div id="groups"></div>
    <div class="pager" id="pagerBottom"></div>
  </main>

  <div class="lightbox" id="lightbox" hidden>
    <div class="lb-bar">
      <div class="lb-text">
        <div class="lb-title" id="lbTitle"></div>
        <div class="lb-info" id="lbInfo"></div>
        <div class="notes" id="lbNotes"></div>
      </div>
      <button type="button" id="lbPrev" title="Previous (Left arrow)">&lsaquo; Prev</button>
      <button type="button" id="lbKeep" class="toggle" title="Keep (K)">&check; Keep</button>
      <button type="button" id="lbMove" class="toggle" title="Move (M)">&rarr; Move</button>
      <button type="button" id="lbNext" title="Next (Right arrow)">Next &rsaquo;</button>
      <button type="button" id="lbClose" title="Close (Esc)">Close</button>
    </div>
    <div class="lb-stage" id="lbStage"><img id="lbImg" alt=""></div>
  </div>

  <div class="toast" id="toast"></div>

  <script>
    const TOKEN = document.querySelector('meta[name="review-token"]').content;
    const PAGE_SIZE = 20;
    let state = null;
    let statusMap = new Map();
    let groupById = new Map();
    let fileLookup = new Map();
    let page = 0;
    let saveTimer = null;
    let pollTimer = null;
    let lb = null;

    const $ = (id) => document.getElementById(id);

    function toast(message) {
      const el = $("toast");
      el.textContent = message;
      el.classList.add("show");
      window.clearTimeout(toast.timer);
      toast.timer = window.setTimeout(() => el.classList.remove("show"), 3600);
    }

    async function api(path, body) {
      const options = { headers: { "X-Review-Token": TOKEN } };
      if (body !== undefined) {
        options.method = "POST";
        options.headers["Content-Type"] = "application/json";
        options.body = JSON.stringify(body);
      }
      const response = await fetch(path, options);
      let data = null;
      try {
        data = await response.json();
      } catch (err) {
        data = null;
      }
      if (!response.ok) throw new Error((data && data.error) || `Request failed (${response.status})`);
      return data;
    }

    function formatBytes(bytes) {
      const units = ["B", "KB", "MB", "GB"];
      let value = Number(bytes || 0);
      let unit = 0;
      while (value >= 1024 && unit < units.length - 1) {
        value /= 1024;
        unit += 1;
      }
      return `${value.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
    }

    function escapeHtml(text) {
      return String(text)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    const thumbUrl = (file) => `/thumb/${file.id}.jpg?v=${file.v}`;
    const imageUrl = (file) => `/image/${file.id}?v=${file.v}`;

    function groupProgress(group) {
      let keep = 0;
      let move = 0;
      group.files.forEach((file) => {
        const status = statusMap.get(file.id);
        if (status === "keep") keep += 1;
        if (status === "move") move += 1;
      });
      return { keep, move, undecided: group.files.length - keep - move };
    }

    // Untouched = no marks except the automatic Keep on the best shot.
    function isUntouched(group) {
      return group.files.every((file) => !statusMap.has(file.id) || (file.id === group.suggested_id && statusMap.get(file.id) === "keep"));
    }

    function markCounts() {
      let keep = 0;
      let move = 0;
      statusMap.forEach((status, id) => {
        if (!fileLookup.has(id)) return;
        if (status === "keep") keep += 1;
        if (status === "move") move += 1;
      });
      return { keep, move };
    }

    function applyState(data) {
      state = data;
      statusMap = new Map(Object.entries(data.decisions || {}));
      groupById = new Map();
      fileLookup = new Map();
      data.groups.forEach((group, index) => {
        group.number = index + 1;
        groupById.set(group.id, group);
        group.files.forEach((file) => fileLookup.set(file.id, { file, group }));
      });
      if (document.activeElement !== $("folderInput")) $("folderInput").value = data.folder || "";
      fillSettings(data.options || {});
      if (lb && !groupById.has(lb.group.id)) closeLightbox();
      renderNotices();
      render();
      handleJob(data.job);
    }

    async function loadState() {
      applyState(await api("/api/state"));
    }

    function fillSettings(options) {
      document.querySelectorAll("[data-option]").forEach((input) => {
        const value = options[input.dataset.option];
        if (input.type === "checkbox") input.checked = Boolean(value);
        else if (value !== undefined) input.value = value;
      });
    }

    function readSettings() {
      const options = {};
      document.querySelectorAll("[data-option]").forEach((input) => {
        options[input.dataset.option] = input.type === "checkbox" ? input.checked : Number(input.value);
      });
      return options;
    }

    function updateStats() {
      if (!state) return;
      const summary = state.summary;
      const counts = markCounts();
      $("subtitle").textContent = state.scanned
        ? `${summary.group_count} groups to review in ${state.folder}`
        : "Choose a folder and press Scan to find duplicate and similar photos.";
      const stats = state.scanned ? [
        `${summary.image_count} photos scanned`,
        `${summary.group_count} groups`,
        `${summary.exact_group_count} exact`,
        `${summary.similar_group_count} similar`,
        `${summary.moment_group_count} same moment`,
        `${summary.people_group_count} with people`,
        `${counts.keep} marked keep`,
        `${counts.move} marked move`,
        `Moves go to: ${state.move_dir}`,
      ] : [];
      $("stats").innerHTML = stats.map((item) => `<span class="pill">${escapeHtml(item)}</span>`).join("");
      $("moveBtn").textContent = `Move marked (${counts.move})`;
      $("moveBtn").disabled = counts.move === 0;
      $("undoBtn").disabled = !state.can_undo;
    }

    function renderNotices() {
      const parts = [];
      if (state.scanned && !state.summary.faces_available) {
        parts.push(`<div class="notice">Face and posture ranking is off for this scan. ${escapeHtml(state.summary.faces_reason || "")}</div>`);
      }
      if (state.scanned && !state.summary.heif_available) {
        parts.push('<div class="notice">HEIC/HEIF photos are not scanned. Run <code>pip install pillow-heif</code> and rescan to include them.</div>');
      }
      if (state.errors && state.errors.length) {
        const items = state.errors.slice(0, 200).map((err) => `<li>${escapeHtml(err.rel_path)}: ${escapeHtml(err.error)}</li>`).join("");
        parts.push(`<details class="notice"><summary>${state.errors.length} files could not be read</summary><ul>${items}</ul></details>`);
      }
      $("notices").innerHTML = parts.join("");
    }

    function filteredGroups() {
      const query = $("searchBox").value.trim().toLowerCase();
      const type = $("typeFilter").value;
      const decision = $("decisionFilter").value;
      return state.groups.filter((group) => {
        if (query && !group.search_text.toLowerCase().includes(query)) return false;
        if (type === "exact" && group.kind !== "exact") return false;
        if (type === "similar" && group.kind !== "similar") return false;
        if (type === "moment" && group.kind !== "moment") return false;
        if (type === "people" && !group.has_faces) return false;
        const progress = groupProgress(group);
        const untouched = isUntouched(group);
        if (decision === "undecided" && !untouched) return false;
        if (decision === "progress" && (untouched || progress.undecided === 0)) return false;
        if (decision === "done" && progress.undecided > 0) return false;
        if (decision === "nokeep" && progress.move !== group.files.length) return false;
        return true;
      });
    }

    function render() {
      updateStats();
      const container = $("groups");
      if (!state.scanned) {
        container.innerHTML = '<div class="empty">No scan yet for this folder. Use <b>Choose folder...</b> or type a path, then press <b>Scan</b>.</div>';
        renderPager(0, 1);
        return;
      }
      const list = filteredGroups();
      const pages = Math.max(1, Math.ceil(list.length / PAGE_SIZE));
      page = Math.min(page, pages - 1);
      const slice = list.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
      if (!state.groups.length) {
        container.innerHTML = '<div class="empty">No duplicate or similar-photo groups were found.</div>';
      } else if (!slice.length) {
        container.innerHTML = '<div class="empty">No groups match the current filters.</div>';
      } else {
        container.innerHTML = slice.map(groupHtml).join("");
      }
      renderPager(list.length, pages);
    }

    function renderPager(total, pages) {
      const html = total > PAGE_SIZE ? `
        <button type="button" data-page="prev" ${page === 0 ? "disabled" : ""}>&lsaquo; Prev</button>
        <span>Page ${page + 1} of ${pages} &middot; ${total} groups</span>
        <button type="button" data-page="next" ${page >= pages - 1 ? "disabled" : ""}>Next &rsaquo;</button>` : "";
      $("pagerTop").innerHTML = html;
      $("pagerBottom").innerHTML = html;
    }

    function groupHtml(group) {
      const progress = groupProgress(group);
      const tags = [
        { exact: '<span class="tag exact">Exact copies</span>', similar: '<span class="tag">Similar</span>', moment: '<span class="tag moment">Same moment</span>' }[group.kind],
        group.has_faces ? '<span class="tag people">People</span>' : "",
        progress.move === group.files.length ? '<span class="tag delete">Nothing kept</span>' : "",
      ].join("");
      const meta = `${group.files.length} photos | ${progress.keep} keep | ${progress.move} move | ${progress.undecided} not decided`;
      return `
        <section class="group" data-group-id="${group.id}">
          <div class="group-head">
            <div>
              <div class="group-title"><span>Group ${String(group.number).padStart(3, "0")}</span>${tags}</div>
              <div class="group-meta">${escapeHtml(meta)}</div>
            </div>
            <div class="group-tools">
              <button type="button" data-action="suggest" data-group-id="${group.id}" title="Keep the best shot and move the others">Keep best, move rest</button>
              <button type="button" data-action="move-rest" data-group-id="${group.id}" ${progress.keep ? "" : "disabled"} title="Move every photo not marked Keep">Move all not kept</button>
              <button type="button" data-action="keep-all" data-group-id="${group.id}">Keep all</button>
              <button type="button" data-action="clear" data-group-id="${group.id}">Clear</button>
            </div>
          </div>
          <div class="grid">${group.files.map((file) => cardHtml(file, group)).join("")}</div>
        </section>`;
    }

    function fileDetails(file, group) {
      const first = [`${file.width} x ${file.height}`, formatBytes(file.size), file.taken].filter(Boolean).join(" | ");
      const second = [
        group.has_faces ? `score ${file.rank.toFixed(1)} (photo ${file.score.toFixed(1)})` : `score ${file.score.toFixed(1)}`,
        `sharp ${file.sharpness.toFixed(0)}`,
        group.kind === "exact" ? "identical" : `match ${file.similarity}%`,
      ].join(" | ");
      return [first, second];
    }

    function notesHtml(file) {
      return file.notes.map((note) => `<span class="note ${note.kind}">${escapeHtml(note.text)}</span>`).join("");
    }

    function cardHtml(file, group) {
      const status = statusMap.get(file.id);
      const cls = status === "keep" ? "photo keep" : (status === "move" ? "photo marked" : "photo");
      const tags = [
        file.id === group.suggested_id ? '<span class="tag best">&starf; Best shot</span>' : "",
        status === "keep" ? '<span class="tag keep">Keep</span>' : "",
        status === "move" ? '<span class="tag delete">Move</span>' : "",
      ].join("");
      const [first, second] = fileDetails(file, group);
      const notes = notesHtml(file);
      return `
        <article class="${cls}">
          <img class="thumb" data-action="view" data-id="${file.id}" src="${thumbUrl(file)}" alt="${escapeHtml(file.name)}" loading="lazy">
          <div class="photo-body">
            <div class="tag-row">${tags}</div>
            <div class="name">${escapeHtml(file.name)}</div>
            <div class="detail">${escapeHtml(first)}</div>
            <div class="detail">${escapeHtml(second)}</div>
            ${notes ? `<div class="notes">${notes}</div>` : ""}
            <div class="detail">${escapeHtml(file.rel_path)}</div>
            <div class="choice-row">
              <button type="button" class="toggle ${status === "keep" ? "on-keep" : ""}" data-action="keep" data-id="${file.id}">&check; Keep</button>
              <button type="button" class="toggle ${status === "move" ? "on-move" : ""}" data-action="move" data-id="${file.id}">&rarr; Move</button>
            </div>
            <a class="open-link" href="${imageUrl(file)}" target="_blank" rel="noreferrer">Open full size</a>
          </div>
        </article>`;
    }

    function refreshGroup(group) {
      const el = document.querySelector(`section.group[data-group-id="${group.id}"]`);
      if (el) el.outerHTML = groupHtml(group);
    }

    function changed(groups) {
      groups.forEach(refreshGroup);
      updateStats();
      scheduleSave();
      if (lb) renderLightbox();
    }

    function toggleStatus(id, value) {
      if (statusMap.get(id) === value) statusMap.delete(id);
      else statusMap.set(id, value);
      const entry = fileLookup.get(id);
      changed(entry ? [entry.group] : []);
    }

    function groupAction(group, action) {
      group.files.forEach((file) => {
        if (action === "suggest") statusMap.set(file.id, file.id === group.suggested_id ? "keep" : "move");
        else if (action === "keep-all") statusMap.set(file.id, "keep");
        else if (action === "move-rest" && statusMap.get(file.id) !== "keep") statusMap.set(file.id, "move");
        else if (action === "clear") statusMap.delete(file.id);
      });
      changed([group]);
    }

    function scheduleSave() {
      window.clearTimeout(saveTimer);
      saveTimer = window.setTimeout(() => {
        saveTimer = null;
        saveDecisions().catch((err) => toast(`Could not save marks: ${err.message}`));
      }, 500);
    }

    async function saveDecisions() {
      if (!state || !state.folder) return;
      await api("/api/decisions", { folder: state.folder, status: Object.fromEntries(statusMap) });
    }

    async function flushSave() {
      if (!saveTimer) return;
      window.clearTimeout(saveTimer);
      saveTimer = null;
      await saveDecisions();
    }

    function handleJob(job) {
      const running = Boolean(job && job.running);
      $("progress").hidden = !running;
      $("scanBtn").disabled = running;
      $("browseBtn").disabled = running;
      $("moveBtn").disabled = running || markCounts().move === 0;
      $("undoBtn").disabled = running || !(state && state.can_undo);
      if (!running) return;
      const fill = $("progressFill");
      if (job.total) {
        fill.classList.remove("indeterminate");
        fill.style.width = `${Math.round((job.done / job.total) * 100)}%`;
        $("progressText").textContent = `${job.phase}: ${job.done} / ${job.total}`;
      } else {
        fill.classList.add("indeterminate");
        fill.style.width = "";
        $("progressText").textContent = `${job.phase}...`;
      }
      if (!pollTimer) pollTimer = window.setTimeout(pollJob, 800);
    }

    async function pollJob() {
      pollTimer = null;
      let job;
      try {
        job = await api("/api/job");
      } catch (err) {
        pollTimer = window.setTimeout(pollJob, 2000);
        return;
      }
      if (job.running) {
        handleJob(job);
        return;
      }
      handleJob(job);
      if (job.error) toast(`Scan failed: ${job.error}`);
      else if (job.message) toast(job.message);
      await loadState();
    }

    async function startScan() {
      const folder = $("folderInput").value.trim();
      if (!folder) {
        toast("Choose or type a folder first.");
        return;
      }
      await flushSave();
      const job = await api("/api/scan", { folder, force: $("forceScan").checked });
      $("forceScan").checked = false;
      handleJob(job);
    }

    async function browse() {
      $("browseBtn").disabled = true;
      try {
        toast("A folder picker opened. It may be behind the browser window.");
        const result = await api("/api/pick-folder", { initial: $("folderInput").value.trim() });
        if (result.path) {
          $("folderInput").value = result.path;
          await startScan();
        }
      } finally {
        $("browseBtn").disabled = false;
      }
    }

    async function applySettings() {
      await flushSave();
      applyState(await api("/api/settings", { options: readSettings() }));
      toast("Settings applied. Groups were rebuilt.");
    }

    function acceptExact() {
      const groups = state.groups.filter((group) => group.kind === "exact" && isUntouched(group));
      groups.forEach((group) => group.files.forEach((file) => statusMap.set(file.id, file.id === group.suggested_id ? "keep" : "move")));
      render();
      scheduleSave();
      toast(groups.length ? `Marked ${groups.length} exact-copy groups: one kept, copies to move.` : "No untouched exact-copy groups left.");
    }

    async function moveMarked() {
      const ids = Array.from(statusMap.entries()).filter(([id, value]) => value === "move" && fileLookup.has(id)).map(([id]) => id);
      if (!ids.length) return;
      const nothingKept = state.groups.filter((group) => group.files.every((file) => statusMap.get(file.id) === "move")).length;
      let message = `Move ${ids.length} photos into\n${state.move_dir}?\n\nNothing is deleted, and you can undo this move.`;
      if (nothingKept) message += `\n\nWarning: in ${nothingKept} group(s) every photo is marked Move, so none of them stays in the folder.`;
      if (!window.confirm(message)) return;
      await flushSave();
      const result = await api("/api/move", { ids });
      toast(`Moved ${result.moved.length} photos.${result.errors.length ? ` ${result.errors.length} could not be moved: ${result.errors[0].error}` : ""}`);
      await loadState();
    }

    async function undoMove() {
      await flushSave();
      const result = await api("/api/undo", {});
      toast(`Restored ${result.restored.length} photos.${result.errors.length ? ` ${result.errors.length} could not be restored: ${result.errors[0].error}` : ""}`);
      await loadState();
    }

    function openLightbox(id) {
      const entry = fileLookup.get(id);
      if (!entry) return;
      lb = { group: entry.group, index: entry.group.files.indexOf(entry.file) };
      $("lightbox").hidden = false;
      document.body.style.overflow = "hidden";
      renderLightbox();
    }

    function closeLightbox() {
      lb = null;
      $("lightbox").hidden = true;
      $("lbImg").removeAttribute("src");
      document.body.style.overflow = "";
    }

    function renderLightbox() {
      const group = lb.group;
      const file = group.files[lb.index];
      const status = statusMap.get(file.id);
      const url = imageUrl(file);
      if ($("lbImg").getAttribute("src") !== url) $("lbImg").src = url;
      const best = file.id === group.suggested_id ? " \u2605 Best shot" : "";
      $("lbTitle").textContent = `Group ${group.number} \u00b7 ${lb.index + 1} of ${group.files.length} \u00b7 ${file.name}${best}`;
      $("lbInfo").textContent = fileDetails(file, group).join(" | ");
      $("lbNotes").innerHTML = notesHtml(file);
      $("lbKeep").className = `toggle ${status === "keep" ? "on-keep" : ""}`;
      $("lbMove").className = `toggle ${status === "move" ? "on-move" : ""}`;
      const next = group.files[(lb.index + 1) % group.files.length];
      new Image().src = imageUrl(next);
    }

    function stepLightbox(delta) {
      const count = lb.group.files.length;
      lb.index = (lb.index + delta + count) % count;
      renderLightbox();
    }

    const report = (promise) => promise.catch((err) => toast(err.message));

    $("groups").addEventListener("click", (event) => {
      const target = event.target.closest("[data-action]");
      if (!target) return;
      const action = target.dataset.action;
      if (action === "view") {
        openLightbox(target.dataset.id);
      } else if (action === "keep" || action === "move") {
        toggleStatus(target.dataset.id, action);
      } else {
        const group = groupById.get(target.dataset.groupId);
        if (group) groupAction(group, action);
      }
    });

    document.addEventListener("click", (event) => {
      const target = event.target.closest("[data-page]");
      if (!target) return;
      page += target.dataset.page === "next" ? 1 : -1;
      render();
      window.scrollTo({ top: 0 });
    });

    document.addEventListener("keydown", (event) => {
      if (!lb) return;
      const file = lb.group.files[lb.index];
      const key = event.key.toLowerCase();
      if (key === "arrowleft") stepLightbox(-1);
      else if (key === "arrowright") stepLightbox(1);
      else if (key === "escape") closeLightbox();
      else if (key === "k") toggleStatus(file.id, "keep");
      else if (key === "m") toggleStatus(file.id, "move");
      else return;
      event.preventDefault();
    });

    ["searchBox", "typeFilter", "decisionFilter"].forEach((id) => {
      $(id).addEventListener(id === "searchBox" ? "input" : "change", () => {
        page = 0;
        render();
      });
    });
    $("folderInput").addEventListener("keydown", (event) => {
      if (event.key === "Enter") report(startScan());
    });
    $("browseBtn").addEventListener("click", () => report(browse()));
    $("scanBtn").addEventListener("click", () => report(startScan()));
    $("settingsBtn").addEventListener("click", () => {
      $("settings").hidden = !$("settings").hidden;
      $("settingsBtn").setAttribute("aria-expanded", String(!$("settings").hidden));
    });
    $("applySettingsBtn").addEventListener("click", () => report(applySettings()));
    $("acceptExactBtn").addEventListener("click", acceptExact);
    $("moveBtn").addEventListener("click", () => report(moveMarked()));
    $("undoBtn").addEventListener("click", () => report(undoMove()));
    $("lbPrev").addEventListener("click", () => stepLightbox(-1));
    $("lbNext").addEventListener("click", () => stepLightbox(1));
    $("lbClose").addEventListener("click", closeLightbox);
    $("lbKeep").addEventListener("click", () => toggleStatus(lb.group.files[lb.index].id, "keep"));
    $("lbMove").addEventListener("click", () => toggleStatus(lb.group.files[lb.index].id, "move"));
    $("lbStage").addEventListener("click", (event) => {
      if (event.target === $("lbStage")) closeLightbox();
    });

    window.addEventListener("beforeunload", () => {
      if (!saveTimer || !state) return;
      fetch("/api/decisions", {
        method: "POST",
        keepalive: true,
        headers: { "Content-Type": "application/json", "X-Review-Token": TOKEN },
        body: JSON.stringify({ folder: state.folder, status: Object.fromEntries(statusMap) }),
      });
    });

    loadState().catch((err) => {
      $("groups").innerHTML = `<div class="empty">${escapeHtml(err.message)}</div>`;
      $("subtitle").textContent = "Could not load groups.";
    });
  </script>
</body>
</html>
"""


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


class ApiError(Exception):
    def __init__(self, message: str, status: HTTPStatus = HTTPStatus.BAD_REQUEST) -> None:
        super().__init__(message)
        self.status = status


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        if self.rank[root_a] < self.rank[root_b]:
            self.parent[root_a] = root_b
        elif self.rank[root_a] > self.rank[root_b]:
            self.parent[root_b] = root_a
        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
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


# ---------------------------------------------------------------------------
# Per-image analysis
# ---------------------------------------------------------------------------


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

    report("Analyzing photos", 0, len(todo))
    if todo:
        with ThreadPoolExecutor(max_workers=options.workers) as pool:
            futures = {pool.submit(analyze_image, library, path, options, model): path for path in todo}
            for done, future in enumerate(as_completed(futures), start=1):
                try:
                    images.append(future.result())
                except Exception as exc:  # one unreadable file must not stop the scan
                    errors.append({"rel_path": library.rel(futures[future]), "error": f"{type(exc).__name__}: {exc}"})
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


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

if hasattr(np, "bitwise_count"):

    def popcount(values: np.ndarray) -> np.ndarray:
        return np.bitwise_count(values)

else:
    _POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

    def popcount(values: np.ndarray) -> np.ndarray:
        values = np.ascontiguousarray(values)
        return _POPCOUNT8[values.view(np.uint8)].reshape(values.shape + (8,)).sum(axis=-1)


class Features:
    """Hashes, colours and capture times of all images as numpy arrays for fast comparison."""

    def __init__(self, images: list[dict[str, Any]], options: ScanOptions) -> None:
        count = len(images)
        rotations = 4 if options.match_rotations else 1
        self.hashes = np.array(
            [[[int(value, 16) for value in item["hashes"][r]] for r in range(rotations)] for item in images],
            dtype=np.uint64,
        ).reshape(count, rotations, 3)
        colors = np.array([item["colors"] for item in images], dtype=np.float32).reshape(count, 3, 3, 3)
        self.colors = np.stack([np.rot90(colors, k=r, axes=(1, 2)).reshape(count, 27) for r in range(rotations)], axis=1)
        # Only real camera timestamps are trusted for the burst rule; file times change on copy/sync.
        self.times = np.array(
            [float(item["capture_ts"]) if item.get("ts_source") == "exif" else np.nan for item in images]
        )
        exact_ids: dict[str, int] = {}
        self.exact = np.array([exact_ids.setdefault(item["sha256"], len(exact_ids)) for item in images])

    def block_size(self, columns: int) -> int:
        per_row = max(1, columns * self.hashes.shape[1] * 3)
        return max(1, min(512, 4_000_000 // per_row))

    def links(self, rows: np.ndarray, cols: np.ndarray, options: ScanOptions) -> tuple[np.ndarray, np.ndarray]:
        """Return (linked, distance) matrices of shape (len(rows), len(cols))."""
        base = self.hashes[rows, 0]
        bits = popcount(base[:, None, None, :] ^ self.hashes[cols][None, :, :, :]).sum(axis=-1, dtype=np.int32)
        best_rotation = bits.argmin(axis=-1)
        dist = np.take_along_axis(bits, best_rotation[..., None], axis=-1)[..., 0]

        exact = self.exact[rows][:, None] == self.exact[cols][None, :]
        linked = exact.copy()
        limit = max(options.strong_threshold, options.burst_threshold)
        if options.session_seconds:
            limit = max(limit, options.session_threshold)
        candidates = (dist <= limit) & ~exact
        r_idx, c_idx = np.nonzero(candidates)
        if r_idx.size:
            left_colors = self.colors[rows[r_idx], 0]
            right_colors = self.colors[cols[c_idx], best_rotation[r_idx, c_idx]]
            color_diff = np.abs(left_colors - right_colors).mean(axis=1)
            pair_dist = dist[r_idx, c_idx]
            gap = np.abs(self.times[rows[r_idx]] - self.times[cols[c_idx]])
            burst = (pair_dist <= options.burst_threshold) & (gap <= options.burst_seconds)
            ok = (color_diff <= options.color_threshold) & ((pair_dist <= options.strong_threshold) | burst)
            # Same moment: shots seconds apart where the pose, framing or lighting changed.
            # Hashes are weak evidence here, so the capture time does most of the work.
            session = (
                (gap <= options.session_seconds)
                & (pair_dist <= options.session_threshold)
                & (color_diff <= 2 * options.color_threshold)
            )
            ok |= session
            linked[r_idx[ok], c_idx[ok]] = True
        return linked, np.where(exact, 0, dist)

    def link_matrix(self, members: np.ndarray, options: ScanOptions) -> tuple[np.ndarray, np.ndarray]:
        size = len(members)
        linked = np.zeros((size, size), dtype=bool)
        dist = np.zeros((size, size), dtype=np.int32)
        block = self.block_size(size)
        for start in range(0, size, block):
            linked[start : start + block], dist[start : start + block] = self.links(
                members[start : start + block], members, options
            )
        # Rotation matching is not perfectly symmetric, so make both directions agree.
        linked |= linked.T
        dist = np.minimum(dist, dist.T)
        np.fill_diagonal(linked, False)
        return linked, dist


def split_component(
    features: Features, members: np.ndarray, scores: np.ndarray, options: ScanOptions
) -> list[list[tuple[int, int]]]:
    """Split a chain of loosely connected photos into star-shaped groups.

    Union-find links A~B and B~C even when A and C look nothing alike. Here every
    group gets an anchor photo, and only photos directly similar to that anchor
    join it, so a long series of shots cannot snowball into one huge group.
    """
    linked, dist = features.link_matrix(members, options)
    remaining = np.ones(len(members), dtype=bool)
    clusters: list[list[tuple[int, int]]] = []
    while remaining.any():
        open_idx = np.nonzero(remaining)[0]
        degree = linked[np.ix_(open_idx, open_idx)].sum(axis=1)
        if degree.max() == 0:
            break
        order = np.lexsort((-scores[members[open_idx]], -degree))
        anchor = int(open_idx[order[0]])
        group = [anchor] + [int(j) for j in open_idx if j != anchor and linked[anchor, j]]
        clusters.append([(int(members[j]), int(dist[anchor, j]) if j != anchor else 0) for j in group])
        remaining[group] = False
    return clusters


def relative_values(values: list[float | None], min_spread: float) -> tuple[list[float], bool]:
    """Scale values to 0..1 within a group; 0.5 for everyone when they barely differ."""
    present = [value for value in values if value is not None]
    if not present:
        return [0.0] * len(values), False
    low, high = min(present), max(present)
    varies = high - low >= min_spread
    return [
        0.0 if value is None else ((value - low) / (high - low) if varies else 0.5) for value in values
    ], varies


def rank_group(items: list[dict[str, Any]], options: ScanOptions) -> list[tuple[dict[str, Any], float, list[tuple[str, str]]]]:
    """Order a group best-first. Returns (item, rank score, notes)."""
    faces = [item.get("faces") or [] for item in items]
    use_faces = options.prefer_faces and any(faces)
    ranked: list[tuple[dict[str, Any], float, list[tuple[str, str]]]] = []
    if not use_faces:
        ranked = [(item, float(item["score"]), []) for item in items]
    else:
        most_faces = max(len(group_faces) for group_faces in faces)

        def mean(group_faces: list[dict[str, float]], key: str) -> float | None:
            return float(np.mean([face[key] for face in group_faces])) if group_faces else None

        eyes, eyes_vary = relative_values([mean(f, "eyes") for f in faces], 0.06)
        smiles, smiles_vary = relative_values([mean(f, "mouth") for f in faces], 0.06)
        focus, focus_varies = relative_values(
            [math.log1p(mean(f, "sharp")) if f else None for f in faces], 0.25
        )
        for k, item in enumerate(items):
            item_faces = faces[k]
            notes: list[tuple[str, str]] = []
            if not item_faces:
                people = 0.0
                notes.append(("bad", "No face visible"))
            else:
                presence = len(item_faces) / most_faces
                frontal = mean(item_faces, "frontal") or 0.0
                people = 25 * presence + 20 * frontal + 10 * eyes[k] + 8 * smiles[k] + 12 * focus[k]
                if len(item_faces) < most_faces:
                    notes.append(("bad", f"{len(item_faces)} of {most_faces} faces visible"))
                if frontal >= 0.7:
                    notes.append(("good", "Facing camera"))
                elif frontal < 0.4:
                    notes.append(("bad", "Face turned away"))
                if eyes_vary and eyes[k] >= 0.75:
                    notes.append(("good", "Eyes open"))
                elif eyes_vary and eyes[k] <= 0.25:
                    notes.append(("bad", "Eyes may be closed"))
                if smiles_vary and smiles[k] >= 0.75:
                    notes.append(("good", "Smiling"))
                if focus_varies and focus[k] >= 0.75:
                    notes.append(("good", "Face in focus"))
                elif focus_varies and focus[k] <= 0.25:
                    notes.append(("bad", "Face blurry"))
            # In people photos, how people look matters more than raw technical quality.
            ranked.append((item, 0.5 * float(item["score"]) + people, notes))

    ranked.sort(
        key=lambda entry: (
            entry[1],
            entry[0]["width"] * entry[0]["height"],
            entry[0]["sharpness"],
            entry[0]["size"],
            -len(entry[0]["rel_path"]),
        ),
        reverse=True,
    )
    return ranked


def format_taken(item: dict[str, Any]) -> str:
    try:
        return datetime.fromtimestamp(float(item["capture_ts"])).strftime("%Y-%m-%d %H:%M:%S")
    except (KeyError, TypeError, ValueError, OSError, OverflowError):
        return ""


def public_file(item: dict[str, Any], rank: float, notes: list[tuple[str, str]], distance: int) -> dict[str, Any]:
    return {
        "id": item["id"],
        "rel_path": item["rel_path"],
        "name": item["name"],
        "size": item["size"],
        "width": item["width"],
        "height": item["height"],
        "v": item["mtime_ns"],
        "taken": format_taken(item),
        "score": float(item["score"]),
        "rank": round(rank, 2),
        "sharpness": float(item["sharpness"]),
        "similarity": round(100 * (1 - distance / HASH_BITS)),
        "faces": len(item.get("faces") or []),
        "notes": [{"kind": kind, "text": text} for kind, text in notes],
    }


def build_groups(images: list[dict[str, Any]], options: ScanOptions, index: dict[str, Any]) -> dict[str, Any]:
    count = len(images)
    clusters: list[list[tuple[int, int]]] = []
    if count > 1:
        features = Features(images, options)
        union = UnionFind(count)
        everything = np.arange(count)
        block = features.block_size(count)
        for start in range(0, count - 1, block):
            rows = everything[start : start + block]
            cols = everything[start + 1 :]
            linked, _ = features.links(rows, cols, options)
            linked &= cols[None, :] > rows[:, None]
            for r, c in zip(*np.nonzero(linked)):
                union.union(int(rows[r]), int(cols[c]))
        components: dict[int, list[int]] = {}
        for position in range(count):
            components.setdefault(union.find(position), []).append(position)
        scores = np.array([float(item["score"]) for item in images])
        for members in components.values():
            if len(members) > 1:
                clusters.extend(split_component(features, np.array(members), scores, options))

    groups: list[dict[str, Any]] = []
    for cluster in clusters:
        items = [images[position] for position, _ in cluster]
        distance = {images[position]["id"]: d for position, d in cluster}
        ranked = rank_group(items, options)
        best = ranked[0][0]
        if len({item["sha256"] for item in items}) == 1:
            kind = "exact"
        elif max(distance.values()) <= max(options.strong_threshold, options.burst_threshold):
            kind = "similar"
        else:
            kind = "moment"
        groups.append(
            {
                "id": hashlib.sha1("|".join(sorted(item["id"] for item in items)).encode("utf-8")).hexdigest()[:16],
                "kind": kind,
                "suggested_id": best["id"],
                "has_faces": options.prefer_faces and any(item.get("faces") for item in items),
                "files": [public_file(item, rank, notes, distance[item["id"]]) for item, rank, notes in ranked],
                "search_text": " ".join(item["rel_path"] for item in items),
            }
        )
    groups.sort(key=lambda group: (0 if group["kind"] == "exact" else 1, min(f["rel_path"].lower() for f in group["files"])))

    summary = {
        "image_count": count,
        "group_count": len(groups),
        "exact_group_count": sum(1 for group in groups if group["kind"] == "exact"),
        "similar_group_count": sum(1 for group in groups if group["kind"] == "similar"),
        "moment_group_count": sum(1 for group in groups if group["kind"] == "moment"),
        "people_group_count": sum(1 for group in groups if group["has_faces"]),
        "faces_available": bool(index.get("faces_available")),
        "faces_reason": index.get("faces_reason", ""),
        "heif_available": bool(HEIF_EXTENSIONS),
        "move_dir": MOVE_DIR_NAME,
    }
    return {"signature": grouping_signature(options), "groups": groups, "summary": summary}


def keep_best_shots(library: Library, groups: list[dict[str, Any]]) -> None:
    """Mark the suggested best shot Keep in every group you have not touched yet."""
    status = library.load_decisions()
    changed = False
    for group in groups:
        if not any(file["id"] in status for file in group["files"]):
            status[group["suggested_id"]] = "keep"
            changed = True
    if changed:
        library.save_decisions(status)


# ---------------------------------------------------------------------------
# Moving and undo
# ---------------------------------------------------------------------------


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
        if failed:
            moves.append({"at": batch.get("at"), "items": failed})
        images.sort(key=lambda item: item["rel_path"].lower())
        index.pop("grouping", None)
        library.save_index(index)
        library.save_moves(moves)
        library.save_decisions(status)
    return {"restored": restored, "errors": errors}


# ---------------------------------------------------------------------------
# App state and HTTP server
# ---------------------------------------------------------------------------


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
        threading.Thread(target=self._run_scan, args=(library, force), daemon=True).start()
        return snapshot

    def _progress(self, phase: str, done: int = 0, total: int = 0) -> None:
        with self.job_lock:
            self.job.update(phase=phase, done=done, total=total)

    def _run_scan(self, library: Library, force: bool) -> None:
        message, error = "", None
        try:
            payload = scan(library, self.options, force=force, progress=self._progress)
            summary = payload["grouping"]["summary"]
            message = f"Scan complete: {summary['image_count']} photos, {summary['group_count']} groups to review."
            if payload["errors"]:
                message += f" {len(payload['errors'])} files could not be read."
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        with self.job_lock:
            self.job.update(running=False, phase="Finished", message=message, error=error)

    def update_options(self, data: dict[str, Any]) -> None:
        self.options = options_from_dict(data, self.options)
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

    def log_request(self, code: Any = "-", size: Any = "-") -> None:
        if str(getattr(code, "value", code)).startswith(("4", "5")):
            super().log_request(code, size)

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
            self.send_bytes(HTML.replace("__TOKEN__", self.app.token).encode("utf-8"), "text/html; charset=utf-8")
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


def serve(app: App, port: int, open_browser: bool) -> None:
    ReviewHandler.app = app
    server = ThreadingHTTPServer((app.host, port), ReviewHandler)
    server.daemon_threads = True
    url = f"http://{'127.0.0.1' if app.host in ('0.0.0.0', '') else app.host}:{port}"
    print(f"Photo review UI is running at {url}")
    print("Press Ctrl+C to stop it.")
    if open_browser:
        threading.Timer(0.6, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


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
    parser.add_argument("--folder", help="Folder to scan. Defaults to the last folder used, or this script's folder.")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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

    folder_text = args.folder or settings.get("folder") or str(SCRIPT_DIR)
    folder = Path(folder_text).expanduser()
    if args.folder and not folder.is_dir():
        sys.exit(f"Folder not found: {folder}")

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
        if not args.serve:
            return

    serve(App(folder, options, args.host), args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
