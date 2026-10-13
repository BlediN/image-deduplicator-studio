"""Group matching photos and rank the best shot in each group."""

from __future__ import annotations

import hashlib
import logging
import math
from datetime import datetime
from typing import Any

import numpy as np

from .config import HEIF_EXTENSIONS, MOVE_DIR_NAME, HASH_BITS, ScanOptions, grouping_signature
from .storage import Library

logger = logging.getLogger(__name__)


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
    marked = 0
    for group in groups:
        if not any(file["id"] in status for file in group["files"]):
            status[group["suggested_id"]] = "keep"
            marked += 1
    if marked:
        library.save_decisions(status)
    logger.debug("Auto-marked the best shot Keep in %d of %d groups", marked, len(groups))
