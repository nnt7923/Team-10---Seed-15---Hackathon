"""Shared helpers: config loading, record schema, label mapping, geometry.

Record (one JSON line per image) — the unified intermediate format:
{
  "uid": "mapillary_vistas_v2/abc123",       # unique id = <source>/<relative stem>
  "source": "mapillary_vistas_v2",
  "image": "D:/.../abc123.jpg",               # absolute path to original image
  "width": 4032, "height": 3024,
  "orig_id": "abc123", "orig_split": "train",
  "group": "seq_or_city",                     # leakage group (video / sequence / city)
  "domain": "vn|asia|global", "tier": "A..E",
  "attrs": {"timeofday": "night", "weather": "rain"},
  "labeled_classes": [...], "missing_classes": [...],
  "instances": [
     {"raw": "object--vehicle--car", "cls": "car",
      "polys": [[[x, y], ...], ...],           # pixel coords, may be multi-part
      "box": [x1, y1, x2, y2],                 # present for box-only sources
      "origin": "human|merged|sam_box|teacher", "conf": 1.0}
  ]
}
"""
from __future__ import annotations

import fnmatch
import json
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np
import yaml

SPECIAL_TOKENS = {"@rider", "@rider_moto", "@rider_bike", "@person", "@ignore"}


# --------------------------------------------------------------------------- config
def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class Classes:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.names: list[str] = [cfg["names"][i] for i in sorted(cfg["names"])]
        self.index = {n: i for i, n in enumerate(self.names)}
        self.display = cfg.get("display_vi", {})

    def __len__(self):
        return len(self.names)


def load_pipeline(path: str | Path) -> tuple[dict, Classes, list[dict], Path]:
    """Return (pipeline_cfg, classes, enabled_sources, base_dir). Paths resolved vs config's project root."""
    path = Path(path).resolve()
    base = path.parent.parent  # configs/ -> project root
    cfg = load_yaml(path)
    for k in ("data_root", "work_dir", "out_dir", "classes", "sources"):
        p = Path(cfg[k])
        cfg[k] = str(p if p.is_absolute() else (base / p))
    classes = Classes(load_yaml(cfg["classes"]))
    reg = load_yaml(cfg["sources"])
    sources = [s for s in reg["sources"] if s.get("enabled", False)]
    return cfg, classes, sources, base


# --------------------------------------------------------------------------- jsonl io
def write_jsonl(path: str | Path, records: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


# --------------------------------------------------------------------------- labels
def map_label(raw: str, label_map: dict) -> str:
    """Exact match first, then glob patterns in declaration order. Unmapped -> @ignore."""
    raw_s = str(raw)
    if raw_s in label_map:
        return label_map[raw_s]
    for pat, tok in label_map.items():
        if any(ch in pat for ch in "*?[") and fnmatch.fnmatchcase(raw_s, pat):
            return tok
    return "@ignore"


# --------------------------------------------------------------------------- geometry
def poly_area(poly) -> float:
    a = np.asarray(poly, dtype=np.float64)
    if len(a) < 3:
        return 0.0
    x, y = a[:, 0], a[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def polys_area(polys) -> float:
    return float(sum(poly_area(p) for p in polys))


def polys_bbox(polys) -> list[float]:
    pts = np.concatenate([np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in polys], 0)
    return [float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())]


def mask_to_polys(mask: np.ndarray, scale: float = 1.0, offset=(0.0, 0.0), min_area: float = 4.0,
                  eps: float = 0.7) -> list[list[list[float]]]:
    """Binary mask -> list of polygons (outer contours) in original pixel coords.

    `scale` = working/original; coords are divided by scale and shifted by offset.
    """
    m = (mask > 0).astype(np.uint8)
    if m.sum() == 0:
        return []
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        c = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(np.float64)
        if len(c) < 3:
            continue
        c = c / scale + np.asarray(offset, dtype=np.float64)
        out.append(c.round(2).tolist())
    return out


def rasterize(polys, x0: float, y0: float, w: int, h: int, scale: float) -> np.ndarray:
    """Fill polygons into a (h, w) crop whose top-left is (x0, y0) in original coords, at `scale`."""
    m = np.zeros((h, w), np.uint8)
    pts = [np.round((np.asarray(p, np.float64) - [x0, y0]) * scale).astype(np.int32) for p in polys if len(p) >= 3]
    if pts:
        cv2.fillPoly(m, pts, 1)
    return m


# --------------------------------------------------------------------------- COCO RLE
def _rle_counts_from_string(s: str) -> list[int]:
    cnts, p, m = [], 0, 0
    while p < len(s):
        x, k, more = 0, 0, 1
        while more:
            c = ord(s[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = c & 0x20
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if m > 2:
            x += cnts[m - 2]
        cnts.append(x)
        m += 1
    return cnts


def rle_to_mask(rle: dict) -> np.ndarray:
    h, w = rle["size"]
    counts = rle["counts"]
    if isinstance(counts, (str, bytes)):
        counts = _rle_counts_from_string(counts.decode() if isinstance(counts, bytes) else counts)
    flat = np.zeros(h * w, np.uint8)
    pos, val = 0, 0
    for c in counts:
        if val:
            flat[pos:pos + c] = 1
        pos += c
        val ^= 1
    return flat.reshape((w, h)).T  # COCO RLE is column-major


def image_size(path: str | Path) -> tuple[int, int]:
    from PIL import Image
    with Image.open(path) as im:
        return im.size  # (w, h)
