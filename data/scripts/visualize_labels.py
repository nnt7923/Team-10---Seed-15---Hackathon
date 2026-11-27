"""QA: draw YOLO-seg labels of the exported dataset onto images (grid of random samples).

  python scripts/visualize_labels.py --dataset dataset_out --split train --n 12 --source mapillary_vistas_v2
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import yaml

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948",
           "#52514e", "#00a3a3", "#a35200"]


def hex2bgr(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (4, 2, 0))


def draw(img_path: Path, lbl_path: Path, names: dict) -> np.ndarray:
    img = cv2.imread(str(img_path))
    H, W = img.shape[:2]
    over = img.copy()
    for line in lbl_path.read_text().splitlines():
        v = line.split()
        c = int(v[0])
        pts = (np.asarray(v[1:], float).reshape(-1, 2) * [W, H]).astype(np.int32)
        col = hex2bgr(PALETTE[c % len(PALETTE)])
        cv2.fillPoly(over, [pts], col)
        cv2.polylines(img, [pts], True, col, 2)
        x, y = pts[:, 0].min(), max(12, pts[:, 1].min() - 4)
        cv2.putText(img, f"{c}:{names[c]}", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
        cv2.putText(img, f"{c}:{names[c]}", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
    return cv2.addWeighted(over, 0.35, img, 0.65, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--source", default=None, help="only images from this source id")
    ap.add_argument("--cls", type=int, default=None, help="only images containing this class id")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    root = Path(a.dataset)
    names = yaml.safe_load(open(root / "data.yaml", encoding="utf-8"))["names"]
    meta = [json.loads(l) for l in open(root / "metadata" / "images.jsonl", encoding="utf-8")]
    meta = [m for m in meta if m["split"] == a.split and (a.source is None or m["source"] == a.source)]
    if a.cls is not None:
        meta = [m for m in meta if names[a.cls] in m["counts"]]
    random.Random(a.seed).shuffle(meta)
    tiles = [draw(root / m["file"], root / m["label_file"], names) for m in meta[: a.n]]
    if not tiles:
        raise SystemExit("no matching images")
    h = 360
    tiles = [cv2.resize(t, (int(t.shape[1] * h / t.shape[0]), h)) for t in tiles]
    w = max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 2, 2, 2, 2 + w - t.shape[1], cv2.BORDER_CONSTANT, value=(252, 252, 251)) for t in tiles]
    cols = 3
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 252))
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    out = Path(a.out or root / "qa" / f"{a.split}_{a.source or 'all'}_{a.cls if a.cls is not None else 'all'}.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid)
    print(out)


if __name__ == "__main__":
    main()
