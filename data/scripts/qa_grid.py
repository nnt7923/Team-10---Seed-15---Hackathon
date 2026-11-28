"""Draw exported YOLO-seg labels on random images of chosen sources -> one PNG grid for a quick visual QA.

  python scripts/qa_grid.py --dataset dataset_v2 --sources vn_traffic_sign_hf,roboflow_vn_traffic_v2i --n 12 --out reports/qa_v2.jpg
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

COLORS = [(56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207), (10, 249, 72),
          (23, 204, 146), (134, 219, 61), (52, 147, 26), (187, 212, 0), (168, 153, 44)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset_v2")
    ap.add_argument("--sources", default="")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--seed", type=int, default=15)
    ap.add_argument("--out", default="reports/qa_v2.jpg")
    a = ap.parse_args()
    root = Path(a.dataset)
    import yaml
    names = yaml.safe_load(open(root / "data.yaml", encoding="utf-8"))["names"]
    names = [names[i] for i in sorted(names)] if isinstance(names, dict) else names
    meta = [json.loads(l) for l in open(root / "metadata" / "images.jsonl", encoding="utf-8")]
    want = set(a.sources.split(",")) if a.sources else None
    meta = [m for m in meta if m["split"] == "train" and (want is None or m["source"] in want)]
    random.Random(a.seed).shuffle(meta)
    tiles = []
    for m in meta[: a.n]:
        img = cv2.imread(str(root / m["file"]))
        H, W = img.shape[:2]
        over = img.copy()
        for line in (root / m["label_file"]).read_text().splitlines():
            v = line.split()
            c = int(v[0])
            pts = (np.asarray(v[1:], float).reshape(-1, 2) * [W, H]).astype(np.int32)
            cv2.fillPoly(over, [pts], COLORS[c % len(COLORS)])
            cv2.polylines(img, [pts], True, COLORS[c % len(COLORS)], 1)
            x, y = pts.min(0)
            cv2.putText(img, names[c][:6], (int(x), max(10, int(y) - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (255, 255, 255), 1, cv2.LINE_AA)
        img = cv2.addWeighted(over, 0.35, img, 0.65, 0)
        s = 480 / max(H, W)
        img = cv2.resize(img, (round(W * s), round(H * s)))
        tile = np.zeros((480, 480, 3), np.uint8)
        tile[: img.shape[0], : img.shape[1]] = img
        cv2.putText(tile, m["source"][:30], (4, 474), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    while len(tiles) % a.cols:
        tiles.append(np.zeros((480, 480, 3), np.uint8))
    rows = [np.hstack(tiles[i:i + a.cols]) for i in range(0, len(tiles), a.cols)]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(a.out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("->", a.out, len(tiles), "tiles")


if __name__ == "__main__":
    main()
