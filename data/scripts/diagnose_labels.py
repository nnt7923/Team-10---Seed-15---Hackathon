"""Label diagnosis: run a trained 11-class checkpoint over the exported dataset and compare with the labels,
per SOURCE and per CLASS. High "extra" = confident predictions with no matching label (likely MISSING labels
in that source); high "missed" = labels the model does not find (hard / possibly wrong labels).

  python scripts/diagnose_labels.py --weights ../best.pt --dataset dataset_out --n 150 --conf 0.5
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--dataset", default="dataset_out")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--out", default="reports/label_diagnosis.json")
    a = ap.parse_args()
    from ultralytics import YOLO
    model = YOLO(a.weights)
    names = [model.names[i] for i in sorted(model.names)]
    root = Path(a.dataset)
    meta = [json.loads(l) for l in open(root / "metadata" / "images.jsonl", encoding="utf-8")]
    by_src = defaultdict(list)
    for m in meta:
        if m["split"] == "train":
            by_src[m["source"]].append(m)
    res, examples = {}, defaultdict(list)
    for src, ms in sorted(by_src.items()):
        random.Random(15).shuffle(ms)
        ms = ms[: a.n]
        tp, extra, missed, gt_n = Counter(), Counter(), Counter(), Counter()
        for m in ms:
            img = root / m["file"]
            r = model(str(img), conf=a.conf, imgsz=640, verbose=False)[0]
            H, W = r.orig_shape
            gts = []
            for line in (root / m["label_file"]).read_text().splitlines():
                v = line.split()
                pts = np.asarray(v[1:], float).reshape(-1, 2) * [W, H]
                gts.append((int(v[0]), [*pts.min(0), *pts.max(0)]))
            for c, _ in gts:
                gt_n[c] += 1
            used = set()
            preds = list(zip(r.boxes.cls.cpu().numpy().astype(int), r.boxes.xyxy.cpu().numpy().tolist())) if r.boxes is not None else []
            for c, b in preds:
                best, bj = 0.0, -1
                for j, (gc, gb) in enumerate(gts):
                    if gc == c and j not in used:
                        v = iou(b, gb)
                        if v > best:
                            best, bj = v, j
                if best >= 0.5:
                    used.add(bj)
                    tp[c] += 1
                else:
                    # matched another class? -> confusion, else extra
                    other = max([(iou(b, gb), gc) for gc, gb in gts] or [(0, -1)])
                    key = f"{names[c]}->as_{names[other[1]]}" if other[0] >= 0.5 else None
                    extra[c] += 1
                    if len(examples[(src, names[c])]) < 6:
                        examples[(src, names[c])].append({"file": m["file"], "box": [round(x) for x in b], "conf_ge": a.conf,
                                                          "overlaps_label": key})
            for j, (gc, _) in enumerate(gts):
                if j not in used:
                    missed[gc] += 1
        rows = {}
        for i, n in enumerate(names):
            p = tp[i] + extra[i]
            rows[n] = {"labels": gt_n[i], "preds": p, "extra_rate": round(extra[i] / p, 3) if p else None,
                       "missed_rate": round(missed[i] / gt_n[i], 3) if gt_n[i] else None}
        res[src] = {"images": len(ms), "per_class": rows}
        print(f"\n== {src} ({len(ms)} imgs)")
        print(f"{'class':<15}{'labels':>8}{'preds':>7}{'extra%':>8}{'missed%':>9}")
        for n, r in rows.items():
            if r["labels"] or r["preds"]:
                e = f"{r['extra_rate'] * 100:.0f}" if r["extra_rate"] is not None else "-"
                mi = f"{r['missed_rate'] * 100:.0f}" if r["missed_rate"] is not None else "-"
                print(f"{n:<15}{r['labels']:>8}{r['preds']:>7}{e:>8}{mi:>9}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"weights": a.weights, "conf": a.conf, "sources": res,
               "examples": {f"{s}|{c}": v for (s, c), v in examples.items()}},
              open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("\n->", a.out)


if __name__ == "__main__":
    main()
