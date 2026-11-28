"""VN-only validation set from Vietnamese images that v2 never trained on.

1. filter  : keep converted VN records whose image is not in dataset_v2 (any split) and whose pHash is farther than
             --min-dist from every dataset_v2 image (drops neighbouring video frames / Roboflow re-exports), then one
             image per near-duplicate cluster inside the candidates.
2. (run_pipeline --stages sam,complete,resolve on work_vnval)  -> 02_resolved.jsonl
3. export  : YOLO-seg val set (640 px) -> dataset_vnval/{images,labels}/val + data.yaml (+ metadata/images.jsonl)

  python scripts/build_vn_val.py filter --work work_vnval --ref dataset_v2 --max-per-source 200
  python scripts/build_vn_val.py export --work work_vnval --out dataset_vnval
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import imagehash  # noqa: E402

from vtb.quality import scan  # noqa: E402

CLASSES = ["car", "bus", "truck", "other_vehicle", "motorcycle", "motorcyclist", "bicycle", "cyclist", "pedestrian",
           "sign", "traffic_light"]


def _norm(p: str) -> str:
    return os.path.abspath(p).replace("\\", "/").lower()


def cmd_filter(a):
    ref = [json.loads(l) for l in open(Path(a.ref) / "metadata" / "images.jsonl", encoding="utf-8")]
    used = {_norm(m["orig_image"]) for m in ref}
    ref_h = [imagehash.hex_to_hash(m["phash"]) for m in ref if m.get("phash")]
    conv = Path(a.work) / "converted"
    rng = random.Random(15)
    for f in sorted(conv.glob("*.jsonl")):
        recs = [json.loads(l) for l in open(f, encoding="utf-8")]
        cand = [r for r in recs if _norm(r["image"]) not in used]
        scan(cand)
        keep, seen = [], []
        for r in cand:
            if not r.get("phash"):
                continue
            h = imagehash.hex_to_hash(r["phash"])
            if min((h - x for x in ref_h), default=64) <= a.min_dist:
                continue                      # near-copy of a training image
            if any(h - x <= a.min_dist for x in seen):
                continue                      # near-copy of another candidate
            seen.append(h)
            keep.append(r)
        rng.shuffle(keep)
        keep = keep[: a.max_per_source]
        for r in keep:
            r["split"] = "val"
        with open(f, "w", encoding="utf-8") as fo:
            for r in keep:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{f.stem:<26} {len(recs):>5} converted · {len(cand):>4} not in v2 · {len(keep):>4} kept (pHash > {a.min_dist})")


def cmd_export(a):
    import cv2
    import numpy as np
    out = Path(a.out)
    (out / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "val").mkdir(parents=True, exist_ok=True)
    (out / "metadata").mkdir(parents=True, exist_ok=True)
    recs = [json.loads(l) for l in open(Path(a.work) / "02_resolved.jsonl", encoding="utf-8")]
    meta, counts = [], {c: 0 for c in CLASSES}
    for r in recs:
        img = cv2.imread(r["image"])
        if img is None:
            continue
        H, W = img.shape[:2]
        s = min(1.0, 640 / max(H, W))
        if s < 1:
            img = cv2.resize(img, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)
        stem = f"{r['source']}__{Path(r['image']).stem}"[:150]
        lines = []
        for it in r["instances"]:
            if it["cls"] not in CLASSES:
                continue
            for poly in it["polygons"]:
                p = np.asarray(poly, float).reshape(-1, 2)
                if len(p) < 3:
                    continue
                p[:, 0] /= W
                p[:, 1] /= H
                lines.append(f"{CLASSES.index(it['cls'])} " + " ".join(f"{v:.5f}" for v in p.clip(0, 1).ravel()))
                counts[it["cls"]] += 1
                break                         # one polygon per instance (largest first after resolve)
        cv2.imwrite(str(out / "images" / "val" / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (out / "labels" / "val" / f"{stem}.txt").write_text("\n".join(lines))
        origin = {}
        for it in r["instances"]:
            origin[it.get("origin", "?")] = origin.get(it.get("origin", "?"), 0) + 1
        meta.append({"file": f"images/val/{stem}.jpg", "source": r["source"], "orig_image": r["image"],
                     "label_origin": origin})
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(
        ["Car", "Bus", "Truck", "Other vehicle", "Motorcycle", "Motorcyclist", "Bicycle", "Cyclist", "Pedestrian",
         "Sign", "Traffic light"]))
    (out / "data.yaml").write_text(f"path: {out.resolve().as_posix()}\ntrain: images/val\nval: images/val\nnames:\n{names}\n",
                                   encoding="utf-8")
    with open(out / "metadata" / "images.jsonl", "w", encoding="utf-8") as fo:
        for m in meta:
            fo.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"{len(meta)} images ->", out)
    print({k: v for k, v in counts.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("filter")
    f.add_argument("--work", default="work_vnval")
    f.add_argument("--ref", default="dataset_v2")
    f.add_argument("--min-dist", type=int, default=10)
    f.add_argument("--max-per-source", type=int, default=200)
    e = sub.add_parser("export")
    e.add_argument("--work", default="work_vnval")
    e.add_argument("--out", default="dataset_vnval")
    a = ap.parse_args()
    cmd_filter(a) if a.cmd == "filter" else cmd_export(a)
