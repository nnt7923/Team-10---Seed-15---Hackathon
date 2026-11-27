"""Select and extract only the nuImages key frames that contain the classes we are short of.

  python scripts/nuimages_select.py --meta-tgz <...>/nuimages-v1.0-all-metadata.tgz \
         --samples-tgz <...>/nuimages-v1.0-all-samples.tgz --out raw/nuimages

1. extracts the metadata tables (v1.0-train / v1.0-val) into <out>/
2. scores every annotated key frame by its rare-class content and keeps a quota per class
3. streams the 16 GB samples archive once and writes only the selected JPEGs to <out>/samples/...
The selection is written to <out>/selection.json (reproducible, cited in the report).
"""
from __future__ import annotations

import argparse
import json
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

RARE = {  # category name -> quota key
    "vehicle.bus.rigid": "bus", "vehicle.bus.bendy": "bus",
    "vehicle.bicycle": "bicycle", "vehicle.construction": "other_vehicle", "vehicle.trailer": "other_vehicle",
    "vehicle.truck": "truck", "vehicle.motorcycle": "motorcycle",
}
QUOTA = {"bus": 4000, "bicycle": 4000, "other_vehicle": 3000, "truck": 2500, "motorcycle": 2500}  # images per key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-tgz", required=True)
    ap.add_argument("--samples-tgz", default=None)
    ap.add_argument("--select-only", action="store_true", help="extract tables + write selection.json only")
    ap.add_argument("--out", default="raw/nuimages")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if not (out / "v1.0-train" / "object_ann.json").exists():
        print("[meta] extracting tables", flush=True)
        with tarfile.open(a.meta_tgz, "r:*") as t:
            safe = [m for m in t.getmembers() if m.isfile() and m.name.endswith(".json")
                    and not m.name.startswith(("/", "..")) and ".." not in m.name.split("/")]
            t.extractall(out, members=safe)

    chosen, stats = {}, Counter()
    for split in ("v1.0-train", "v1.0-val"):
        d = out / split
        if not (d / "object_ann.json").exists():
            continue
        cats = {c["token"]: c["name"] for c in json.load(open(d / "category.json"))}
        sd = {s["token"]: s for s in json.load(open(d / "sample_data.json")) if s.get("is_key_frame")}
        per_img = defaultdict(Counter)
        for o in json.load(open(d / "object_ann.json")):
            key = RARE.get(cats.get(o["category_token"], ""))
            if key and o["sample_data_token"] in sd:
                per_img[o["sample_data_token"]][key] += 1
        # rarest keys first, images with the most rare instances first
        for key in sorted(QUOTA, key=lambda k: QUOTA[k]):
            cand = sorted((t for t, c in per_img.items() if c[key] and t not in chosen),
                          key=lambda t: -sum(per_img[t].values()))
            take = cand[: max(0, QUOTA[key] - stats[key])]
            for t in take:
                chosen[t] = {"split": split, "file": sd[t]["filename"], "counts": dict(per_img[t])}
            stats[key] += len(take)
        print(f"[select] {split}: {len(per_img)} key frames with rare classes", flush=True)
    files = {v["file"] for v in chosen.values()}
    print(f"[select] {len(files)} images selected, per-key {dict(stats)}", flush=True)
    (out / "selection.json").write_text(json.dumps({"quota": QUOTA, "per_key": stats, "images": chosen}),
                                        encoding="utf-8")

    if a.select_only or not a.samples_tgz:
        return
    n = 0
    with tarfile.open(a.samples_tgz, "r|*") as t:  # stream: one pass over the archive
        for m in t:
            if m.isfile() and m.name in files and ".." not in m.name.split("/") and not m.name.startswith("/"):
                dst = out / m.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                with t.extractfile(m) as s, open(dst, "wb") as f:
                    f.write(s.read())
                n += 1
                if n % 1000 == 0:
                    print(f"[extract] {n}/{len(files)}", flush=True)
    print(f"[extract] done {n}/{len(files)}", flush=True)


if __name__ == "__main__":
    main()
