"""Carve a 100% Vietnamese val split out of dataset_v2 by VIDEO group (no frame of a val video stays in train).

Writes next to dataset_v2/data.yaml (originals untouched):
  val_vn.txt            VN val images (by group, ~--frac of each VN source)
  train_vnsplit.txt     train.txt (RFS lines) without those images
  data_vnval.yaml       train: train_vnsplit.txt · val: val_vn.txt
  metadata/val_vn.json  groups and counts

Video id: roboflow_vn_traffic_v2i "IMG_4004-MOV_out0013..." -> IMG_4004-MOV; vn_traffic_sign_hf "0123" -> block of
50 consecutive numbers (frames were numbered in shooting order); roboflow_traffic_vn -> one video.

  python scripts/split_vn_val.py --dataset dataset_v2 --frac 0.15
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


def video_id(m: dict) -> str:
    name = Path(m["orig_image"].replace("\\", "/")).stem
    s = m["source"]
    if s == "roboflow_vn_traffic_v2i":
        return s + "/" + re.split(r"_out\d+", name)[0]
    if s == "vn_traffic_sign_hf" and name.isdigit():
        return f"{s}/block{int(name) // 50:03d}"
    if s == "roboflow_traffic_vn":
        return s + "/video"
    return m["group"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset_v2")
    ap.add_argument("--frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=15)
    a = ap.parse_args()
    root = Path(a.dataset)
    meta = [json.loads(l) for l in open(root / "metadata" / "images.jsonl", encoding="utf-8")]
    vn = [m for m in meta if m.get("domain") == "vn" and m["split"] == "train"]
    by_src = defaultdict(lambda: defaultdict(list))
    for m in vn:
        by_src[m["source"]][video_id(m)].append(m)
    val, report = [], {}
    for src, groups in sorted(by_src.items()):
        n = sum(len(v) for v in groups.values())
        order = sorted(groups, key=lambda g: hashlib.md5(f"{a.seed}:{g}".encode()).hexdigest())
        picked, k = [], 0
        for g in order:                      # whole videos until the source reaches ~frac of its images
            if k >= a.frac * n:
                break
            if len(groups) > 1 and k + len(groups[g]) > 2 * a.frac * n:
                continue                     # skip a video that alone would overshoot
            picked.append(g)
            k += len(groups[g])
        for g in picked:
            val += groups[g]
        report[src] = {"images": n, "videos": len(groups), "val_videos": len(picked), "val_images": k}
        print(f"{src:<26} {n:>5} imgs · {len(groups):>3} videos -> val {len(picked):>3} videos / {k:>4} imgs")
    vfiles = {m["file"] for m in val}
    (root / "val_vn.txt").write_text("".join(f"./{f}\n" for f in sorted(vfiles)), encoding="utf-8")
    tr = [l for l in (root / "train.txt").read_text(encoding="utf-8").splitlines() if l.strip()[2:] not in vfiles]
    (root / "train_vnsplit.txt").write_text("\n".join(tr) + "\n", encoding="utf-8")
    y = (root / "data.yaml").read_text(encoding="utf-8")
    y = re.sub(r"(?m)^train: .*$", "train: train_vnsplit.txt", y)
    y = re.sub(r"(?m)^val: .*$", "val: val_vn.txt", y)
    (root / "data_vnval.yaml").write_text("# val 100% ảnh VN, tách theo video (scripts/split_vn_val.py)\n" + y,
                                          encoding="utf-8")
    report["total_val"] = len(vfiles)
    report["train_lines"] = len(tr)
    (root / "metadata" / "val_vn.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"val_vn: {len(vfiles)} images · train_vnsplit: {len(tr)} lines (was "
          f"{len((root / 'train.txt').read_text().splitlines())})")


if __name__ == "__main__":
    main()
