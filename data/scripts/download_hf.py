"""Download public mirrors of the source datasets from Hugging Face (no registration needed).

  python scripts/download_hf.py mapillary cityscapes          # into raw/
  python scripts/download_hf.py --list

Mirrors used (checked 27/11/2026). They are third-party re-uploads: the ORIGINAL licenses apply
(Mapillary Vistas: CC BY-NC-SA 4.0; Cityscapes: Cityscapes license, non-commercial) and must be
cited in the report.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]

MIRRORS = {
    # 9,999 training images of Mapillary Vistas v2.0 + v2.0 panoptic labels (9,982 PNG + panoptic_2020.json)
    "mapillary": {
        "repo": "ali-sh07/Mapillary-dataset",
        "dst": "mapillary",
        "allow": ["training/images/*", "training/v2.0/panoptic/*", "config_v2.0.json", "LICENSE", "README"],
    },
    # Cityscapes fine annotations (polygons) + images, train + val only (test has no labels)
    "cityscapes": {
        "repo": "tackhwa/cityscapes",
        "dst": "",  # repo already has a top-level cityscapes/ folder
        "allow": ["cityscapes/gtFine/train/*_polygons.json", "cityscapes/gtFine/val/*_polygons.json",
                  "cityscapes/leftImg8bit/train/*", "cityscapes/leftImg8bit/val/*"],
    },
    # Vietnamese traffic-sign images (YOLO boxes, sign classes only) — VN domain, needs SAM + teacher
    "vn_signs": {
        "repo": "thanhhiepvos/vietnam_traffic_sign",
        "dst": "vn_traffic_sign_hf",
        "allow": ["archive/*", "README.md"],
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--raw", default=str(ROOT / "raw"))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    if a.list or not a.names:
        for k, v in MIRRORS.items():
            print(f"{k:<12} https://huggingface.co/datasets/{v['repo']}")
        return
    from huggingface_hub import snapshot_download
    for n in a.names:
        m = MIRRORS[n]
        dst = Path(a.raw) / m["dst"]
        dst.mkdir(parents=True, exist_ok=True)
        print(f"[{n}] {m['repo']} -> {dst}", flush=True)
        snapshot_download(repo_id=m["repo"], repo_type="dataset", local_dir=str(dst),
                          allow_patterns=m["allow"], max_workers=a.workers)
        print(f"[{n}] done", flush=True)


if __name__ == "__main__":
    main()
