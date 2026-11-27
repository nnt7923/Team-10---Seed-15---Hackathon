"""Download helpers for sources that allow scripted download. Everything else is manual
(registration required) — this script prints the instructions from configs/sources.yaml.

  python scripts/download.py --list
  python scripts/download.py coco2017_street            # annotations + only street-scene images
  python scripts/download.py roboflow_vn_traffic        # needs $env:ROBOFLOW_API_KEY and `pip install roboflow`

Downloads are large; check free disk space first (COCO annotations zip ≈ 250 MB, filtered
street images ≈ 6k × 160 KB ≈ 1 GB).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from vtb.common import load_yaml, map_label  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _get(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    tmp = dst.with_suffix(dst.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dst)


def coco(src: dict, raw: Path, limit: int | None):
    zip_url = src["download"]["files"][0]
    z = raw / "coco" / "annotations_trainval2017.zip"
    print(f"downloading {zip_url}")
    _get(zip_url, z)
    with zipfile.ZipFile(z) as zf:
        zf.extract("annotations/instances_train2017.json", raw / "coco")
    data = json.load(open(raw / "coco/annotations/instances_train2017.json", encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in data["categories"]}
    sel = src.get("select", {})
    per = {}
    for a in data["annotations"]:
        tok = map_label(cats[a["category_id"]], src["label_map"])
        if tok != "@ignore" and not a.get("iscrowd"):
            per.setdefault(a["image_id"], []).append(tok)
    keep = [im for im in data["images"]
            if len(per.get(im["id"], [])) >= sel.get("min_target_instances", 0)
            and any(t in sel.get("require_any", [t]) for t in per.get(im["id"], []))]
    keep = keep[: limit or src.get("max_images") or len(keep)]
    print(f"{len(keep)} street-scene images selected; downloading…")
    url = src["download"]["images_url"]
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(lambda im: _get(url.format(file_name=im["file_name"]),
                                    raw / "coco/train2017" / im["file_name"]), keep))


def roboflow(src: dict, raw: Path):
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        sys.exit("Set $env:ROBOFLOW_API_KEY first (Roboflow > Settings > API key).")
    from roboflow import Roboflow
    d = src["download"]
    proj = Roboflow(api_key=key).workspace(d["workspace"]).project(d["project"])
    proj.version(d["version"]).download(d["format"], location=str(raw / src["id"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="?")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--registry", default=str(ROOT / "configs/sources.yaml"))
    ap.add_argument("--raw", default=str(ROOT / "raw"))
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    reg = {s["id"]: s for s in load_yaml(a.registry)["sources"]}
    if a.list or not a.source:
        for s in reg.values():
            print(f"{s['id']:<26} {s['download']['method']:<9} {s.get('homepage') or ''}")
        return
    s = reg[a.source]
    m = s["download"]["method"]
    raw = Path(a.raw)
    if a.source.startswith("coco") and m == "url":
        coco(s, raw, a.limit)
    elif m == "roboflow":
        roboflow(s, raw)
    else:
        print(f"[{a.source}] manual download:\n  {s['download'].get('instructions')}\n  {s.get('homepage')}")


if __name__ == "__main__":
    main()
