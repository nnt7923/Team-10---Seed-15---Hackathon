"""Raw-data analysis: what is in each downloaded source BEFORE the shared pipeline runs.

Two levels, so it stays fast on 10k+ high-res images:
  1. Label vocabulary over the WHOLE source (annotation files only, no pixels): every raw label,
     its count, and the token it maps to — including labels that match no rule (silently dropped).
  2. A random SAMPLE of images converted + resolved exactly like the real pipeline (rider merge,
     de-overlap, 1.25% rule) -> estimated final class counts, object sizes, resolution, brightness.
     If work/converted/<source>.jsonl exists it is used in full instead of a sample.
"""
from __future__ import annotations

import base64
import copy
import fnmatch
import json
import random
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path

import cv2
import numpy as np

from .common import map_label, polys_bbox, read_jsonl
from .converters import IMG_EXT, _iter_images, _yolo_names, convert_source
from .resolve import resolve_record

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948",
           "#52514e", "#00a3a3", "#a35200"]


# --------------------------------------------------------------------------- vocabulary
def _image_stems(root: Path) -> set[str]:
    return {p.stem for p in _iter_images(root)} if root.exists() else set()


def label_vocab(src: dict, data_root: Path) -> tuple[Counter, int]:
    """(raw label -> instance count over the whole source, #images with a label file)."""
    fmt, P = src["format"], src["paths"]
    vocab, n_img = Counter(), 0
    if fmt in ("yolo_seg", "yolo_det"):
        names = _yolo_names(src, data_root)
        for f in (data_root / P["labels"]).rglob("*.txt"):
            if "labels" not in f.parts and P["labels"] == P["images"]:
                continue  # README.*.txt etc. next to the images
            n_img += 1
            for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.split():
                    cid = line.split()[0]
                    vocab[names.get(cid, cid)] += 1
    elif fmt in ("coco", "coco_bbox") and isinstance(P["annotations"], list):
        for a in P["annotations"]:
            sub = dict(src, paths=dict(P, annotations=a, images=str(Path(a).parent)))
            v, n = label_vocab(sub, data_root)
            vocab.update(v)
            n_img += n
    elif fmt in ("coco", "coco_bbox", "coco_panoptic"):
        data = json.load(open(data_root / P["annotations"], encoding="utf-8"))
        cats = {c["id"]: c["name"] for c in data["categories"]}
        have = _image_stems(data_root / P["images"])
        ok_ids = {im["id"] for im in data["images"] if Path(im["file_name"]).stem in have}
        n_img = len(ok_ids)
        if fmt == "coco_panoptic":
            for a in data["annotations"]:
                if a["image_id"] in ok_ids:
                    for s in a["segments_info"]:
                        vocab[cats.get(s["category_id"], str(s["category_id"]))] += 1
        else:
            for a in data.get("annotations", []):
                if a["image_id"] in ok_ids:
                    vocab[cats.get(a["category_id"], str(a["category_id"]))] += 1
    elif fmt == "polygons_json":
        a_suffix = src.get("name_replace", [".json", ".jpg"])[0]
        for f in (data_root / P["annotations"]).rglob(f"*{a_suffix}"):
            n_img += 1
            for o in json.load(open(f, encoding="utf-8")).get("objects", []):
                if not o.get("deleted"):
                    vocab[o["label"]] += 1
    elif fmt == "nuimages":
        from .converters import _nuimages_tables
        root = data_root / P["root"]
        for _split, cats, _attrs, sd, objs in _nuimages_tables(root):
            for tok, x in sd.items():
                if (root / x["filename"]).exists():
                    n_img += 1
                    for o in objs.get(tok, []):
                        vocab[cats.get(o["category_token"], "")] += 1
    elif fmt == "bdd_scalabel":
        frames = json.load(open(data_root / P["annotations"], encoding="utf-8"))
        frames = frames.get("frames", frames) if isinstance(frames, dict) else frames
        for fr in frames:
            n_img += 1
            for lb in fr.get("labels") or []:
                vocab[lb.get("category", "")] += 1
    return vocab, n_img


def has_rule(raw: str, label_map: dict) -> bool:
    if str(raw) in label_map:
        return True
    return any(any(c in k for c in "*?[") and fnmatch.fnmatchcase(str(raw), k) for k in label_map)


# --------------------------------------------------------------------------- disk
def disk_bytes(src: dict, data_root: Path) -> int:
    seen, tot = set(), 0
    paths = []
    for key in ("images", "labels", "annotations", "panoptic_png", "root"):
        v = src["paths"].get(key)
        paths += v if isinstance(v, list) else ([v] if v else [])
    for p in paths:
        p = data_root / p
        items = [p] if p.is_file() else (p.rglob("*") if p.exists() else [])
        for f in items:
            if f.is_file() and f not in seen:
                seen.add(f)
                tot += f.stat().st_size
    return tot


# --------------------------------------------------------------------------- per-source analysis
def _luma(path: str) -> float | None:
    from PIL import Image
    try:
        with Image.open(path) as im:
            im.draft("L", (128, 128))
            return float(np.asarray(im.convert("L").resize((64, 64))).mean())
    except Exception:
        return None


def _thumb(rec: dict, class_index: dict, width: int = 420) -> str:
    img = cv2.imread(rec["image"])
    if img is None:
        return ""
    s = width / img.shape[1]
    img = cv2.resize(img, (width, int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    over = img.copy()
    for it in rec["instances"]:
        c = class_index.get(it["cls"])
        if c is None or not it.get("polys"):
            continue
        h = PALETTE[c].lstrip("#")
        col = (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))
        for p in it["polys"]:
            pts = np.round(np.asarray(p) * s).astype(np.int32)
            cv2.fillPoly(over, [pts], col)
            cv2.polylines(img, [pts], True, col, 1, cv2.LINE_AA)
    out = cv2.addWeighted(over, 0.45, img, 0.55, 0)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 78])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode() if ok else ""


def analyze_source(src: dict, data_root: Path, classes, conv_dir: Path, sample: int, n_thumbs: int = 3) -> dict:
    vocab, n_lab = label_vocab(src, data_root)
    lm = src.get("label_map") or {}
    vocab_rows = []
    tokens = Counter()
    for raw, n in vocab.most_common():
        tok = map_label(raw, lm)
        tokens[tok] += n
        vocab_rows.append({"raw": raw, "token": tok, "count": n, "rule": has_rule(raw, lm)})
    n_images = sum(1 for _ in _iter_images(data_root / src["paths"]["images"])) \
        if (data_root / src["paths"]["images"]).exists() else 0

    cache = conv_dir / f"{src['id']}.jsonl"
    if cache.exists():
        recs, mode = list(read_jsonl(cache)), "full (work/converted)"
    else:
        s2 = dict(src, _sample=sample)
        recs, mode = list(islice(convert_source(s2, data_root), sample)), f"sample {sample}"
    n_conv = len(recs)
    origin = Counter(it.get("origin", "human") for r in recs for it in r["instances"])
    raw_inst = sum(len(r["instances"]) for r in recs)

    rlog = Counter()
    resolved = [resolve_record(copy.deepcopy(r), classes, classes.cfg, rlog) for r in recs]
    scale = (n_images / n_conv) if n_conv and not cache.exists() else 1.0
    est = Counter()
    for r in resolved:
        for c, k in r["counts"].items():
            est[c] += k
    est = {c: int(round(est[c] * scale)) for c in classes.names}

    W = np.array([r["width"] for r in recs]) if recs else np.zeros(1)
    H = np.array([r["height"] for r in recs]) if recs else np.zeros(1)
    per_img = np.array([sum(r["counts"].values()) for r in resolved]) if resolved else np.zeros(1)
    short = []
    for r in recs:
        L = max(r["width"], r["height"])
        for it in r["instances"]:
            b = polys_bbox(it["polys"]) if it.get("polys") else it.get("box")
            if b:
                short.append(min(b[2] - b[0], b[3] - b[1]) / L)
    short = np.array(short) if short else np.zeros(1)
    rng = random.Random(15)
    lum_s = [x for x in (_luma(r["image"]) for r in rng.sample(recs, min(150, len(recs)))) if x is not None]
    night = float(np.mean(np.array(lum_s) < 70)) if lum_s else 0.0
    dropped_small = sum(v for k, v in rlog.items() if k.startswith("drop_short_side"))
    cand = [r for r in resolved if len(r["instances"]) >= 4] or resolved
    thumbs = [_thumb(r, classes.index) for r in rng.sample(cand, min(n_thumbs, len(cand)))]
    labeled = set(src.get("labeled_classes") or [])
    if recs and any(r.get("completed") for r in recs):  # after SAM/teacher: classes now covered per record
        labeled = set.intersection(*[set(r.get("labeled_classes") or []) for r in recs])
    return {
        "id": src["id"], "name": src.get("name"), "tier": src.get("tier"), "domain": src.get("domain"),
        "homepage": src.get("homepage"), "license": src.get("license"), "format": src["format"],
        "citation": src.get("citation"),
        "labeled_classes": sorted(labeled), "accept_missing": src.get("accept_missing") or [],
        "teacher_completed": bool(recs and any(r.get("completed") for r in recs)),
        "missing_classes": [c for c in classes.names if c not in labeled],
        "n_images": n_images, "n_label_files": n_lab, "disk_bytes": disk_bytes(src, data_root),
        "vocab": vocab_rows, "tokens": dict(tokens), "unmapped": [v for v in vocab_rows if not v["rule"]],
        "analysis_mode": mode, "n_analyzed": n_conv, "estimated_classes": est,
        "label_origin": dict(origin), "raw_instances_analyzed": raw_inst,
        "resolve_log": dict(rlog), "dropped_small": dropped_small,
        "res_median": [int(np.median(W)), int(np.median(H))], "res_min": [int(W.min()), int(H.min())],
        "res_max": [int(W.max()), int(H.max())],
        "inst_per_img_median": float(np.median(per_img)), "inst_per_img_p90": float(np.percentile(per_img, 90)),
        "short_side_pct": {"p10": float(np.percentile(short, 10) * 100), "p50": float(np.median(short) * 100),
                           "below_1_25": float(np.mean(short < 0.0125) * 100)},
        "night_share": night, "thumbs": [t for t in thumbs if t],
    }


def analyze(sources: list[dict], data_root: Path, classes, conv_dir: Path, sample: int = 300) -> dict:
    res = []
    for s in sources:
        if not (data_root / s["paths"]["images"]).exists():
            res.append({"id": s["id"], "name": s.get("name"), "missing_on_disk": True,
                        "homepage": s.get("homepage"), "license": s.get("license")})
            print(f"[analyze] {s['id']}: not on disk, skipped", flush=True)
            continue
        print(f"[analyze] {s['id']} ...", flush=True)
        res.append(analyze_source(s, data_root, classes, conv_dir, sample))
    return {"sources": res}
