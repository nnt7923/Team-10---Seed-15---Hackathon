"""Source-format readers -> unified records (see common.py for the schema).

Every converter yields records whose instances carry a *token* (`cls`): one of the
11 target classes or a special token (@rider, @person, ...) resolved later by merge.py.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .common import image_size, map_label, mask_to_polys, rasterize, rle_to_mask

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
WORK_MAX_SIDE = 1600  # resolution used when rasterizing occlusion / panoptic masks


# --------------------------------------------------------------------------- helpers
def _base_record(src: dict, img: Path, w: int, h: int, rel_stem: str, group: str | None,
                 orig_split: str | None = None, attrs: dict | None = None) -> dict:
    attrs = dict(attrs or {})
    for key, rx in (src.get("attr_from_path") or {}).items():
        m = re.search(rx, img.as_posix())
        if m:
            attrs[key] = m.group(1)
    if src.get("group_regex"):
        m = re.search(src["group_regex"], rel_stem)
        if m:
            group = m.group(1)
    labeled = list(src.get("labeled_classes") or [])
    return {
        "uid": f"{src['id']}/{rel_stem}",
        "source": src["id"],
        "image": str(img.resolve()),
        "width": int(w),
        "height": int(h),
        "orig_id": Path(rel_stem).name,
        "orig_split": orig_split,
        "group": f"{src['id']}/{group or rel_stem}",
        "domain": src.get("domain", "global"),
        "tier": src.get("tier"),
        "attrs": attrs,
        "labeled_classes": labeled,
        "accept_missing": list(src.get("accept_missing") or []),
        "ridden_default": bool(src.get("ridden_default")),
        "instances": [],
    }


def _inst(raw, tok, polys=None, box=None, origin="human", conf=1.0) -> dict:
    d = {"raw": str(raw), "cls": tok, "origin": origin, "conf": conf}
    if polys is not None:
        d["polys"] = polys
    if box is not None:
        d["box"] = [float(v) for v in box]
    return d


def _iter_images(root: Path):
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in IMG_EXT:
            yield p


def _yolo_names(src: dict, data_root: Path) -> dict:
    """Return {index(str): name}. Uses data.yaml if given, else identity indices."""
    dy = src["paths"].get("data_yaml")
    if dy and (data_root / dy).exists():
        import yaml
        names = yaml.safe_load(open(data_root / dy, encoding="utf-8"))["names"]
        if isinstance(names, list):
            names = dict(enumerate(names))
        return {str(k): str(v) for k, v in names.items()}
    nf = src["paths"].get("names_file")  # plain text, one class name per line (index = line number)
    if nf and (data_root / nf).exists():
        lines = [l.strip() for l in open(data_root / nf, encoding="utf-8") if l.strip()]
        return {str(i): n for i, n in enumerate(lines)}
    return {}


def _select_ok(src: dict, instances: list[dict]) -> bool:
    sel = src.get("select")
    if not sel:
        return True
    toks = [i["cls"] for i in instances if i["cls"] != "@ignore"]
    if len(toks) < sel.get("min_target_instances", 0):
        return False
    req = sel.get("require_any")
    return not req or any(t in req for t in toks)


# --------------------------------------------------------------------------- YOLO
def convert_yolo(src: dict, data_root: Path, seg: bool = True) -> Iterator[dict]:
    img_root = data_root / src["paths"]["images"]
    lbl_root = data_root / src["paths"]["labels"]
    names = _yolo_names(src, data_root)
    imgs = list(_iter_images(img_root))
    if src.get("_sample"):  # analysis mode: random subset
        import random
        random.Random(15).shuffle(imgs)
        imgs = imgs[: src["_sample"] * 2]
    for img in imgs:
        rel = img.relative_to(img_root).with_suffix("")
        lbl = lbl_root / rel.with_suffix(".txt")
        if not lbl.exists():  # Roboflow / Ultralytics layout: .../images/x.jpg -> .../labels/x.txt
            parts = list(img.with_suffix(".txt").parts)
            if "images" in parts:
                parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
                lbl = Path(*parts)
        w, h = image_size(img)
        group = rel.parent.as_posix() if rel.parent.as_posix() != "." else None
        rec = _base_record(src, img, w, h, rel.as_posix(), group)
        if lbl.exists():
            for line in lbl.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cid = parts[0]
                raw = names.get(cid, cid)
                tok = map_label(raw, src["label_map"])
                if tok == "@ignore":
                    tok = map_label(cid, src["label_map"])  # allow index-keyed maps too
                vals = np.asarray(parts[1:], np.float64)
                if seg and len(vals) >= 6:
                    poly = (vals.reshape(-1, 2) * [w, h]).round(2).tolist()
                    rec["instances"].append(_inst(raw, tok, polys=[poly]))
                else:
                    cx, cy, bw, bh = vals[:4] * [w, h, w, h]
                    rec["instances"].append(_inst(raw, tok, box=[cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]))
        if _select_ok(src, rec["instances"]):
            yield rec


def convert_images_only(src: dict, data_root: Path) -> Iterator[dict]:
    img_root = data_root / src["paths"]["images"]
    for img in _iter_images(img_root):
        rel = img.relative_to(img_root).with_suffix("")
        w, h = image_size(img)
        group = rel.parent.as_posix() if rel.parent.as_posix() != "." else None
        yield _base_record(src, img, w, h, rel.as_posix(), group)


# --------------------------------------------------------------------------- COCO
def convert_coco(src: dict, data_root: Path, bbox_only: bool = False) -> Iterator[dict]:
    anns = src["paths"]["annotations"]
    if isinstance(anns, list):  # several split files; images live next to each json
        for a in anns:
            sub = dict(src, paths=dict(src["paths"], annotations=a, images=str(Path(a).parent)))
            yield from convert_coco(sub, data_root, bbox_only)
        return
    ann_path = data_root / anns
    img_root = data_root / src["paths"]["images"]
    data = json.load(open(ann_path, encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in data["categories"]}
    by_img = defaultdict(list)
    for a in data.get("annotations", []):
        by_img[a["image_id"]].append(a)
    for im in data["images"]:
        img = img_root / im["file_name"]
        w, h = im.get("width"), im.get("height")
        insts = []
        for a in by_img.get(im["id"], []):
            raw = cats.get(a["category_id"], str(a["category_id"]))
            tok = "@ignore" if a.get("iscrowd") else map_label(raw, src["label_map"])
            if tok == "@ignore":
                continue
            seg = a.get("segmentation")
            if bbox_only or not seg:
                x, y, bw, bh = a["bbox"]
                insts.append(_inst(raw, tok, box=[x, y, x + bw, y + bh]))
            elif isinstance(seg, list):
                polys = [np.asarray(s, np.float64).reshape(-1, 2).round(2).tolist() for s in seg if len(s) >= 6]
                if polys:
                    insts.append(_inst(raw, tok, polys=polys))
            else:  # RLE
                polys = mask_to_polys(rle_to_mask(seg))
                if polys:
                    insts.append(_inst(raw, tok, polys=polys))
        probe = [{"cls": i["cls"]} for i in insts]
        if not _select_ok(src, probe):
            continue
        if not img.exists():
            continue  # filtered subsets may not have every image downloaded
        if not w or not h:
            w, h = image_size(img)
        rel = Path(im["file_name"]).with_suffix("").as_posix()
        if src.get("prefix_split"):  # same file names can repeat across split folders -> keep uids unique
            rel = Path(src["paths"]["images"]).as_posix().replace("/", "_") + "/" + rel
        rec = _base_record(src, img, w, h, rel, None, orig_split=src.get("orig_split"))
        rec["instances"] = insts
        yield rec


# ------------------------------------------------------- Cityscapes / IDD / Mapillary
def convert_polygons_json(src: dict, data_root: Path) -> Iterator[dict]:
    ann_root = data_root / src["paths"]["annotations"]
    img_root = data_root / src["paths"]["images"]
    a_suffix, i_suffix = src.get("name_replace", [".json", ".jpg"])
    occl = src.get("occlusion_order", "none") == "back_to_front"
    anns = sorted(ann_root.rglob(f"*{a_suffix}"))
    if src.get("_sample"):  # analysis mode: random subset
        import random
        random.Random(15).shuffle(anns)
        anns = anns[: src["_sample"] * 2]
    for ann in anns:
        rel_ann = ann.relative_to(ann_root).as_posix()
        img = img_root / (rel_ann[: -len(a_suffix)] + i_suffix)
        if not img.exists():
            continue
        d = json.load(open(ann, encoding="utf-8"))
        w = d.get("imgWidth") or d.get("width")
        h = d.get("imgHeight") or d.get("height")
        if not w or not h:
            w, h = image_size(img)
        objs = [o for o in d.get("objects", []) if not o.get("deleted") and len(o.get("polygon", [])) >= 3]
        rel_stem = rel_ann[: -len(a_suffix)]
        parts = Path(rel_stem).parts
        rec = _base_record(src, img, w, h, rel_stem, "/".join(parts[:-1]) or None,
                           orig_split=parts[0] if len(parts) > 1 else None)
        toks = [map_label(o["label"], src["label_map"]) for o in objs]
        if occl:
            s = min(1.0, WORK_MAX_SIDE / max(w, h))
            W, H = int(round(w * s)), int(round(h * s))
            idmap = np.zeros((H, W), np.int32)
            for k, o in enumerate(objs):  # paint back -> front: later objects occlude earlier ones
                pts = np.round(np.asarray(o["polygon"], np.float64) * s).astype(np.int32)
                cv2.fillPoly(idmap, [pts], k + 1)
            for k, (o, tok) in enumerate(zip(objs, toks)):
                if tok == "@ignore":
                    continue
                pts = np.asarray(o["polygon"], np.float64) * s
                x0, y0 = np.floor(pts.min(0)).astype(int).clip(0)
                x1, y1 = (np.ceil(pts.max(0)).astype(int) + 1)
                x1, y1 = min(x1, W), min(y1, H)
                crop = (idmap[y0:y1, x0:x1] == k + 1).astype(np.uint8)
                polys = mask_to_polys(crop, scale=s, offset=(x0 / s, y0 / s), min_area=2.0)
                if polys:
                    rec["instances"].append(_inst(o["label"], tok, polys=polys))
        else:
            for o, tok in zip(objs, toks):
                if tok != "@ignore":
                    rec["instances"].append(_inst(o["label"], tok, polys=[o["polygon"]]))
        if _select_ok(src, rec["instances"]):
            yield rec


# --------------------------------------------------------------------------- panoptic
def _panoptic_one(args):
    """Worker: one panoptic PNG -> record. The id map is downscaled (nearest) to WORK_MAX_SIDE and each
    segment is processed inside its own bbox, so 4000x3000 Mapillary frames take ~0.1 s instead of seconds."""
    src, img, png, rel, segs, cats = args
    bgr = cv2.imread(str(png), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    s = min(1.0, WORK_MAX_SIDE / max(w, h))
    if s < 1.0:
        bgr = cv2.resize(bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_NEAREST)
    rgb = bgr[:, :, ::-1].astype(np.int64)
    ids = rgb[..., 0] + 256 * rgb[..., 1] + 65536 * rgb[..., 2]
    Hs, Ws = ids.shape
    rec = _base_record(src, img, w, h, rel, str(Path(rel).parent) if "/" in rel else None)
    split_comp = set(src.get("split_components") or [])
    for seg in segs:
        raw = cats.get(seg["category_id"], str(seg["category_id"]))
        tok = "@ignore" if seg.get("iscrowd") else map_label(raw, src["label_map"])
        if tok == "@ignore":
            continue
        if seg.get("bbox"):
            bx, by, bw, bh = seg["bbox"]
            x0, y0 = max(0, int(bx * s) - 1), max(0, int(by * s) - 1)
            x1, y1 = min(Ws, int((bx + bw) * s) + 2), min(Hs, int((by + bh) * s) + 2)
        else:
            x0, y0, x1, y1 = 0, 0, Ws, Hs
        m = (ids[y0:y1, x0:x1] == seg["id"]).astype(np.uint8)
        if not m.any():
            continue
        off = (x0 / s, y0 / s)
        if tok in split_comp:  # stuff-class: one segment for all signs -> split into components
            n, cc = cv2.connectedComponents(m)
            parts = [cc == c for c in range(1, n)]
        else:
            parts = [m]
        for part in parts:
            polys = mask_to_polys(part, scale=s, offset=off, min_area=2.0)
            if polys:
                rec["instances"].append(_inst(raw, tok, polys=polys))
    return rec if _select_ok(src, rec["instances"]) else None


def convert_coco_panoptic(src: dict, data_root: Path, workers: int = 8) -> Iterator[dict]:
    from concurrent.futures import ProcessPoolExecutor
    ann_path = data_root / src["paths"]["annotations"]
    png_root = data_root / src["paths"]["panoptic_png"]
    img_root = data_root / src["paths"]["images"]
    data = json.load(open(ann_path, encoding="utf-8"))
    field = src.get("category_field", "name")
    cats = {c["id"]: c.get(field) or c["name"] for c in data["categories"]}
    images = {im["id"]: im for im in data["images"]}
    png_index = {p.name: p for p in png_root.rglob("*.png")}
    img_index = {p.stem: p for p in _iter_images(img_root)}
    jobs = []
    for a in data["annotations"]:
        im = images[a["image_id"]]
        img = img_root / im["file_name"]
        if not img.exists():
            img = img_index.get(Path(im["file_name"]).stem)
            if img is None:
                continue
        png = png_index.get(Path(a["file_name"]).name)
        if png is None:
            continue
        rel = img.relative_to(img_root).with_suffix("").as_posix()
        jobs.append((src, img, png, rel, a["segments_info"], cats))
    del data
    if src.get("_sample"):  # analysis mode: random subset only
        import random
        random.Random(15).shuffle(jobs)
        jobs = jobs[: src["_sample"]]
    with ProcessPoolExecutor(workers) as ex:
        for rec in ex.map(_panoptic_one, jobs, chunksize=8):
            if rec is not None:
                yield rec


# --------------------------------------------------------------------------- BDD100K
def convert_bdd_scalabel(src: dict, data_root: Path) -> Iterator[dict]:
    ann_path = data_root / src["paths"]["annotations"]
    img_root = data_root / src["paths"]["images"]
    frames = json.load(open(ann_path, encoding="utf-8"))
    if isinstance(frames, dict):
        frames = frames.get("frames", [])
    for fr in frames:
        img = img_root / fr["name"]
        if not img.exists():
            continue
        w, h = image_size(img)
        fa = fr.get("attributes") or {}
        attrs = {k: fa[k] for k in ("timeofday", "weather", "scene") if k in fa}
        rel = Path(fr["name"]).with_suffix("").as_posix()
        rec = _base_record(src, img, w, h, rel, fr.get("videoName"), attrs=attrs)
        for lb in fr.get("labels") or []:
            raw = lb.get("category", "")
            la = lb.get("attributes") or {}
            tok = "@ignore" if la.get("crowd") else map_label(raw, src["label_map"])
            if tok == "@ignore":
                continue
            polys = []
            for p in lb.get("poly2d") or []:
                v = p.get("vertices") or []
                if len(v) >= 3:
                    polys.append([[float(x), float(y)] for x, y in v])
            if not polys and lb.get("rle"):
                polys = mask_to_polys(rle_to_mask(lb["rle"]))
            if polys:
                rec["instances"].append(_inst(raw, tok, polys=polys))
        if _select_ok(src, rec["instances"]):
            yield rec


# --------------------------------------------------------------------------- nuImages
def _nuimages_tables(root: Path):
    """Yield (split_dir, categories, attributes, key-frame sample_data, objects grouped by sample_data)."""
    for split in ("v1.0-train", "v1.0-val"):
        d = root / split
        if not (d / "object_ann.json").exists():
            continue
        cats = {c["token"]: c["name"] for c in json.load(open(d / "category.json", encoding="utf-8"))}
        attrs = {a["token"]: a["name"] for a in json.load(open(d / "attribute.json", encoding="utf-8"))}
        sd = {x["token"]: x for x in json.load(open(d / "sample_data.json", encoding="utf-8")) if x.get("is_key_frame")}
        objs = defaultdict(list)
        for o in json.load(open(d / "object_ann.json", encoding="utf-8")):
            objs[o["sample_data_token"]].append(o)
        yield split, cats, attrs, sd, objs


def convert_nuimages(src: dict, data_root: Path) -> Iterator[dict]:
    """nuImages v1.0: instance masks as base64-wrapped COCO RLE; bicycles / motorcycles carry the attribute
    cycle.with_rider / cycle.without_rider, which marks the instance `ridden` (-> class 5/7 even if the
    rider's own mask is not matched)."""
    import base64
    root = data_root / src["paths"]["root"]
    for split, cats, attrs, sd, objs in _nuimages_tables(root):
        items = list(sd.items())
        if src.get("_sample"):
            import random
            random.Random(15).shuffle(items)
            items = items[: src["_sample"] * 3]
        for tok, x in items:
            img = root / x["filename"]
            if not img.exists():
                continue
            rel = Path(x["filename"]).with_suffix("").as_posix()
            rec = _base_record(src, img, x["width"], x["height"], rel, None, orig_split=split)
            for o in objs.get(tok, []):
                raw = cats.get(o["category_token"], "")
                tok_cls = map_label(raw, src["label_map"])
                if tok_cls == "@ignore" or not o.get("mask"):
                    continue
                m = dict(o["mask"])
                m["counts"] = base64.b64decode(m["counts"]).decode("utf-8")
                polys = mask_to_polys(rle_to_mask(m), min_area=4.0)
                if not polys:
                    continue
                inst = _inst(raw, tok_cls, polys=polys)
                an = {attrs.get(a, "") for a in o.get("attribute_tokens", [])}
                if "cycle.with_rider" in an:
                    inst["ridden"] = True
                rec["instances"].append(inst)
            if _select_ok(src, rec["instances"]):
                yield rec


# --------------------------------------------------------------------------- dispatch
def convert_source(src: dict, data_root: Path) -> Iterator[dict]:
    fmt = src["format"]
    if fmt == "yolo_seg":
        yield from convert_yolo(src, data_root, seg=True)
    elif fmt == "yolo_det":
        yield from convert_yolo(src, data_root, seg=False)
    elif fmt == "coco":
        yield from convert_coco(src, data_root)
    elif fmt == "coco_bbox":
        yield from convert_coco(src, data_root, bbox_only=True)
    elif fmt == "coco_panoptic":
        yield from convert_coco_panoptic(src, data_root)
    elif fmt == "polygons_json":
        yield from convert_polygons_json(src, data_root)
    elif fmt == "bdd_scalabel":
        yield from convert_bdd_scalabel(src, data_root)
    elif fmt == "nuimages":
        yield from convert_nuimages(src, data_root)
    elif fmt == "images_only":
        yield from convert_images_only(src, data_root)
    else:
        raise ValueError(f"Unknown format {fmt!r} for source {src['id']}")
