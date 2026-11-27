"""GPU-assisted labeling (optional stages; need ultralytics + a GPU).

* sam_masks       : box-only sources (Zalo, Roboflow detect) -> masks via SAM box prompts.
* teacher_complete: partially-labeled images (COCO has no signs, BDD has no lights/signs, Zalo
                    has ONLY signs, self-collected frames have nothing) get the missing classes
                    from a strong teacher (e.g. yolo26x-seg trained on the fully-labeled pool).
                    Without this, those images teach the model that signs are background.
Every auto-generated instance keeps origin + conf so reviewers (CVAT / Label Studio) can
filter them and so statistics can report human vs auto labels.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from .common import mask_to_polys, polys_bbox
from .resolve import deoverlap


def _box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _masks_to_polys(result, W, H) -> list[list]:
    out = []
    if result.masks is None:
        return out
    data = result.masks.data.cpu().numpy()
    mh, mw = data.shape[1:]
    s = mw / W
    for m in data:
        out.append(mask_to_polys(m > 0.5, scale=s, min_area=4.0))
    return out


def sam_masks(records: list[dict], model_path: str, device=None) -> Counter:
    from ultralytics import SAM
    log = Counter()
    todo = [r for r in records if r.get("needs_mask")]
    if not todo:
        return log
    model = SAM(model_path)
    for r in todo:
        boxed = [it for it in r["instances"] if not it.get("polys") and it.get("box")]
        if not boxed:
            r["needs_mask"] = False
            continue
        res = model(r["image"], bboxes=[it["box"] for it in boxed], device=device, verbose=False)[0]
        polys = _masks_to_polys(res, r["width"], r["height"])
        for it, ps in zip(boxed, polys):
            if ps:
                it["polys"], it["origin"] = ps, "sam_box"
                log["sam_mask_ok"] += 1
            else:
                it["cls"] = "@ignore"
                log["sam_mask_empty"] += 1
        r["instances"] = [it for it in r["instances"] if it["cls"] != "@ignore"]
        deoverlap(r, log)  # SAM masks of neighbouring boxes overlap -> nearer object keeps pixels
        r["counts"] = {c: 0 for c in r["counts"]}
        for it in r["instances"]:
            r["counts"][it["cls"]] += 1
        r["needs_mask"] = False
    return log


def teacher_complete(records: list[dict], class_names: list[str], model_path: str, conf: float = 0.5,
                     device=None) -> Counter:
    from ultralytics import YOLO
    log = Counter()
    todo = [r for r in records if r.get("missing_classes") and not r.get("completed")]
    if not todo:
        return log
    model = YOLO(model_path)
    for r in todo:
        missing = set(r["missing_classes"])
        res = model(r["image"], conf=conf, retina_masks=True, device=device, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            r["completed"] = True
            continue
        cls_ids = res.boxes.cls.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()
        boxes = res.boxes.xyxy.cpu().numpy()
        polys = _masks_to_polys(res, r["width"], r["height"])
        existing = [(it["cls"], polys_bbox(it["polys"]) if it.get("polys") else it.get("box"))
                    for it in r["instances"]]
        for k, cid in enumerate(cls_ids):
            name = class_names[cid]
            if name not in missing or not polys[k]:
                continue
            b = boxes[k].tolist()
            if any(c == name and eb and _box_iou(b, eb) > 0.5 for c, eb in existing):
                continue
            r["instances"].append({"raw": f"teacher:{name}", "cls": name, "polys": polys[k],
                                   "origin": "teacher", "conf": round(float(confs[k]), 3)})
            r["counts"][name] = r["counts"].get(name, 0) + 1
            log[f"teacher_added_{name}"] += 1
        deoverlap(r, log)
        r["counts"] = {c: sum(1 for it in r["instances"] if it["cls"] == c) for c in class_names}
        r["completed"] = True
        r["needs_review"] = True
        log["images_completed"] += 1
    return log
