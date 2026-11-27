"""GPU-assisted labeling stages. They run on 01_converted.jsonl, i.e. BEFORE `resolve`, so their
output goes through the same rider merge / de-overlap / size filter as human labels.

* sam_masks       : box-only sources (Zalo / VN sign boxes, Roboflow detect) -> masks via SAM box prompts.
* teacher_complete: classes a source never labeled are filled by a teacher model. The teacher can be
                    - an 11-class model trained on our data (teacher_label_map: identity), or
                    - a COCO-pretrained YOLO-seg (teacher_label_map maps COCO names -> tokens; `person`
                      becomes @person so resolve decides rider vs pedestrian with the BTC rule).
                    Without this, partially labeled images teach the model that e.g. cars are background.
Every auto-generated instance keeps origin + conf so reviewers can filter them, and statistics report
human vs auto labels.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from .common import map_label, mask_to_polys, polys_bbox

# COCO-pretrained teacher -> our tokens. COCO has no sign (only stop sign) / other_vehicle class,
# so those stay unlabeled and must come from the source itself.
COCO_TEACHER_MAP = {"person": "@person", "bicycle": "bicycle", "car": "car", "motorcycle": "motorcycle",
                    "bus": "bus", "truck": "truck", "traffic light": "traffic_light"}
TOKEN_CLASSES = {"@person": {"pedestrian", "motorcyclist", "cyclist"}}


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


def needs_mask(r: dict) -> bool:
    return any(not it.get("polys") and it.get("box") for it in r["instances"])


def sam_masks(records: list[dict], model_path: str, device=None) -> Counter:
    from ultralytics import SAM
    log = Counter()
    todo = [r for r in records if needs_mask(r)]
    if not todo:
        return log
    import cv2
    model = SAM(model_path)
    for k, r in enumerate(todo):
        boxed = [it for it in r["instances"] if not it.get("polys") and it.get("box")]
        # SAM returns masks at input resolution: N boxes x a 4000px photo can fill a 4 GB GPU, so feed a
        # downscaled copy (export keeps <= 1280 px anyway) and prompt at most 16 boxes per call.
        img = cv2.imread(r["image"])
        if img is None:
            continue
        s = min(1.0, 1600 / max(img.shape[:2]))
        if s < 1.0:
            img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
        polys = []
        for i in range(0, len(boxed), 16):
            chunk = [[v * s for v in it["box"]] for it in boxed[i:i + 16]]
            res = model(img, bboxes=chunk, device=device, verbose=False)[0]
            got = _masks_to_polys(res, img.shape[1], img.shape[0])
            got += [[]] * (len(chunk) - len(got))
            polys += [[(np.asarray(p) / s).round(2).tolist() for p in ps] for ps in got]
        for it, ps in zip(boxed, polys + [[]] * (len(boxed) - len(polys))):
            if ps:
                it["polys"], it["origin"] = ps, "sam_box"
                log["sam_mask_ok"] += 1
            else:
                it["cls"] = "@ignore"
                log["sam_mask_empty"] += 1
        r["instances"] = [it for it in r["instances"] if it["cls"] != "@ignore"]
        if k % 200 == 0:
            print(f"[sam] {k}/{len(todo)}", flush=True)
    # free VRAM before the teacher runs in the same process (4 GB laptop GPUs otherwise spill to
    # shared memory and the teacher slows down ~10x)
    del model
    try:
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass
    return log


def teacher_complete(records: list[dict], class_names: list[str], model_path: str, conf: float = 0.5,
                     label_map: dict | None = None, device=None, imgsz: int = 1024) -> Counter:
    """Add teacher predictions for classes the record's source did not label."""
    from ultralytics import YOLO
    log = Counter()
    todo = [r for r in records if set(class_names) - set(r.get("labeled_classes") or []) and not r.get("completed")]
    if not todo:
        return log
    model = YOLO(model_path)
    tnames = model.names
    if label_map is None:  # 11-class teacher: identity
        label_map = {n: n for n in class_names}
    for k, r in enumerate(todo):
        labeled = set(r.get("labeled_classes") or [])
        res = model(r["image"], conf=conf, retina_masks=True, imgsz=imgsz, device=device, verbose=False)[0]
        covered = set()
        if res.boxes is not None and len(res.boxes):
            cls_ids = res.boxes.cls.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()
            boxes = res.boxes.xyxy.cpu().numpy()
            polys = _masks_to_polys(res, r["width"], r["height"])
            existing = [(it["cls"], polys_bbox(it["polys"]) if it.get("polys") else it.get("box"))
                        for it in r["instances"]]
            for j, cid in enumerate(cls_ids):
                name = tnames[int(cid)]
                tok = map_label(name, label_map)
                if tok == "@ignore" or j >= len(polys) or not polys[j]:
                    continue
                targets = TOKEN_CLASSES.get(tok, {tok})
                if targets & labeled:  # source already labels this class -> never overwrite humans
                    continue
                b = boxes[j].tolist()
                if any(c == tok and eb and _box_iou(b, eb) > 0.5 for c, eb in existing):
                    continue
                r["instances"].append({"raw": f"teacher:{name}", "cls": tok, "polys": polys[j],
                                       "origin": "teacher", "conf": round(float(confs[j]), 3)})
                log[f"teacher_added_{tok}"] += 1
        for tok in set(label_map.values()):
            covered |= TOKEN_CLASSES.get(tok, {tok})
        r["labeled_classes"] = sorted((labeled | covered) & set(class_names))
        r["completed"] = True
        r["needs_review"] = True
        log["images_completed"] += 1
        if k % 200 == 0:
            print(f"[complete] {k}/{len(todo)}", flush=True)
    return log
