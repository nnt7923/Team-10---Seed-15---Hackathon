"""Resolve special tokens into the 11 target classes, then filter.

Rider merge (the class 4/5 and 6/7 problem):
  * explicit riders (@rider, @rider_moto, @rider_bike) and ambiguous persons (@person)
    are matched to the two-wheeler (motorcycle / bicycle) they overlap most;
  * a two-wheeler with >=1 rider becomes motorcyclist / cyclist (VN: 2-3 people per bike is common);
  * the output mask follows classes.yaml rider_merge.mode (union | vehicle | rider);
  * two-wheelers with no rider stay motorcycle / bicycle;
  * unmatched @person -> pedestrian; unmatched @rider_moto/@rider_bike keep their rider mask
    as motorcyclist / cyclist; unmatched generic @rider is dropped (counted in the log).
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from ultralytics.data.converter import merge_multi_segment

from .common import mask_to_polys, polys_area, polys_bbox, rasterize

TWO_WHEEL = {"motorcycle": "motorcyclist", "bicycle": "cyclist"}
RIDER_HINT = {"@rider_moto": "motorcycle", "@rider_bike": "bicycle"}
WORK = 640  # working resolution (long side) for overlap tests


def _overlap(r_polys, v_polys, dilate_px: float, s: float) -> tuple[float, list[float], list[float]]:
    rb, vb = polys_bbox(r_polys), polys_bbox(v_polys)
    x0, y0 = min(rb[0], vb[0]) - dilate_px, min(rb[1], vb[1]) - dilate_px
    x1, y1 = max(rb[2], vb[2]) + dilate_px, max(rb[3], vb[3]) + dilate_px
    W, H = int(np.ceil((x1 - x0) * s)) + 2, int(np.ceil((y1 - y0) * s)) + 2
    if W * H > 4_000_000:
        return 0.0, rb, vb
    mr = rasterize(r_polys, x0, y0, W, H, s)
    mv = rasterize(v_polys, x0, y0, W, H, s)
    k = max(1, int(round(dilate_px * s)))
    import cv2
    mr = cv2.dilate(mr, np.ones((2 * k + 1, 2 * k + 1), np.uint8))
    inter = float((mr & mv).sum())
    return inter / max(1.0, float(mv.sum())), rb, vb


def _union_polys(polys_list, s_full: float) -> list:
    allp = [p for ps in polys_list for p in ps]
    x0, y0, x1, y1 = polys_bbox(allp)
    W, H = int(np.ceil((x1 - x0) * s_full)) + 3, int(np.ceil((y1 - y0) * s_full)) + 3
    m = rasterize(allp, x0 - 1, y0 - 1, W, H, s_full)
    return mask_to_polys(m, scale=s_full, offset=(x0 - 1, y0 - 1), min_area=2.0)


def merge_riders(rec: dict, cfg: dict, log: Counter) -> None:
    rm = cfg["rider_merge"]
    insts = rec["instances"]
    W, H = rec["width"], rec["height"]
    s = min(1.0, WORK / max(W, H))
    s_full = min(1.0, 1600 / max(W, H))
    vehicles = [i for i, it in enumerate(insts) if it["cls"] in TWO_WHEEL and it.get("polys")]
    people = [i for i, it in enumerate(insts) if it["cls"] in ("@rider", "@rider_moto", "@rider_bike", "@person")
              and it.get("polys")]
    if not people:
        return
    assign: dict[int, list[int]] = {}
    unmatched = []
    for pi in people:
        p = insts[pi]
        pb = polys_bbox(p["polys"])
        dil = rm["dilate_frac"] * max(pb[2] - pb[0], pb[3] - pb[1])
        best, best_v = 0.0, None
        for vi in vehicles:
            v = insts[vi]
            hint = RIDER_HINT.get(p["cls"])
            if hint and v["cls"] != hint:
                continue
            # Masks from occlusion-ordered sources (Cityscapes/Mapillary/IDD/panoptic) are DISJOINT:
            # the rider hides the top of the bike. So we measure contact: the rider mask is dilated
            # (dilate_frac x its size) and we count how much of the visible vehicle it touches.
            ov, rb, vb = _overlap(p["polys"], v["polys"], dil, s)
            cx = (rb[0] + rb[2]) / 2
            vw = vb[2] - vb[0]
            geometric = (vb[0] - 0.15 * vw <= cx <= vb[2] + 0.15 * vw  # rider centred over the bike
                         and rb[1] < vb[1] + 0.25 * (vb[3] - vb[1]))   # rider's head above the bike's top
            if p["cls"] == "@person":
                # Ambiguous person (COCO): BTC rule "người dắt xe là Pedestrian". A rider sits ON the
                # bike (feet clearly above the bike's bottom, centred inside it); a person walking a bike
                # stands on the ground next to it (feet at the bike's bottom, centre beside it).
                vh = vb[3] - vb[1]
                geometric = (vb[0] <= cx <= vb[2]
                             and rb[1] < vb[1]
                             and rb[3] <= vb[3] - rm.get("rider_feet_above_frac", 0.08) * vh)
            thr = rm["person_overlap_thr"] if p["cls"] == "@person" else rm["rider_overlap_thr"]
            if not geometric or ov < thr:
                continue
            if ov > best:
                best, best_v = ov, vi
        if best_v is None:
            unmatched.append(pi)
        else:
            assign.setdefault(best_v, []).append(pi)

    drop = set()
    new = []
    for vi, pis in assign.items():
        v = insts[vi]
        tgt = TWO_WHEEL[v["cls"]]
        if rm["mode"] == "union":
            polys = _union_polys([v["polys"]] + [insts[i]["polys"] for i in pis], s_full)
        elif rm["mode"] == "vehicle":
            polys = v["polys"]
        else:
            polys = _union_polys([insts[i]["polys"] for i in pis], s_full)
        raw = "+".join([v["raw"]] + [insts[i]["raw"] for i in pis])
        new.append({"raw": raw, "cls": tgt, "polys": polys, "origin": "merged", "conf": 1.0,
                    "n_riders": len(pis)})
        drop.update([vi, *pis])
        log[f"merged_{tgt}"] += 1
        log["riders_attached"] += len(pis)
    for pi in unmatched:
        p = insts[pi]
        if p["cls"] == "@person":
            p["cls"] = "pedestrian"
            log["person_to_pedestrian"] += 1
        elif p["cls"] in RIDER_HINT:
            p["cls"] = TWO_WHEEL[RIDER_HINT[p["cls"]]]
            p["origin"] = "rider_only"
            log["unmatched_rider_kept"] += 1
        else:
            p["cls"] = "@ignore"
            log["unmatched_rider_dropped"] += 1
    rec["instances"] = [it for i, it in enumerate(insts) if i not in drop] + new


FRONT_LAST = {"sign", "traffic_light"}  # thin roadside objects: never hidden by a vehicle's polygon


def deoverlap(rec: dict, log: Counter) -> None:
    """BTC rule: masks cover only the visible part and never overlap; the object in front keeps the
    pixels. Sources without depth order (COCO, BDD, merged rider unions) have overlapping polygons,
    so we repaint them far -> near: an object whose bottom edge is lower in the image is closer to
    the camera. Signs/lights are painted last. Non-overlapping instances keep their exact polygons."""
    insts = [it for it in rec["instances"] if it.get("polys")]
    n = len(insts)
    if n < 2:
        return
    boxes = np.array([polys_bbox(it["polys"]) for it in insts])
    inter = ((np.minimum(boxes[:, None, 2], boxes[None, :, 2]) > np.maximum(boxes[:, None, 0], boxes[None, :, 0]))
             & (np.minimum(boxes[:, None, 3], boxes[None, :, 3]) > np.maximum(boxes[:, None, 1], boxes[None, :, 1])))
    np.fill_diagonal(inter, False)
    touched = np.flatnonzero(inter.any(1))
    if len(touched) == 0:
        return
    W, H = rec["width"], rec["height"]
    s = min(1.0, 1600 / max(W, H))
    Wi, Hi = int(round(W * s)) + 1, int(round(H * s)) + 1
    idmap = np.zeros((Hi, Wi), np.int32)
    order = sorted(range(n), key=lambda i: (insts[i]["cls"] in FRONT_LAST, boxes[i][3]))
    import cv2
    for k in order:
        pts = [np.round(np.asarray(p, np.float64) * s).astype(np.int32) for p in insts[k]["polys"] if len(p) >= 3]
        cv2.fillPoly(idmap, pts, k + 1)
    gone = set()
    for k in touched:
        x0, y0, x1, y1 = boxes[k]
        cx0, cy0 = max(0, int(np.floor(x0 * s))), max(0, int(np.floor(y0 * s)))
        cx1, cy1 = min(Wi, int(np.ceil(x1 * s)) + 1), min(Hi, int(np.ceil(y1 * s)) + 1)
        crop = idmap[cy0:cy1, cx0:cx1]
        vis = crop == k + 1
        full = rasterize(insts[k]["polys"], cx0 / s, cy0 / s, cx1 - cx0, cy1 - cy0, s)
        if vis.sum() >= full.sum():
            continue  # nothing hidden
        polys = mask_to_polys(vis, scale=s, offset=(cx0 / s, cy0 / s), min_area=2.0)
        if polys:
            insts[k]["polys"] = polys
            insts[k]["occluded"] = True
            log["deoverlap_clipped"] += 1
        else:
            gone.add(id(insts[k]))
            log["deoverlap_fully_hidden"] += 1
    if gone:
        rec["instances"] = [it for it in rec["instances"] if id(it) not in gone]


def resolve_record(rec: dict, classes, cls_cfg: dict, log: Counter) -> dict | None:
    """Merge riders, make masks non-overlapping, drop ignored/too-small instances."""
    merge_riders(rec, cls_cfg, log)
    for it in rec["instances"]:
        if it["cls"] == "@person":
            it["cls"] = "pedestrian"
    rec["instances"] = [it for it in rec["instances"] if it["cls"] in classes.index
                        or log.update([f"drop_token_{it['cls']}"])]
    deoverlap(rec, log)
    W, H = rec["width"], rec["height"]
    k640 = (640.0 / max(W, H)) ** 2
    min_area = cls_cfg.get("min_area_640", {})
    # BTC: "đối tượng quá nhỏ (cạnh ngắn dưới khoảng 1,25% cạnh dài của ảnh)" are NOT labeled.
    min_short = cls_cfg.get("min_short_side_frac", 0.0) * max(W, H)
    keep = []
    for it in rec["instances"]:
        b = polys_bbox(it["polys"]) if it.get("polys") else it.get("box")
        if b and min(b[2] - b[0], b[3] - b[1]) < min_short:
            log[f"drop_short_side_{it['cls']}"] += 1
            continue
        if it.get("polys"):
            a = polys_area(it["polys"]) * k640
            if a < min_area.get(it["cls"], min_area.get("default", 0)):
                log[f"drop_small_{it['cls']}"] += 1
                continue
            it["area640"] = round(a, 1)
        elif it.get("box"):
            it["area640"] = round((b[2] - b[0]) * (b[3] - b[1]) * k640, 1)
        keep.append(it)
    rec["instances"] = keep
    labeled = set(rec.get("labeled_classes") or [])
    rec["missing_classes"] = [c for c in classes.names if c not in labeled]
    rec["needs_mask"] = any(not it.get("polys") for it in keep)
    rec["counts"] = {c: 0 for c in classes.names}
    for it in keep:
        rec["counts"][it["cls"]] += 1
    log["images_resolved"] += 1
    return rec


def to_single_polygon(polys: list) -> np.ndarray:
    """YOLO-seg needs one polygon per instance: connect multi-part masks (Ultralytics' method)."""
    if len(polys) == 1:
        return np.asarray(polys[0], np.float64)
    segs = merge_multi_segment([np.asarray(p, np.float64).reshape(-1).tolist() for p in polys])
    return np.concatenate(segs, 0)
