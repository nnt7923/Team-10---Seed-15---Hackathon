"""Generate a tiny SYNTHETIC multi-source dataset (coloured shapes) that exercises every converter
and the rider-merge logic, so the pipeline can be tested end-to-end without downloading anything.
The statistics produced from it are NOT real dataset statistics.

  python scripts/make_demo_data.py && python scripts/run_pipeline.py --config configs/pipeline_demo.yaml --synthetic
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "demo" / "raw"
W, H = 640, 384
COL = {"car": (60, 60, 200), "bus": (40, 160, 230), "truck": (90, 90, 90), "other": (180, 60, 160),
       "motorcycle": (30, 30, 30), "bicycle": (40, 140, 40), "person": (200, 150, 120),
       "sign": (20, 20, 220), "light": (30, 200, 30)}


def rect(x, y, w, h):
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def make_scene(rng: random.Random, weights: dict, night: bool):
    """Return (image, objects) where objects = [(kind, polygon, rider_of_index_or_None)] back->front."""
    base = 35 if night else 150
    # low-frequency random background so each synthetic frame has a distinct perceptual hash
    g = np.random.default_rng(rng.randint(0, 10**6))
    bg = cv2.resize(g.normal(0, 1, (6, 10, 3)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    img = np.clip(base + bg * (25 if night else 45), 0, 255).astype(np.uint8)
    cv2.rectangle(img, (0, H // 2 + rng.randint(-30, 30)), (W, H), (base // 2 + 10,) * 3, -1)
    objs = []
    kinds = list(weights)
    n = rng.randint(3, 12)
    for _ in range(n):
        k = rng.choices(kinds, [weights[x] for x in kinds])[0]
        x, y = rng.randint(10, W - 120), rng.randint(H // 2 - 40, H - 80)
        if k in ("car", "truck", "bus", "other"):
            w, h = {"car": (90, 45), "truck": (110, 60), "bus": (130, 65), "other": (70, 45)}[k]
            objs.append((k, rect(x, y, w, h), None))
        elif k in ("motorcycle", "bicycle", "motorcyclist", "cyclist"):
            veh = "motorcycle" if k.startswith("motor") else "bicycle"
            objs.append((veh, rect(x, y + 30, 40, 22), None))
            if k in ("motorcyclist", "cyclist"):
                vi = len(objs) - 1
                for j in range(rng.choice([1, 1, 2])):  # VN: often 2 people per bike
                    objs.append(("rider", rect(x + 6 + 12 * j, y - 5, 16, 42), vi))
        elif k == "pedestrian":
            objs.append(("person", rect(x, y - 10, 16, 50), None))
        elif k == "sign":
            objs.append(("sign", rect(x, rng.randint(20, H // 2 - 30), 14, 14), None))
        elif k == "light":
            objs.append(("light", rect(x, rng.randint(10, H // 2 - 40), 9, 22), None))
    for kind, poly, _ in objs:
        cv2.fillPoly(img, [np.asarray(poly, np.int32)], COL.get(kind, (200, 150, 120)))
    noise = np.random.default_rng(rng.randint(0, 10**6)).integers(0, 18, img.shape, dtype=np.uint8)
    return cv2.add(img, noise), objs


def norm(poly):
    return " ".join(f"{x / W:.6f} {y / H:.6f}" for x, y in poly)


def demo_vn(rng, n=160):
    """YOLO-seg with competition ids; riders already merged (union)."""
    out = ROOT / "demo_vn"
    ids = {"car": 0, "bus": 1, "truck": 2, "other": 3, "motorcycle": 4, "bicycle": 6, "person": 8, "sign": 9, "light": 10}
    w = {"motorcyclist": 10, "motorcycle": 4, "car": 3, "pedestrian": 2, "truck": 1, "bus": 0.6, "other": 0.6,
         "sign": 1.2, "light": 0.3, "cyclist": 0.4, "bicycle": 0.2}
    for i in range(n):
        g = f"video{i // 8:02d}"  # 8 frames per "video" -> group-aware split
        (out / "images" / g).mkdir(parents=True, exist_ok=True)
        (out / "labels" / g).mkdir(parents=True, exist_ok=True)
        img, objs = make_scene(rng, w, night=rng.random() < 0.15)
        cv2.imwrite(str(out / "images" / g / f"f{i:04d}.jpg"), img)
        lines, riders = [], {}
        for kind, poly, of in objs:
            if kind == "rider":
                riders.setdefault(of, []).append(poly)
        for j, (kind, poly, of) in enumerate(objs):
            if kind == "rider":
                continue
            if j in riders:  # union bbox polygon of vehicle + riders
                pts = np.asarray([poly] + riders[j]).reshape(-1, 2)
                x0, y0, x1, y1 = pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()
                cid = 5 if kind == "motorcycle" else 7
                lines.append(f"{cid} {norm(rect(x0, y0, x1 - x0, y1 - y0))}")
            else:
                lines.append(f"{ids[kind]} {norm(poly)}")
        (out / "labels" / g / f"f{i:04d}.txt").write_text("\n".join(lines) + "\n")
    # a near-duplicate pair to exercise dedupe
    src = out / "images" / "video00" / "f0000.jpg"
    (out / "images" / "dup").mkdir(exist_ok=True)
    (out / "labels" / "dup").mkdir(exist_ok=True)
    cv2.imwrite(str(out / "images" / "dup" / "copy.jpg"), cv2.imread(str(src)), [cv2.IMWRITE_JPEG_QUALITY, 80])
    (out / "labels" / "dup" / "copy.txt").write_text((out / "labels" / "video00" / "f0000.txt").read_text())


def demo_mapillary(rng, n=220):
    """Mapillary/Cityscapes-style polygons json, riders separate, back->front order."""
    out = ROOT / "demo_mapillary"
    (out / "polygons").mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(parents=True, exist_ok=True)
    lab = {"car": "object--vehicle--car", "bus": "object--vehicle--bus", "truck": "object--vehicle--truck",
           "other": "object--vehicle--other-vehicle", "motorcycle": "object--vehicle--motorcycle",
           "bicycle": "object--vehicle--bicycle", "person": "human--person--individual",
           "sign": "object--traffic-sign--front", "light": "object--traffic-light--general-upright"}
    w = {"car": 8, "pedestrian": 5, "sign": 5, "light": 3, "truck": 2, "bus": 1.5, "bicycle": 1, "cyclist": 1.2,
         "motorcycle": 0.8, "motorcyclist": 0.8, "other": 0.5}
    for i in range(n):
        img, objs = make_scene(rng, w, night=rng.random() < 0.1)
        cv2.imwrite(str(out / "images" / f"m{i:04d}.jpg"), img)
        o = []
        for kind, poly, of in objs:
            if kind == "rider":
                label = "human--rider--motorcyclist" if objs[of][0] == "motorcycle" else "human--rider--bicyclist"
            else:
                label = lab[kind]
            o.append({"label": label, "polygon": poly})
        o.append({"label": "object--traffic-sign--back", "polygon": rect(5, 5, 10, 10)})
        json.dump({"width": W, "height": H, "objects": o}, open(out / "polygons" / f"m{i:04d}.json", "w"))


def demo_coco(rng, n=120):
    """COCO instances json: person is ambiguous (no rider label) -> geometric rider test."""
    out = ROOT / "demo_coco"
    (out / "images").mkdir(parents=True, exist_ok=True)
    cats = [{"id": 1, "name": "person"}, {"id": 2, "name": "bicycle"}, {"id": 3, "name": "car"},
            {"id": 4, "name": "motorcycle"}, {"id": 6, "name": "bus"}, {"id": 8, "name": "truck"},
            {"id": 10, "name": "traffic light"}]
    cid = {"person": 1, "rider": 1, "bicycle": 2, "car": 3, "motorcycle": 4, "bus": 6, "truck": 8, "light": 10}
    w = {"car": 5, "pedestrian": 4, "bus": 2, "truck": 2, "light": 2, "motorcyclist": 1.5, "cyclist": 1.5,
         "motorcycle": 1, "bicycle": 1}
    images, anns, aid = [], [], 1
    for i in range(n):
        img, objs = make_scene(rng, w, night=rng.random() < 0.2)
        fn = f"c{i:05d}.jpg"
        cv2.imwrite(str(out / "images" / fn), img)
        images.append({"id": i, "file_name": fn, "width": W, "height": H})
        for kind, poly, of in objs:
            if kind == "sign":
                continue  # COCO has no generic sign class -> partial label
            p = np.asarray(poly, float)
            x0, y0 = p.min(0)
            x1, y1 = p.max(0)
            anns.append({"id": aid, "image_id": i, "category_id": cid[kind], "iscrowd": 0,
                         "segmentation": [p.reshape(-1).tolist()], "bbox": [x0, y0, x1 - x0, y1 - y0],
                         "area": float((x1 - x0) * (y1 - y0))})
            aid += 1
    json.dump({"images": images, "annotations": anns, "categories": cats}, open(out / "instances.json", "w"))


def demo_bdd(rng, n=100):
    """BDD100K Scalabel json with frame attributes (weather/timeofday); no signs/lights labeled."""
    out = ROOT / "demo_bdd"
    (out / "images").mkdir(parents=True, exist_ok=True)
    w = {"car": 6, "pedestrian": 3, "truck": 2, "bus": 1, "motorcyclist": 1, "cyclist": 1, "bicycle": 0.5}
    frames = []
    for i in range(n):
        night = rng.random() < 0.4
        img, objs = make_scene(rng, w, night=night)
        name = f"b{i:04d}.jpg"
        cv2.imwrite(str(out / "images" / name), img)
        cat = {"car": "car", "truck": "truck", "bus": "bus", "motorcycle": "motorcycle", "bicycle": "bicycle",
               "person": "pedestrian", "rider": "rider", "other": "other vehicle"}
        labels = [{"category": cat[k], "poly2d": [{"vertices": p, "types": "LLLL", "closed": True}]}
                  for k, p, _ in objs if k in cat]
        frames.append({"name": name, "videoName": f"seq{i // 10}",
                       "attributes": {"weather": rng.choice(["clear", "rainy", "rainy", "foggy", "overcast"]),
                                      "timeofday": "night" if night else "daytime"},
                       "labels": labels})
    json.dump(frames, open(out / "ins_seg.json", "w"))


def demo_panoptic(rng, n=60):
    """COCO-panoptic (ACDC-like): signs/lights are ONE stuff segment -> split_components."""
    out = ROOT / "demo_acdc"
    for sub in ("rgb/rain", "rgb/fog", "rgb/night", "panoptic"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    cats = [{"id": 24, "name": "person"}, {"id": 25, "name": "rider"}, {"id": 26, "name": "car"},
            {"id": 27, "name": "truck"}, {"id": 28, "name": "bus"}, {"id": 32, "name": "motorcycle"},
            {"id": 33, "name": "bicycle"}, {"id": 19, "name": "traffic light"}, {"id": 20, "name": "traffic sign"}]
    cmap = {"person": 24, "rider": 25, "car": 26, "truck": 27, "bus": 28, "motorcycle": 32, "bicycle": 33,
            "light": 19, "sign": 20}
    w = {"car": 5, "pedestrian": 3, "sign": 3, "light": 2, "truck": 1, "bus": 1, "motorcyclist": 1, "cyclist": 1}
    images, anns = [], []
    for i in range(n):
        cond = ["rain", "fog", "night"][i % 3]
        img, objs = make_scene(rng, w, night=cond == "night")
        if cond == "fog":
            img = cv2.addWeighted(img, 0.5, np.full_like(img, 200), 0.5, 0)
        fn = f"{cond}/a{i:04d}.png"
        cv2.imwrite(str(out / "rgb" / fn), img)
        ids = np.zeros((H, W), np.int32)
        segs, stuff = [], {}
        for k, (kind, poly, _) in enumerate(objs):
            if kind in ("sign", "light"):
                sid = stuff.setdefault(kind, 1000 + len(stuff))
            else:
                sid = k + 1
                segs.append({"id": sid, "category_id": cmap[kind], "iscrowd": 0})
            cv2.fillPoly(ids, [np.asarray(poly, np.int32)], sid)
        segs += [{"id": sid, "category_id": cmap[kind], "iscrowd": 0} for kind, sid in stuff.items()]
        rgb = np.stack([ids % 256, (ids // 256) % 256, ids // 65536], -1).astype(np.uint8)
        cv2.imwrite(str(out / "panoptic" / f"a{i:04d}_panoptic.png"), rgb[:, :, ::-1])
        images.append({"id": i, "file_name": fn, "width": W, "height": H})
        anns.append({"image_id": i, "file_name": f"a{i:04d}_panoptic.png", "segments_info": segs})
    json.dump({"images": images, "annotations": anns, "categories": cats}, open(out / "panoptic.json", "w"))


if __name__ == "__main__":
    rng = random.Random(0)
    demo_vn(rng)
    demo_mapillary(rng)
    demo_coco(rng)
    demo_bdd(rng)
    demo_panoptic(rng)
    print(f"Synthetic demo data written to {ROOT}")
