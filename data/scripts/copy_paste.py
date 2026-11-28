"""v3 — offline CONTEXT-AWARE copy-paste (Ghiasi et al. CVPR 2021 "Simple Copy-Paste"; Dwibedi et al. ICCV 2017
"Cut, Paste and Learn"; traffic-context placement as in Li et al. 2022). Only HUMAN labels are used:

* backgrounds : IDD + Cityscapes TRAIN images (human masks for all 11 classes + road/sidewalk polygons)
* object bank : nuImages (human masks: trailer / construction / bicycle / motorcycle), COCO street (human masks:
                bicycle / motorcycle), VN traffic signs (human boxes -> SAM masks; simple convex shapes)

Per pasted object:
  1. placement  : foot point sampled on the right ground type of the background (road / sidewalk polygons)
  2. scale      : flat-ground pinhole model  px_per_metre(y) = a*y + b, fitted per image on the human-labelled
                  cars / people / trucks in that image (fallback: per-source fit) -> object height = real height
                  * px_per_metre at its foot row; signs are mounted 2.2-2.8 m above the ground on a drawn pole
  3. depth      : objects whose foot is lower in the image (closer) occlude the pasted object; the pasted object
                  occludes farther objects (their masks are cut, visible-only like the BTC guideline); placements
                  that hide >60 % of an existing object, or touch an existing sign / traffic light, are rejected
  4. blending   : LAB colour shift towards the surrounding background, feathered alpha, soft contact shadow
  5. labels     : pasted mask (visible part) + updated masks; BTC size rule (short side >= 1.25 % long side,
                  area >= 64 px^2 at 640) re-applied

  python scripts/copy_paste.py --config configs/pipeline_v3.yaml --n 4000 --workers 3
  python scripts/copy_paste.py --config configs/pipeline_v3.yaml --n 24 --qa reports/qa_cp.jpg   # preview only
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vtb.common import load_pipeline, mask_to_polys, polys_bbox, rasterize, read_jsonl  # noqa: E402
from vtb.quality import _hash01  # noqa: E402
from vtb.resolve import resolve_record  # noqa: E402

BG_SOURCES = {"idd_segmentation": 0.65, "cityscapes": 0.35}
BANK_SOURCES = ["nuimages", "coco2017_street_yoloseg", "vn_traffic_sign_hf"]
MAX_SIDE = 1280
# real heights (m) of reference classes for the ground-plane fit
REF_H = {"car": 1.5, "pedestrian": 1.7, "motorcyclist": 1.6, "cyclist": 1.7, "truck": 3.0, "bus": 3.2}
# what we paste: class -> (weight, ground, real height m)
PASTE = {"sign": (0.35, "side", 0.75), "other_vehicle": (0.25, "road", None),
         "bicycle": (0.20, "side", 1.0), "motorcycle": (0.20, "side", 1.15)}
OV_H = {"vehicle.trailer": 3.6, "vehicle.construction": 3.2}
FILL_MIN = {"sign": 0.45, "other_vehicle": 0.45, "motorcycle": 0.30, "bicycle": 0.18}
GROUND = {"road": 1, "drivable fallback": 1, "parking": 2, "sidewalk": 2, "non-drivable fallback": 2, "curb": 2}
GS = 0.25  # ground-map scale


# ----------------------------------------------------------------------------------------------- helpers
def area(polys) -> float:
    return float(sum(abs(cv2.contourArea(np.asarray(p, np.float32).reshape(-1, 2))) for p in polys if len(p) >= 3))


def scale_rec(r: dict, s: float) -> dict:
    if s == 1.0:
        return r
    r["width"], r["height"] = round(r["width"] * s), round(r["height"] * s)
    for it in r["instances"]:
        it["polys"] = [(np.asarray(p, np.float64) * s).round(2).tolist() for p in it["polys"]]
    return r


def gt_json(img_path: str) -> Path:
    p = Path(img_path)
    parts = list(p.parts)
    parts[len(parts) - 1 - parts[::-1].index("leftImg8bit")] = "gtFine"
    return Path(*parts[:-1]) / (p.name.split("_leftImg8bit")[0] + "_gtFine_polygons.json")


def ground_map(img_path: str, W: int, H: int) -> np.ndarray | None:
    f = gt_json(img_path)
    if not f.exists():
        return None
    d = json.load(open(f, encoding="utf-8"))
    sx, sy = W / d["imgWidth"] * GS, H / d["imgHeight"] * GS
    g = np.zeros((max(1, round(H * GS)), max(1, round(W * GS))), np.uint8)
    for o in d["objects"]:  # listed back to front: later objects cover earlier ones
        pts = np.asarray(o.get("polygon") or [], np.float64)
        if len(pts) < 3:
            continue
        pts = np.round(pts * [sx, sy]).astype(np.int32)
        cv2.fillPoly(g, [pts], GROUND.get(o["label"], 0))
    return g


def fit_plane(r: dict):
    """px_per_metre = a*y_bottom + b from labelled references; None if unreliable."""
    W, H = r["width"], r["height"]
    ys, vs = [], []
    for it in r["instances"]:
        if it["cls"] not in REF_H or not it.get("polys"):
            continue
        x0, y0, x1, y1 = polys_bbox(it["polys"])
        if y1 > H - 3 or y0 < 2 or (y1 - y0) < 12 or x0 < 2 or x1 > W - 3:
            continue
        ys.append(y1)
        vs.append((y1 - y0) / REF_H[it["cls"]])
    if len(ys) < 4:
        return None
    ys, vs = np.asarray(ys), np.asarray(vs)
    for _ in range(2):
        a, b = np.polyfit(ys, vs, 1)
        res = np.abs(vs - (a * ys + b))
        keep = res <= max(2.5 * np.median(res), 1e-6)
        if keep.sum() < 4:
            break
        ys, vs = ys[keep], vs[keep]
    a, b = np.polyfit(ys, vs, 1)
    if a <= 0:
        return None
    yh = -b / a
    if not (0.15 * H <= yh <= 0.75 * H):
        return None
    return float(a), float(b)


# ----------------------------------------------------------------------------------------------- bank
def build_bank(conv: Path, classes, cls_cfg) -> dict[str, list[dict]]:
    bank = defaultdict(list)
    for sid in BANK_SOURCES:
        f = conv / f"{sid}.jsonl"
        if not f.exists():
            print(f"[bank] {sid}: not converted -> skipped")
            continue
        n0 = 0
        for r in read_jsonl(f):
            if not Path(r["image"]).exists():
                continue
            r = resolve_record(r, classes, cls_cfg, Counter())
            W, H = r["width"], r["height"]
            boxes = [polys_bbox(it["polys"]) for it in r["instances"] if it.get("polys")]
            for it in r["instances"]:
                c = it["cls"]
                if c not in PASTE or not it.get("polys"):
                    continue
                org = it.get("origin", "human")
                if not (org == "human" or (sid == "vn_traffic_sign_hf" and org.startswith("sam_box"))):
                    continue
                if c == "other_vehicle" and it.get("raw") not in OV_H:
                    continue
                x0, y0, x1, y1 = b = polys_bbox(it["polys"])
                bw, bh = x1 - x0, y1 - y0
                if x0 < 3 or y0 < 3 or x1 > W - 4 or y1 > H - 4:
                    continue  # truncated
                if bh < (20 if c == "sign" else 40) or bw < 10:
                    continue
                a = area(it["polys"])
                if a / (bw * bh) < FILL_MIN[c]:
                    continue
                big = max(area([p]) for p in it["polys"])
                if big < 0.9 * a:
                    continue  # fragmented (occluded) mask
                ob = 0.0
                for b2 in boxes:
                    if b2 is b or b2 == b:
                        continue
                    ix = max(0, min(x1, b2[2]) - max(x0, b2[0]))
                    iy = max(0, min(y1, b2[3]) - max(y0, b2[1]))
                    ob = max(ob, ix * iy / (bw * bh))
                if ob > 0.10:
                    continue  # touching / occluded by another object (e.g. rider next to a bike)
                bank[c].append({"src": sid, "image": r["image"], "polys": it["polys"], "box": b,
                                "raw": it.get("raw"), "uid": r["uid"]})
                n0 += 1
        print(f"[bank] {sid}: {n0} objects", flush=True)
    return dict(bank)


def load_crop(o: dict):
    img = cv2.imread(o["image"])
    if img is None:
        return None
    x0, y0, x1, y1 = [int(round(v)) for v in o["box"]]
    x0, y0 = max(0, x0 - 1), max(0, y0 - 1)
    x1, y1 = min(img.shape[1], x1 + 2), min(img.shape[0], y1 + 2)
    m = rasterize(o["polys"], x0, y0, x1 - x0, y1 - y0, 1.0)
    return img[y0:y1, x0:x1].copy(), m


# ----------------------------------------------------------------------------------------------- paste
def lab_match(crop, mask, bg_patch, k=0.45):
    c = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = cv2.cvtColor(bg_patch, cv2.COLOR_BGR2LAB).astype(np.float32)
    sel = mask > 0
    if sel.sum() < 10:
        return crop
    for ch in range(3):
        mu_c, mu_b = c[..., ch][sel].mean(), b[..., ch].mean()
        c[..., ch] += k * (mu_b - mu_c) * (1.0 if ch == 0 else 0.5)
    return cv2.cvtColor(np.clip(c, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def pick(bank_c, need_h, rng):
    """random object whose crop is at least need_h/1.5 px tall (avoids blurry up-scaling)."""
    import bisect
    hs = [o["h"] for o in bank_c]
    i = bisect.bisect_left(hs, need_h / 1.5)
    if i >= len(bank_c):
        i = max(0, len(bank_c) - 50)
    return bank_c[rng.randrange(i, len(bank_c))]


def try_paste(img, rec, gmap, plane, cls, bank_c, rng, log) -> bool:
    H, W = img.shape[:2]
    o = None
    a, b = plane
    want_g = PASTE[cls][1]
    cand = np.nonzero(gmap == (1 if want_g == "road" else 2))
    if cls in ("motorcycle", "bicycle"):  # parked: on the sidewalk, or on the road strip next to it
        side = cv2.dilate((gmap == 2).astype(np.uint8), np.ones((3, 9), np.uint8))
        edge = np.nonzero((gmap == 1) & (side > 0))
    for _ in range(25):
        ys, xs = cand
        if cls in ("motorcycle", "bicycle") and (len(ys) == 0 or rng.random() < 0.3) and len(edge[0]):
            ys, xs = edge
        if len(ys) == 0:
            log["no_ground"] += 1
            return False
        k = rng.randrange(len(ys))
        px, yb = (xs[k] + rng.random()) / GS, (ys[k] + rng.random()) / GS
        ppm = a * yb + b
        if ppm <= 0:
            continue
        if o is None:
            est = ppm * (PASTE[cls][2] or 3.4)
            o = pick(bank_c, est, rng)
            got = load_crop(o)
            if got is None:
                return False
            crop, m = got
            ch_, cw_ = m.shape
        if cls == "sign":
            side = PASTE["sign"][2] * ppm * rng.uniform(0.85, 1.3)
            f = side / max(ch_, cw_)
            th, tw = ch_ * f, cw_ * f
            cy = yb - ppm * rng.uniform(2.2, 2.8)
            x0, y0 = px - tw / 2, cy - th / 2
        else:
            real = OV_H.get(o.get("raw"), 2.0) if cls == "other_vehicle" else PASTE[cls][2]
            th = real * ppm * rng.uniform(0.92, 1.08)
            f = th / ch_
            tw = cw_ * f
            x0, y0 = px - tw / 2, yb - th
        if f > 1.6:
            log["reject_upscale"] += 1
            continue
        if min(th, tw) < max(8, 0.0125 * max(W, H)):
            log["reject_small"] += 1
            continue
        if x0 < 1 or y0 < 1 or x0 + tw > W - 2 or y0 + th > H - 2:
            continue
        if cls != "sign":  # the whole footprint must stand on ground
            xx = np.linspace(x0 + 0.1 * tw, x0 + 0.9 * tw, 9)
            gy = min(gmap.shape[0] - 1, int(yb * GS))
            sup = np.mean([gmap[gy, min(gmap.shape[1] - 1, int(x * GS))] > 0 for x in xx])
            if sup < 0.7:
                continue
        X0, Y0 = int(np.floor(x0)), int(np.floor(y0))
        tw_i, th_i = max(2, int(round(tw))), max(2, int(round(th)))
        X1, Y1 = min(W, X0 + tw_i), min(H, Y0 + th_i)
        tw_i, th_i = X1 - X0, Y1 - Y0
        flip = cls != "sign" and rng.random() < 0.5
        cr = cv2.resize(crop, (tw_i, th_i), interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)
        mr = cv2.resize(m.astype(np.float32), (tw_i, th_i), interpolation=cv2.INTER_LINEAR) > 0.5
        if flip:
            cr, mr = cr[:, ::-1], mr[:, ::-1]
        P = mr.copy()
        pa0 = P.sum()
        if pa0 < 20:
            continue
        # ---- depth / occlusion against existing labels (window = paste box)
        depth = yb
        changes, ok, farther = [], True, []
        for idx, it in enumerate(rec["instances"]):  # pass 1: closer objects hide the pasted one
            if not it.get("polys"):
                continue
            bx = polys_bbox(it["polys"])
            if bx[2] < X0 or bx[0] > X1 or bx[3] < Y0 or bx[1] > Y1:
                continue
            Me = rasterize(it["polys"], X0, Y0, tw_i, th_i, 1.0) > 0
            if not (Me & P).any():
                continue
            if it["cls"] in ("sign", "traffic_light"):
                ok = False
                break
            if bx[3] > depth:
                P &= ~Me
            else:
                farther.append(idx)
        for idx in (farther if ok else []):  # pass 2: the pasted object hides farther ones (visible-only masks)
            full = rasterize(rec["instances"][idx]["polys"], 0, 0, W, H, 1.0) > 0
            fa = full.sum()
            full[Y0:Y1, X0:X1] &= ~P
            if fa == 0 or full.sum() / fa < 0.4:
                ok = False
                break
            changes.append((idx, full))
        if not ok or P.sum() < 0.5 * pa0:
            log["reject_occlusion"] += 1
            continue
        # ---- render
        win = img[Y0:Y1, X0:X1]
        ring = img[max(0, Y0 - th_i // 2):min(H, Y1 + th_i // 2), max(0, X0 - tw_i // 2):min(W, X1 + tw_i // 2)]
        cr = lab_match(cr, mr, ring)
        if cls == "sign":  # pole from the sign to the ground (not labelled, like the BTC guideline)
            pw = max(2, int(round(0.06 * ppm)))
            cxp = int(round(x0 + tw / 2))
            ytop, ybot = int(round(y0 + th * 0.9)), int(round(min(H - 1, yb)))
            if ybot > ytop:
                pole = np.zeros((H, W), bool)
                pole[ytop:ybot, max(0, cxp - pw // 2):min(W, cxp - pw // 2 + pw)] = True
                for it in rec["instances"]:  # closer objects stay in front of the pole
                    bx = polys_bbox(it["polys"])
                    if bx[3] > yb and not (bx[2] < cxp - pw or bx[0] > cxp + pw or bx[3] < ytop or bx[1] > ybot):
                        pole &= ~(rasterize(it["polys"], 0, 0, W, H, 1.0) > 0)
                g = rng.randint(85, 140)
                img[pole] = (0.15 * img[pole] + 0.85 * np.array([g, g, g + rng.randint(-8, 8)])).astype(np.uint8)
        else:  # soft contact shadow
            sh = np.zeros((H, W), np.float32)
            cv2.ellipse(sh, (int(px), int(yb)), (max(2, int(0.45 * tw)), max(2, int(0.05 * th) + 2)), 0, 0, 360, 1.0, -1)
            sh = cv2.GaussianBlur(sh, (0, 0), max(1.0, 0.03 * th))
            img[:] = (img * (1 - 0.35 * sh[..., None])).astype(np.uint8)
            win = img[Y0:Y1, X0:X1]
        alpha = cv2.GaussianBlur(P.astype(np.float32), (0, 0), max(0.5, 0.004 * th_i))
        alpha[mr & ~P] = 0.0  # pixels of closer objects stay untouched
        win[:] = (win * (1 - alpha[..., None]) + cr * alpha[..., None]).astype(np.uint8)
        # ---- labels
        for idx, full in changes:
            rec["instances"][idx]["polys"] = mask_to_polys(full)
        polys = mask_to_polys(P.astype(np.uint8), offset=(X0, Y0))
        if not polys:
            continue
        rec["instances"].append({"raw": f"paste:{o['src']}:{o.get('raw') or cls}", "cls": cls, "polys": polys,
                                 "origin": "paste", "conf": 1.0, "paste_from": o["uid"]})
        log[f"pasted_{cls}"] += 1
        return True
    return False


_G = {}


def _init(bank, src_plane, classes, cls_cfg):
    _G.update(bank=bank, src_plane=src_plane, classes=classes, cls_cfg=cls_cfg)


def make_one(args):
    bg, k, out_dir, seed = args
    bank, src_plane, classes, cls_cfg = _G["bank"], _G["src_plane"], _G["classes"], _G["cls_cfg"]
    rng = random.Random(f"{seed}:{bg['uid']}:{k}")
    log = Counter()
    img = cv2.imread(bg["image"])
    if img is None:
        log["bg_unreadable"] += 1
        return None, log
    s = min(1.0, MAX_SIDE / max(img.shape[:2]))
    if s < 1:
        img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    rec = json.loads(json.dumps(bg))
    rec = scale_rec(rec, s)
    rec["width"], rec["height"] = img.shape[1], img.shape[0]
    gmap = ground_map(bg["image"], rec["width"], rec["height"])
    if gmap is None or not (gmap > 0).any():
        log["no_ground_map"] += 1
        return None, log
    plane = fit_plane(rec)
    if plane is None:
        a0, b0 = src_plane[bg["source"]]  # normalised fit: ppm/H = a0*(y/H) + b0
        plane = (a0, b0 * rec["height"])
        log["plane_fallback"] += 1
    n_paste = rng.choice([1, 1, 2, 2, 3])
    names, wts = list(PASTE), [PASTE[c][0] for c in PASTE]
    done = 0
    for _ in range(n_paste * 3):
        if done >= n_paste:
            break
        cls = rng.choices(names, wts)[0]
        if not bank.get(cls):
            continue
        if try_paste(img, rec, gmap, plane, cls, bank[cls], rng, log):
            done += 1
    if done == 0:
        log["bg_no_paste"] += 1
        return None, log
    # BTC size rule again (pasting cut some masks)
    W, H = rec["width"], rec["height"]
    k640 = (640.0 / max(W, H)) ** 2
    mins = cls_cfg.get("min_short_side_frac", 0.0) * max(W, H)
    keep = []
    for it in rec["instances"]:
        if not it.get("polys"):
            continue
        bx = polys_bbox(it["polys"])
        a = area(it["polys"]) * k640
        if min(bx[2] - bx[0], bx[3] - bx[1]) < mins or a < cls_cfg["min_area_640"].get("default", 64):
            log[f"drop_small_after_paste_{it['cls']}"] += 1
            continue
        it["area640"] = round(a, 1)
        keep.append(it)
    rec["instances"] = keep
    rec["counts"] = {c: 0 for c in classes.names}
    for it in keep:
        rec["counts"][it["cls"]] += 1
    rec["uid"] = f"cp_synth/{bg['uid']}#{k}"
    rec["source"] = "cp_synth"
    rec["group"] = f"cp_synth/{bg['group']}"
    rec["attrs"] = {"bg_uid": bg["uid"], "bg_source": bg["source"], "pasted": done}
    rec["synthetic"] = rec["completed"] = True
    rec["missing_classes"], rec["needs_mask"] = [], False
    name = rec["uid"].replace("/", "__").replace("#", "_") + ".jpg"
    path = Path(out_dir) / name
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 94])
    rec["image"] = str(path)
    log["images"] += 1
    return rec, log


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pipeline_v3.yaml")
    ap.add_argument("--n", type=int, default=4000, help="synthetic images to generate")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--qa", default="", help="write a preview grid (pasted objects outlined) and stop")
    a = ap.parse_args()
    cfg, classes, sources, _ = load_pipeline(a.config)
    cls_cfg = classes.cfg
    work, raw = Path(cfg["work_dir"]), Path(cfg["data_root"])
    conv = work / "converted"
    seed = cfg["seed"]
    vt = cfg["split"]["val_fraction"] + cfg["split"]["test_fraction"]
    # backgrounds: TRAIN groups only (same hash as vtb.quality.assign_splits) -> no val leakage
    bgs, src_fits = [], defaultdict(list)
    for sid in BG_SOURCES:
        for r in read_jsonl(conv / f"{sid}.jsonl"):
            if _hash01(r["group"], seed) < vt or not Path(r["image"]).exists():
                continue
            r = resolve_record(r, classes, cls_cfg, Counter())
            fp = fit_plane(r)
            if fp:
                src_fits[sid].append((fp[0], fp[1] / r["height"]))
            bgs.append(r)
    src_plane = {s: tuple(np.median(np.asarray(v), 0)) for s, v in src_fits.items()}
    print(f"[cp] backgrounds (train only): {Counter(b['source'] for b in bgs)}  plane fits: "
          f"{ {s: len(v) for s, v in src_fits.items()} }", flush=True)
    bank = build_bank(conv, classes, cls_cfg)
    for c in bank:
        for o in bank[c]:
            o["h"] = o["box"][3] - o["box"][1] if c != "sign" else max(o["box"][3] - o["box"][1], o["box"][2] - o["box"][0])
        bank[c].sort(key=lambda o: o["h"])
    print(f"[cp] bank: { {c: len(v) for c, v in bank.items()} }", flush=True)
    rng = random.Random(seed)
    by_src = defaultdict(list)
    for b in bgs:
        by_src[b["source"]].append(b)
    n = 24 if a.qa else a.n
    jobs = []
    for k in range(n):
        sid = rng.choices(list(BG_SOURCES), list(BG_SOURCES.values()))[0]
        jobs.append(rng.choice(by_src[sid]))
    out_dir = raw / "cp_synth"
    out_dir.mkdir(parents=True, exist_ok=True)
    recs, log = [], Counter()
    args = [(bg, k, str(out_dir), seed) for k, bg in enumerate(jobs)]
    _init(bank, src_plane, classes, cls_cfg)
    if a.workers > 1:
        with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(bank, src_plane, classes, cls_cfg)) as ex:
            res = ex.map(make_one, args, chunksize=8)
            for i, (r, lg) in enumerate(res):
                log.update(lg)
                if r:
                    recs.append(r)
                if i % 500 == 0:
                    print(f"[cp] {i}/{n}", flush=True)
    else:
        for x in args:
            r, lg = make_one(x)
            log.update(lg)
            if r:
                recs.append(r)
    print(f"[cp] {dict(log)}", flush=True)
    if a.qa:
        tiles = []
        for r in recs[:12]:
            img = cv2.imread(r["image"])
            for it in r["instances"]:
                pts = [np.asarray(p, np.int32) for p in it["polys"]]
                col = (0, 0, 255) if it.get("origin") == "paste" else (0, 200, 0)
                cv2.polylines(img, pts, True, col, 2 if it.get("origin") == "paste" else 1)
                if it.get("origin") == "paste":
                    x, y = pts[0].min(0)
                    cv2.putText(img, it["cls"], (int(x), max(12, int(y) - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 0, 255), 2, cv2.LINE_AA)
            s = 720 / img.shape[1]
            tiles.append(cv2.resize(img, (720, int(img.shape[0] * s))))
        hmax = max(t.shape[0] for t in tiles)
        tiles = [np.pad(t, ((0, hmax - t.shape[0]), (0, 0), (0, 0))) for t in tiles]
        while len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])
        Path(a.qa).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(a.qa, grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print("[cp] qa ->", a.qa)
        return
    with open(work / "cp_synth.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[cp] {len(recs)} synthetic records -> {work / 'cp_synth.jsonl'}")


if __name__ == "__main__":
    main()
