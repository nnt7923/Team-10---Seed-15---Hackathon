"""Write the final YOLO-seg dataset + full provenance metadata."""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from .resolve import to_single_polygon


def _safe(uid: str, max_len: int = 96) -> str:
    """Filesystem-safe, unique name. Long uids (Roboflow names reach 200+ chars) are cut and suffixed with
    a hash so the full path stays under Windows' 260-character limit."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "__", uid)
    if len(name) > max_len:
        import hashlib
        name = name[: max_len - 13] + "_" + hashlib.md5(uid.encode()).hexdigest()[:12]
    return name


def _write_one(args):
    r, out_dir, max_side, q, cls_index = args
    name = _safe(r["uid"])
    img_rel = Path("images") / r["split"] / f"{name}.jpg"
    lbl_rel = Path("labels") / r["split"] / f"{name}.txt"
    img_out, lbl_out = out_dir / img_rel, out_dir / lbl_rel
    if not img_out.exists():
        with Image.open(r["image"]) as im:
            im = im.convert("RGB")
            s = min(1.0, max_side / max(im.size))
            if s < 1.0:
                im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
            ew, eh = im.size
            im.save(img_out, quality=q)
    else:
        with Image.open(img_out) as im:
            ew, eh = im.size
    W, H = r["width"], r["height"]
    lines = []
    for it in r["instances"]:
        if not it.get("polys"):
            continue
        p = to_single_polygon(it["polys"]) / [W, H]
        p = np.clip(p, 0.0, 1.0)
        if len(p) < 3:
            continue
        lines.append(f"{cls_index[it['cls']]} " + " ".join(f"{v:.6f}" for v in p.reshape(-1)))
    lbl_out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return r["uid"], img_rel.as_posix(), lbl_rel.as_posix(), ew, eh, len(lines)


def export(records: list[dict], classes, registry: dict[str, dict], cfg: dict, out_dir: Path,
           unselected_uids: list[str], stages: dict, logs: dict) -> Counter:
    log = Counter()
    ex = cfg["export"]
    for sp in ("train", "val", "test"):
        (out_dir / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / sp).mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata").mkdir(parents=True, exist_ok=True)
    jobs = [(r, out_dir, ex["max_side"], ex["jpeg_quality"], classes.index) for r in records]
    with ThreadPoolExecutor(cfg.get("workers", 8)) as pool:
        results = {u: (i, l, w, h, n) for u, i, l, w, h, n in pool.map(_write_one, jobs)}

    lists = defaultdict(list)
    meta_lines = []
    raw_map = Counter()
    for r in records:
        img_rel, lbl_rel, ew, eh, n = results[r["uid"]]
        rep = r.get("repeat", 1) if r["split"] == "train" else 1
        lists[r["split"]].extend([f"./{img_rel}"] * rep)
        src = registry[r["source"]]
        origins = Counter(it.get("origin", "human") for it in r["instances"])
        for it in r["instances"]:
            raw_map[(r["source"], it["raw"], it["cls"])] += 1
        meta_lines.append({
            "file": img_rel, "label_file": lbl_rel, "uid": r["uid"], "split": r["split"],
            "repeat": rep, "rfs_factor": r.get("rfs_factor"),
            "source": r["source"], "source_name": src.get("name"), "tier": r.get("tier"),
            "license": src.get("license"), "homepage": src.get("homepage"), "citation": src.get("citation"),
            "orig_image": r["image"], "orig_id": r.get("orig_id"), "orig_split": r.get("orig_split"),
            "group": r.get("group"), "domain": r.get("domain"),
            "timeofday": r.get("condition"), "weather": r.get("weather"), "attrs": r.get("attrs"),
            "orig_size": [r["width"], r["height"]], "export_size": [ew, eh],
            "n_instances": n, "counts": {c: v for c, v in r["counts"].items() if v},
            "label_origin": dict(origins), "labeled_classes": r.get("labeled_classes"),
            "teacher_completed": bool(r.get("completed")), "needs_review": bool(r.get("needs_review")),
            "phash": r.get("phash"), "luma": r.get("luma"),
            "dup_cluster_size": r.get("dup_cluster_size", 1), "dup_of": r.get("dup_of", []),
        })
        log[f"exported_{r['split']}"] += 1
    for sp in ("train", "val", "test"):
        (out_dir / f"{sp}.txt").write_text("\n".join(lists[sp]) + ("\n" if lists[sp] else ""), encoding="utf-8")
    log["train_lines_with_repeats"] = len(lists["train"])

    data_yaml = {"path": str(out_dir.resolve()), "train": "train.txt", "val": "val.txt",
                 "names": {i: n for i, n in enumerate(classes.names)}}
    if lists["test"]:
        data_yaml["test"] = "test.txt"
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8")

    md = out_dir / "metadata"
    with open(md / "images.jsonl", "w", encoding="utf-8") as f:
        for m in meta_lines:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    (md / "unselected_train_pool_uids.txt").write_text("\n".join(unselected_uids), encoding="utf-8")
    src_meta = []
    for sid, s in registry.items():
        src_meta.append({k: s.get(k) for k in ("id", "name", "tier", "domain", "homepage", "download", "license",
                                                "citation", "format", "labeled_classes", "max_images", "priority")}
                        | {"counts": {st: stages[st]["per_source"].get(sid) for st in stages if "per_source" in stages[st]}})
    (md / "sources.json").write_text(json.dumps(src_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    cm = defaultdict(list)
    for (s, raw, cls), n in sorted(raw_map.items()):
        cm[s].append({"raw_label": raw, "target_class": cls, "class_id": classes.index[cls], "instances": n})
    (md / "class_mapping_observed.json").write_text(json.dumps(cm, ensure_ascii=False, indent=2), encoding="utf-8")
    build = {"built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "pipeline_config": cfg, "classes": classes.names, "logs": logs}
    (md / "build_info.json").write_text(json.dumps(build, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (md / "stats.json").write_text(json.dumps(stages, ensure_ascii=False, indent=2), encoding="utf-8")
    (md / "datasets_report.md").write_text(datasets_report(meta_lines, registry, cm, classes), encoding="utf-8")
    if ex.get("make_zip", True):
        log["zip_bytes"] = make_zip(out_dir, classes, bool(lists["test"]), ex.get("zip_name", "du_lieu.zip"))
    return log


def datasets_report(meta_lines: list[dict], registry: dict, mapping: dict, classes) -> str:
    """Table required by the competition report: name, link, license, #images used, mapping to 11 classes."""
    per = defaultdict(lambda: Counter())
    inst = defaultdict(lambda: Counter())
    orig = defaultdict(lambda: Counter())
    for m in meta_lines:
        s = m["source"]
        per[s][m["split"]] += 1
        per[s]["train_lines"] += m["repeat"] if m["split"] == "train" else 0
        inst[s].update(m["counts"])
        orig[s].update(m["label_origin"])
    L = ["# Danh sách bộ dữ liệu đã dùng (tự sinh từ pipeline)\n",
         "| Bộ dữ liệu | Đường dẫn | Giấy phép | Ảnh train (duy nhất) | Dòng train (có lặp) | Ảnh val | Ảnh test | Instance | Nguồn nhãn |",
         "|---|---|---|---:|---:|---:|---:|---:|---|"]
    for sid in sorted(per):
        s = registry[sid]
        o = ", ".join(f"{k}: {v:,}" for k, v in orig[sid].most_common())
        L.append(f"| {s.get('name')} (`{sid}`) | {s.get('homepage') or '–'} | {s.get('license')} | {per[sid]['train']:,} | "
                 f"{per[sid]['train_lines']:,} | {per[sid]['val']:,} | {per[sid]['test']:,} | "
                 f"{sum(inst[sid].values()):,} | {o} |")
    L.append("\n## Ánh xạ nhãn gốc → 11 lớp (đếm thực tế trong bộ dữ liệu cuối)\n")
    for sid in sorted(per):
        L.append(f"### {registry[sid].get('name')} (`{sid}`)\n")
        L.append("| Nhãn gốc | Lớp đích | Instance |\n|---|---|---:|")
        rows = Counter()
        for e in mapping.get(sid, []):
            rows[(e["raw_label"], f"{e['class_id']} {e['target_class']}")] += e["instances"]
        for (raw, tgt), n in rows.most_common():
            L.append(f"| `{raw}` | {tgt} | {n:,} |")
        L.append("")
    L.append("Ghi chú: nhãn gốc dạng `a+b` = xe và người lái đã được ghép thành 1 instance (class 5/7) theo "
             "guideline BTC; `teacher:*` = nhãn do mô hình teacher bổ sung; `sam_box` = mask SAM sinh từ box.")
    return "\n".join(L)


def make_zip(out_dir: Path, classes, has_test: bool, name: str) -> int:
    """du_lieu.zip in the BTC layout (images/, labels/, data.yaml). data.yaml has NO `path` key so
    Ultralytics resolves train.txt / val.txt relative to the yaml after unzipping anywhere.
    The sample notebook's Step 1 must be edited to use this data.yaml instead of re-splitting at random,
    otherwise RFS repeats and the VN-only, group-aware val split are lost."""
    import zipfile
    dy = {"train": "train.txt", "val": "val.txt", "names": {i: n for i, n in enumerate(classes.names)}}
    if has_test:
        dy["test"] = "test.txt"
    zpath = out_dir.parent / name
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        z.writestr("data.yaml", yaml.safe_dump(dy, allow_unicode=True, sort_keys=False))
        for f in ("train.txt", "val.txt", "test.txt"):
            if (out_dir / f).exists():
                z.write(out_dir / f, f)
        for sub in ("images", "labels", "metadata", "stats"):
            for p in sorted((out_dir / sub).rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(out_dir).as_posix())
    return zpath.stat().st_size
