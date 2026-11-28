"""v3 setup: human-labelled backgrounds only + offline context copy-paste (see scripts/copy_paste.py).

1. writes configs/{classes,sources,pipeline}_v3.yaml from the v2 files (v1/v2 untouched)
2. imports converted records (work_v2/converted/*.jsonl, built on another machine) into <work>/converted:
   - image paths re-rooted to --raw (Windows -> Linux), IDD images are .jpg here (PNG -> JPEG q95 upload)
   - IDD "vehicle fallback" -> @ignore (ambiguous label; may contain hand carts, which BTC does not label)

  python scripts/v3_prepare.py --raw ~/raw --work ~/v3/work --src ~/v3/work/converted_v2
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
C = ROOT / "configs"
BASE_SOURCES = ["idd_segmentation", "cityscapes", "mapillary_vistas_v2", "mapillary_vistas_v2_poly"]
BANK_SOURCES = ["nuimages", "coco2017_street_yoloseg", "vn_traffic_sign_hf"]


def write_configs(raw: str, work: str, out: str):
    s = (C / "classes_v2.yaml").read_text(encoding="utf-8")
    (C / "classes_v3.yaml").write_text("# v3 = v2 classes (min_area 64 = BTC scorer)\n" + s, encoding="utf-8")

    s = (C / "sources_v2.yaml").read_text(encoding="utf-8")
    s = re.sub(r"\n    enabled: (true|false)[^\n]*", "\n    enabled: false", s)

    def block(sid):
        i = s.index(f"  - id: {sid}\n")
        j = s.find("\n  - id:", i + 5)
        return i, (len(s) if j < 0 else j)

    for sid in BASE_SOURCES:
        i, j = block(sid)
        b = s[i:j].replace("\n    enabled: false", "\n    enabled: true")
        b = re.sub(r"\n    max_images: [^\n]*", "", b)
        if sid == "idd_segmentation":
            b = b.replace('"_leftImg8bit.png"]', '"_leftImg8bit.jpg"]')
            b = b.replace('"vehicle fallback": other_vehicle', '"vehicle fallback": "@ignore"  # v3: nhãn mơ hồ')
        s = s[:i] + b + s[j:]
    s += '''
  # v3: ảnh tổng hợp copy-paste có ngữ cảnh (scripts/copy_paste.py). Nền = ảnh train IDD/Cityscapes (nhãn người,
  # đủ 11 lớp); vật dán = mask người gán (nuImages, COCO) hoặc box người gán + SAM (biển báo VN).
  - id: cp_synth
    name: "Copy-paste có ngữ cảnh (tổng hợp từ IDD/Cityscapes + nuImages/COCO/biển báo VN)"
    tier: B
    domain: asia
    priority: 10
    homepage: "scripts/copy_paste.py"
    download: {method: generated}
    license: "Dẫn xuất từ các nguồn gốc (giữ license nguồn)"
    citation: "Ghiasi et al., Simple Copy-Paste, CVPR 2021 (phương pháp)"
    format: generated
    paths: {images: cp_synth}
    labeled_classes: *ALL
    use_for_val: false
    enabled: true
'''
    s = "# v3 source registry: chỉ nguồn người gán đủ lớp làm ảnh nền + cp_synth. v1/v2 không đổi.\n" + s
    (C / "sources_v3.yaml").write_text(s, encoding="utf-8")

    p = (C / "pipeline_v2.yaml").read_text(encoding="utf-8")
    p = re.sub(r"data_root: [^\n#]*", f"data_root: {raw}   ", p, count=1)
    p = re.sub(r"work_dir: [^\n#]*", f"work_dir: {work}   ", p, count=1)
    p = re.sub(r"out_dir: [^\n#]*", f"out_dir: {out}   ", p, count=1)
    p = p.replace("configs/classes_v2.yaml", "configs/classes_v3.yaml").replace("configs/sources_v2.yaml", "configs/sources_v3.yaml")
    p = re.sub(r"teacher_sources: [^\n]*", "teacher_sources: []", p)
    p = re.sub(r"workers: \d+", "workers: 4", p, count=1)
    p = p.replace("always_include_domains: [vn]", "always_include_domains: []")
    p = p.replace("min_domain_share: {vn: 0.35}", "min_domain_share: {}")
    p = p.replace("zip_name: du_lieu_v2.zip", "zip_name: du_lieu_v3.zip")
    p = "# v3 — nhãn người + copy-paste có ngữ cảnh. Bản v1/v2: pipeline.yaml / pipeline_v2.yaml (không đổi)\n" + p
    (C / "pipeline_v3.yaml").write_text(p, encoding="utf-8")


def import_records(src: Path, dst: Path, raw: str):
    dst.mkdir(parents=True, exist_ok=True)
    for sid in BASE_SOURCES + BANK_SOURCES:
        f = src / f"{sid}.jsonl"
        if not f.exists():
            print("missing", f)
            continue
        n = miss = ign = 0
        with open(f, encoding="utf-8") as fi, open(dst / f"{sid}.jsonl", "w", encoding="utf-8") as fo:
            for line in fi:
                r = json.loads(line)
                p = r["image"].replace("\\", "/")
                k = p.find("/data/raw/")
                p = raw.rstrip("/") + "/" + p[k + len("/data/raw/"):]
                if sid == "idd_segmentation":
                    p = p[:-4] + ".jpg" if p.endswith(".png") else p
                    for it in r["instances"]:
                        if it.get("raw") == "vehicle fallback":
                            it["cls"] = "@ignore"
                            ign += 1
                r["image"] = p
                if not os.path.exists(p):
                    miss += 1
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
        print(f"{sid:<28} {n:>6} records, {miss} images missing, {ign} vehicle-fallback -> @ignore")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", required=True)
    a = ap.parse_args()
    write_configs(a.raw, a.work, a.out)
    import_records(Path(a.src), Path(a.work) / "converted", a.raw)
