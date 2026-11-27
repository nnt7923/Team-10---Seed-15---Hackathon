"""End-to-end builder for the balanced VN traffic instance-segmentation dataset.

Usage:
  python scripts/run_pipeline.py --config configs/pipeline.yaml                # all stages
  python scripts/run_pipeline.py --config configs/pipeline.yaml --stages balance,report,export

Stages (each writes a checkpoint in work_dir, so you can re-run from any stage):
  convert  -> converted/<src>.jsonl  read each enabled source into the unified record format
  sam      -> converted/<src>.jsonl  (GPU) masks for box-only sources
  complete -> converted/<src>.jsonl  (GPU) teacher fills classes a source never labeled
             (these three accept --sources a,b to process datasets one at a time)
  resolve  -> 02_resolved.jsonl    rider+vehicle merge, de-overlap, token -> class, tiny-object filter
  clean    -> 03_clean.jsonl       pHash scan, day/night/weather tags, near-dup removal, group cap
  balance  -> 04_final.jsonl       group split (val/test = VN only), greedy selection, RFS repeats
  report   -> <out>/stats/         statistics before/after + charts + REPORT.md
  export   -> <out>/               YOLO-seg images/labels, train.txt (with repeats), data.yaml, metadata/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from vtb import balance as B  # noqa: E402
from vtb import quality as Q  # noqa: E402
from vtb import stats as S  # noqa: E402
from vtb.common import load_pipeline, read_jsonl, write_jsonl  # noqa: E402
from vtb.converters import convert_source  # noqa: E402
from vtb.resolve import resolve_record  # noqa: E402

def _resolve_worker(rec, classes, cls_cfg):
    lg = Counter()
    return resolve_record(rec, classes, cls_cfg, lg), lg


ALL = ["convert", "sam", "complete", "resolve", "clean", "balance", "report", "export"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--stages", default=",".join(ALL))
    ap.add_argument("--synthetic", action="store_true", help="mark report as synthetic demo data")
    ap.add_argument("--sources", default="", help="convert/sam/complete only these source ids (comma list)")
    a = ap.parse_args()
    cfg, classes, sources, base = load_pipeline(a.config)
    registry = {s["id"]: s for s in sources}
    work, out, data_root = Path(cfg["work_dir"]), Path(cfg["out_dir"]), Path(cfg["data_root"])
    work.mkdir(parents=True, exist_ok=True)
    logs_path, stages_path = work / "logs.json", work / "stages.json"
    logs = json.loads(logs_path.read_text(encoding="utf-8")) if logs_path.exists() else {}
    stages = json.loads(stages_path.read_text(encoding="utf-8")) if stages_path.exists() else {}
    todo = [s.strip() for s in a.stages.split(",") if s.strip()]
    cls_cfg = classes.cfg

    def save():
        logs_path.write_text(json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8")
        stages_path.write_text(json.dumps(stages, ensure_ascii=False, indent=2), encoding="utf-8")

    if not sources:
        sys.exit("No enabled sources. Set enabled: true in the sources registry after downloading data.")

    # convert / sam / complete work PER SOURCE (work/converted/<source>.jsonl), so each dataset can be
    # processed as soon as it is downloaded (--sources a,b). Later stages always combine every source.
    conv_dir = work / "converted"
    conv_dir.mkdir(parents=True, exist_ok=True)
    picked = [s for s in sources if not a.sources or s["id"] in a.sources.split(",")]

    if "convert" in todo:
        t = time.time()
        lg = Counter()
        for s in picked:
            recs = []
            for r in convert_source(s, data_root):
                recs.append(r)
                cap = s.get("max_images_raw")
                if cap and len(recs) >= cap:
                    break
            if s.get("sample_images") and len(recs) > s["sample_images"]:  # random subset (seeded)
                import random
                random.Random(cfg["seed"]).shuffle(recs)
                recs = recs[: s["sample_images"]]
            write_jsonl(conv_dir / f"{s['id']}.jsonl", recs)
            lg[f"images_{s['id']}"] = len(recs)
            print(f"[convert] {s['id']:<28} {len(recs):>8,} images", flush=True)
        logs["convert"] = logs.get("convert", {}) | dict(lg) | {"seconds_last_run": round(time.time() - t, 1)}
        save()

    if "sam" in todo:
        from vtb.labeling import needs_mask, sam_masks
        for s in picked:
            f = conv_dir / f"{s['id']}.jsonl"
            if not f.exists():
                continue
            recs = list(read_jsonl(f))
            need = sum(1 for r in recs if needs_mask(r))
            if need and cfg.get("sam_model"):
                logs.setdefault("sam", {})[s["id"]] = dict(sam_masks(recs, cfg["sam_model"]))
                write_jsonl(f, recs)
            elif need:
                print(f"[sam] {s['id']}: {need} images need masks but sam_model is not set -> they stay ineligible")
        save()

    if "complete" in todo:
        only = set(cfg.get("teacher_sources") or [s["id"] for s in sources])
        for s in picked:
            f = conv_dir / f"{s['id']}.jsonl"
            if s["id"] not in only or not f.exists():
                continue
            recs = list(read_jsonl(f))
            need = sum(1 for r in recs if set(classes.names) - set(r.get("labeled_classes") or [])
                       and not r.get("completed"))
            if need and cfg.get("teacher_model"):
                from vtb.labeling import COCO_TEACHER_MAP, teacher_complete
                lm = COCO_TEACHER_MAP if cfg.get("teacher_label_map") == "coco" else None
                logs.setdefault("complete", {})[s["id"]] = dict(teacher_complete(
                    recs, classes.names, cfg["teacher_model"], cfg["teacher_conf"], label_map=lm,
                    imgsz=cfg.get("teacher_imgsz", 1024)))
                write_jsonl(f, recs)
            elif need:
                print(f"[complete] {s['id']}: {need} partially-labeled images, teacher_model not set "
                      f"-> policy '{cfg['partial_label_policy']}' applies")
        save()

    if "resolve" in todo:
        from concurrent.futures import ProcessPoolExecutor
        from functools import partial
        lg = Counter()
        recs = []
        for s in sources:
            f = conv_dir / f"{s['id']}.jsonl"
            if f.exists():
                recs.extend(read_jsonl(f))
            else:
                print(f"[resolve] WARNING {s['id']} enabled but not converted yet -> skipped")
        stages["raw"] = S.raw_stats(recs)
        with ProcessPoolExecutor(cfg.get("workers", 8)) as ex:
            resolved = list(ex.map(partial(_resolve_worker, classes=classes, cls_cfg=cls_cfg), recs, chunksize=16))
        recs = [r for r, _ in resolved]
        for _, l in resolved:
            lg.update(l)
        write_jsonl(work / "02_resolved.jsonl", recs)
        logs["resolve"] = dict(lg)
        print(f"[resolve] {dict(lg)}")
        save()

    if "clean" in todo:
        recs = list(read_jsonl(work / "02_resolved.jsonl"))
        lg = Q.scan(recs, cfg.get("workers", 8))
        Q.tag_conditions(recs, cfg["conditions"]["night_luma_thr"])
        prio = {s["id"]: s.get("priority", 0) for s in sources}
        kept, lg2 = Q.dedupe(recs, prio, cfg["dedupe"]["hamming_thr"], cfg["dedupe"].get("max_per_group"))
        write_jsonl(work / "03_clean.jsonl", kept)
        logs["clean"] = dict(lg + lg2)
        print(f"[clean] {dict(lg + lg2)}")
        save()

    if "balance" in todo:
        recs = list(read_jsonl(work / "03_clean.jsonl"))
        val_src = {s["id"] for s in sources if s.get("use_for_val")}
        lg = Q.assign_splits(recs, val_src, cfg["split"]["val_fraction"], cfg["split"]["test_fraction"], cfg["seed"],
                             fallback=cfg["split"].get("val_fallback", True))
        caps = {s["id"]: s.get("max_images") for s in sources}
        sel, lg2, unchosen = B.select(recs, classes.names, cfg, caps, cfg["partial_label_policy"], cfg["seed"])
        lg3 = B.repeat_factors(sel, classes.names, cfg, cfg["seed"])
        pool = [r for r in recs if r["split"] == "train" and B.eligible(r, cfg["partial_label_policy"])[0]]
        val = [r for r in recs if r["split"] == "val"]
        test = [r for r in recs if r["split"] == "test"]
        stages["cleaned"] = S.compute(recs, classes.names)
        stages["train_pool"] = S.compute(pool, classes.names)
        stages["selected"] = S.compute(sel, classes.names)
        stages["train_effective"] = S.compute(sel, classes.names, weighted=True)
        stages["val"] = S.compute(val, classes.names)
        stages["test"] = S.compute(test, classes.names)
        write_jsonl(work / "04_final.jsonl", sel + val + test)
        (work / "unselected_uids.txt").write_text("\n".join(r["uid"] for r in unchosen), encoding="utf-8")
        logs["balance"] = dict(lg) | dict(lg2) | dict(lg3)
        print(f"[balance] split={dict(lg)} select={dict(lg2)}")
        ib, ia = stages["cleaned"]["imbalance_instances"], stages["train_effective"]["imbalance_instances"]
        print(f"[balance] max/min {ib['max_min_ratio']} -> {ia['max_min_ratio']} | "
              f"entropy {ib['norm_entropy']} -> {ia['norm_entropy']} | gini {ib['gini']} -> {ia['gini']}")
        save()

    if "report" in todo:
        sd = out / "stats"
        S.charts(stages, classes.names, classes.display, sd)
        (sd / "REPORT.md").write_text(
            S.report_md(stages, logs, classes.names, classes.display, ".", synthetic=a.synthetic), encoding="utf-8")
        (sd / "stats.json").write_text(json.dumps(stages, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[report] -> {sd}")

    if "export" in todo:
        from vtb.export import export
        recs = list(read_jsonl(work / "04_final.jsonl"))
        uns = (work / "unselected_uids.txt").read_text(encoding="utf-8").splitlines()
        lg = export(recs, classes, registry, cfg, out, uns, stages, logs)
        logs["export"] = dict(lg)
        print(f"[export] {dict(lg)} -> {out}")
        save()


if __name__ == "__main__":
    main()
