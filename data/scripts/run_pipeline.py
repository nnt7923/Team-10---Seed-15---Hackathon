"""End-to-end builder for the balanced VN traffic instance-segmentation dataset.

Usage:
  python scripts/run_pipeline.py --config configs/pipeline.yaml                # all stages
  python scripts/run_pipeline.py --config configs/pipeline.yaml --stages balance,report,export

Stages (each writes a checkpoint in work_dir, so you can re-run from any stage):
  convert  -> 01_converted.jsonl   read every enabled source into the unified record format
  resolve  -> 02_resolved.jsonl    rider+vehicle merge, token -> class, tiny-object filter
  sam      -> 02_resolved.jsonl    (GPU) masks for box-only sources
  complete -> 02_resolved.jsonl    (GPU) teacher fills classes a source never labeled
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

ALL = ["convert", "resolve", "sam", "complete", "clean", "balance", "report", "export"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--stages", default=",".join(ALL))
    ap.add_argument("--synthetic", action="store_true", help="mark report as synthetic demo data")
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

    if "convert" in todo:
        t = time.time()
        recs, lg = [], Counter()
        for s in sources:
            n0 = len(recs)
            for r in convert_source(s, data_root):
                recs.append(r)
                cap = s.get("max_images_raw")
                if cap and len(recs) - n0 >= cap:
                    break
            lg[f"images_{s['id']}"] = len(recs) - n0
            print(f"[convert] {s['id']:<28} {len(recs) - n0:>8,} images")
        write_jsonl(work / "01_converted.jsonl", recs)
        stages["raw"] = S.raw_stats(recs)
        logs["convert"] = dict(lg) | {"seconds": round(time.time() - t, 1)}
        save()

    if "resolve" in todo:
        lg = Counter()
        recs = [resolve_record(r, classes, cls_cfg, lg) for r in read_jsonl(work / "01_converted.jsonl")]
        write_jsonl(work / "02_resolved.jsonl", recs)
        logs["resolve"] = dict(lg)
        print(f"[resolve] {dict(lg)}")
        save()

    if "sam" in todo:
        recs = list(read_jsonl(work / "02_resolved.jsonl"))
        need = sum(1 for r in recs if r.get("needs_mask"))
        if need and cfg.get("sam_model"):
            from vtb.labeling import sam_masks
            logs["sam"] = dict(sam_masks(recs, cfg["sam_model"]))
            write_jsonl(work / "02_resolved.jsonl", recs)
        elif need:
            print(f"[sam] {need} images need masks but sam_model is not set -> they stay ineligible")
        save()

    if "complete" in todo:
        recs = list(read_jsonl(work / "02_resolved.jsonl"))
        need = sum(1 for r in recs if r.get("missing_classes") and not r.get("completed"))
        if need and cfg.get("teacher_model"):
            from vtb.labeling import teacher_complete
            logs["complete"] = dict(teacher_complete(recs, classes.names, cfg["teacher_model"], cfg["teacher_conf"]))
            write_jsonl(work / "02_resolved.jsonl", recs)
        elif need:
            print(f"[complete] {need} partially-labeled images, teacher_model not set "
                  f"-> policy '{cfg['partial_label_policy']}' applies")
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
        lg = Q.assign_splits(recs, val_src, cfg["split"]["val_fraction"], cfg["split"]["test_fraction"], cfg["seed"])
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
