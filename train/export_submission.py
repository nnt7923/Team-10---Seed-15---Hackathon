"""Package a checkpoint for submission with the organisers' notebook, Step 4 (+ optional Step 5).

  python train/export_submission.py --weights best.pt --out submissions/v1_ep2
  python train/export_submission.py --weights best.pt --out submissions/v1_ep2 --val-dir <notebook dataset dir>

Step 4 (make_submission) is the notebook's mandatory, unmodified cell: export ONNX with the fixed contract,
validate it with the embedded platform checker, build submission.zip (fails loudly if anything is off).
With --val-dir, Step 5 also runs: platform-style inference + mask mAP on up to --n-val val images.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import NOTEBOOK, load_notebook_library, windows_resource_shim  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-dir", default=None, help="notebook dataset dir with images/val + labels/val")
    ap.add_argument("--n-val", type=int, default=300)
    a = ap.parse_args()
    windows_resource_shim()
    import os
    if os.name == "nt":  # the scorer symlinks val images; Windows needs admin for that -> copy instead
        os.symlink = lambda src, dst, *args, **kw: shutil.copyfile(src, dst)
    g = {"__name__": "__notebook__"}
    load_notebook_library(NOTEBOOK, g)
    from ultralytics import YOLO
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    work = out / "_work"
    selfcheck = g["load_selfcheck"](g["SELFCHECK_BUNDLE"], work / "_selfcheck")
    names = YOLO(a.weights).names
    if [names[i] for i in sorted(names)] != list(g["CLASSES"]):
        raise SystemExit(f"class names {names} differ from the competition classes {g['CLASSES']}")
    sub = g["make_submission"](Path(a.weights), work, YOLO, selfcheck)  # notebook Step 4, unchanged
    shutil.copyfile(sub["zip"], out / "submission.zip")
    info = {"weights": str(Path(a.weights).resolve()), "weights_sha256": g["sha256_file"](Path(a.weights)),
            "zip_bytes": (out / "submission.zip").stat().st_size, "validator": sub.get("info")}
    if a.val_dir:
        vd = Path(a.val_dir)
        stems = g["sample_stems"](vd / "images" / "val", a.n_val, 15)
        pred, inf = g["run_platform_inference"](sub["zip"], stems, work, selfcheck)
        res = g["print_score"](g["score_predictions"](pred, vd / "labels" / "val", stems, selfcheck),
                               g["CLASSES"], g["CLASS_DISPLAY_VI"])
        info["val_selfcheck"] = {"map50_95": res["map50_95"], "score_x100": res["score"], "map50": res["map50"],
                                 "per_class_ap": res["per_class_ap"], "n_images": res["n_images"],
                                 "latency_ms_p50": inf["latency_ms_p50"]}
    (out / "export_info.json").write_text(json.dumps(info, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    print(f"[ok] {out / 'submission.zip'}  {info['zip_bytes'] / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
