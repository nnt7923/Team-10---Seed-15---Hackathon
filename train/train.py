"""Train YOLO26n-seg on the team dataset with the organisers' notebook, step for step, and save a version.

  python train/train.py --version v1 --data data/dataset_out --epochs 60
  python train/train.py --version bench --data data/dataset_out --bench          # 1 short epoch to time the GPU

Why a script: the BTC notebook (train_and_export.ipynb) is executed *verbatim* for its library cells (constants,
data checks, ONNX export contract, embedded platform scorer, manifest). Only Step 3 — which the notebook says
teams may edit freely — is customised:
  * train list = our RFS-balanced train.txt (rare-class images repeated), remapped through the notebook's
    source_map.json, because Step 1 copies images/train and drops train.txt;
  * tuned augmentation / schedule (see TRAIN_ARGS below).
Everything the platform checks (YOLO26n-seg, 11 classes, imgsz 640, fp32 ONNX opset 17, seed) is unchanged.

Outputs are copied to models/<version>/: best.pt, submission.zip, run_report.json, results.csv, args.yaml,
train_args.json, val_selfcheck.json.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = Path(os.environ.get("BTC_NOTEBOOK", REPO.parent / "train_and_export.ipynb"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def windows_resource_shim() -> None:
    """The embedded platform scorer imports the Linux-only `resource` module just to log peak RSS
    (resource.getrusage(RUSAGE_SELF).ru_maxrss). Provide that one call on Windows; scoring is unaffected."""
    if os.name != "nt" or "resource" in sys.modules:
        return
    import types
    mod = types.ModuleType("resource")
    mod.RUSAGE_SELF = 0

    def getrusage(_who):
        try:
            import psutil
            kib = psutil.Process().memory_info().peak_wset / 1024
        except Exception:
            kib = 0.0
        return types.SimpleNamespace(ru_maxrss=kib)

    mod.getrusage = getrusage
    sys.modules["resource"] = mod


def load_notebook_library(nb_path: Path, g: dict) -> None:
    """Exec the notebook's constant + library cells (1-5) unchanged into globals ``g``."""
    nb = json.load(open(nb_path, encoding="utf-8"))
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    for c in cells[:5]:  # constants, library 1/4 .. 4/4
        exec(compile("".join(c["source"]), f"<notebook {nb_path.name}>", "exec"), g)


def rfs_train_list(dataset_out: Path, nb_dataset: Path) -> Path:
    """Map our train.txt (with RFS repeats) to the image paths the notebook created in Step 1."""
    smap = json.load(open(nb_dataset / "source_map.json", encoding="utf-8"))["files"]
    by_stem = {p.stem: p.resolve() for p in (nb_dataset / "images" / "train").iterdir()}  # list the folder once
    rel2new = {}
    for key, rel in smap.items():
        split, stem = key.split("/", 1)
        if split == "train" and stem in by_stem:
            rel2new[rel.replace("\\", "/")] = by_stem[stem]
    lines, missing = [], 0
    for line in (dataset_out / "train.txt").read_text(encoding="utf-8").splitlines():
        rel = line.strip().lstrip("./")
        if not rel:
            continue
        p = rel2new.get(rel)
        if p is None:
            missing += 1
            continue
        lines.append(str(p))
    if missing:
        print(f"[rfs] {missing} lines of train.txt not found in the notebook dataset (skipped)")
    out = nb_dataset / "train_rfs.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[rfs] train list: {len(lines)} lines for {len(set(lines))} unique images -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--data", default=str(REPO / "data" / "dataset_out"))
    ap.add_argument("--work", default=str(Path.home() / "hackathon_work"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-rfs", action="store_true", help="train on images/train only (notebook default)")
    ap.add_argument("--bench", action="store_true", help="1 epoch on 5%% of train to measure speed")
    ap.add_argument("--resume", action="store_true", help="resume an interrupted training run")
    ap.add_argument("--reuse-data", action="store_true", help="skip Step 1-2 if the notebook dataset already exists")
    ap.add_argument("--install", action="store_true", help="run the notebook's pinned pip install (Colab/Kaggle)")
    ap.add_argument("--project", default=None, help="Ultralytics runs dir (e.g. on Google Drive, survives disconnects)")
    ap.add_argument("--save-dir", default=None, help="where to copy the final version (default: <repo>/models)")
    ap.add_argument("--init-weights", default=None,
                    help="start from our own 11-class checkpoint (fine-tune) instead of the pretrained yolo26n-seg.pt")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a TRAIN_ARGS entry, e.g. --set scale=0.8 --set close_mosaic=3 (repeatable)")
    a = ap.parse_args()

    windows_resource_shim()
    g = {"__name__": "__notebook__"}
    load_notebook_library(NOTEBOOK, g)
    work = Path(a.work) / a.version
    # ---- the notebook's parameter cell, filled in ----
    g.update(TEAM_ID=15, DATASET_PATH=a.data, MODEL_SIZE="n", EPOCHS=a.epochs, BATCH=a.batch, PATIENCE=30,
             TRAIN_IMGSZ=640, WORKERS=a.workers, CACHE=False, PLOTS=True, VAL_RATIO=0.2, MIN_TRAIN_INSTANCES=100,
             MAX_IMBALANCE_RATIO=30, DUP_HAMMING_MAX=2, REMOVE_LEAKAGE_FROM_TRAIN=False, SELFCHECK_MAX_IMAGES=300,
             CONF_SHOW=0.25, N_ERROR_IMAGES=6, INSTALL_PACKAGES=a.install, REQUIRE_GPU=True, DRIVE_OUTPUT_DIR="",
             WORK_DIR=str(work))
    g["MODEL_WEIGHTS"] = f"yolo26{g['MODEL_SIZE']}-seg.pt"
    P = {k: g[k] for k in ("TEAM_ID", "MODEL_SIZE", "MODEL_WEIGHTS", "EPOCHS", "BATCH", "PATIENCE", "TRAIN_IMGSZ",
                           "WORKERS", "CACHE", "VAL_RATIO", "MIN_TRAIN_INSTANCES", "MAX_IMBALANCE_RATIO",
                           "DUP_HAMMING_MAX", "REMOVE_LEAKAGE_FROM_TRAIN", "SELFCHECK_MAX_IMAGES", "CONF_SHOW")}
    g["PARAMS"] = P
    t0 = time.time()

    # ---- Step 0: setup (notebook cell, unchanged) ----
    SEED = g["validate_team_id"](g["TEAM_ID"])
    g["check_model_choice"](g["MODEL_SIZE"], g["MODEL_WEIGHTS"])
    if g["INSTALL_PACKAGES"]:
        g["install_requirements"](g["INSTALL_SPECS"])
    ENV = g["check_environment"]()
    DEVICE, GPU_NAME = g["detect_device"](g["REQUIRE_GPU"])
    print(f"Thiết bị huấn luyện: GPU {GPU_NAME}")
    g["seed_everything"](SEED)
    WORK = Path(g["WORK_DIR"])
    WORK.mkdir(parents=True, exist_ok=True)
    SELFCHECK = g["load_selfcheck"](g["SELFCHECK_BUNDLE"], WORK / "_selfcheck")

    # ---- Step 1-2: data (notebook cells, unchanged) ----
    nb_ds = WORK / "dataset"
    if (a.reuse_data or a.resume) and (WORK / "dataset_state.json").exists():
        DATASET = json.load(open(WORK / "dataset_state.json", encoding="utf-8"))
        REPORT = json.load(open(WORK / "data_report.json", encoding="utf-8"))
    else:
        RAW_DIR = g["locate_dataset"](g["DATASET_PATH"], WORK)
        DATASET = g["prepare_dataset"](RAW_DIR, nb_ds, g["CLASSES"], SEED, g["VAL_RATIO"])
        C = g
        REPORT = C["analyze_dataset"](DATASET["out_dir"], C["CLASSES"], C["MIN_GT_AREA_PX"], C["IMGSZ"],
                                      C["MIN_TRAIN_INSTANCES"], C["MAX_IMBALANCE_RATIO"], C["DUP_HAMMING_MAX"],
                                      C["MIN_GT_PER_CLASS"], C["MAX_DET"])
        C["print_report"](REPORT, C["CLASSES"], C["CLASS_DISPLAY_VI"])
        DATASET["counts"] = {s: {"images": v["images"], "instances": v["instances"]} for s, v in REPORT["splits"].items()}
        C["make_previews"](DATASET["out_dir"], C["CLASSES"], WORK / "report" / "preview_nhan.png", n=8, seed=SEED)
        if not REPORT["ok"]:
            raise C["NotebookError"]("Dữ liệu có lỗi nặng:\n  - " + "\n  - ".join(REPORT["fatal"][:10]), "DATA")
        DATASET["sha256"] = C["dataset_fingerprint"](DATASET["out_dir"])
        json.dump(DATASET, open(WORK / "dataset_state.json", "w", encoding="utf-8"), default=str, ensure_ascii=False)
        json.dump(REPORT, open(WORK / "data_report.json", "w", encoding="utf-8"), default=str, ensure_ascii=False)
    print(f"[data] {DATASET['counts']}  sha256={DATASET['sha256'][:16]}")

    # ---- Step 3: training (customised, allowed by the notebook) ----
    from ultralytics import YOLO
    data_yaml = Path(DATASET["data_yaml"])
    if not a.no_rfs:
        train_list = rfs_train_list(Path(a.data), nb_ds)
        names = "\n".join(f"  {i}: {n}" for i, n in enumerate(g["CLASSES"]))
        data_yaml = nb_ds / "data_rfs.yaml"
        data_yaml.write_text(f"path: {nb_ds.resolve()}\ntrain: {train_list.name}\nval: images/val\nnames:\n{names}\n",
                             encoding="utf-8")
    TRAIN_ARGS = dict(
        data=str(data_yaml), imgsz=640, epochs=1 if a.bench else g["EPOCHS"], batch=g["BATCH"], seed=SEED,
        deterministic=True, device=DEVICE, workers=g["WORKERS"], patience=g["PATIENCE"], cache=False, plots=True,
        project=a.project or str(WORK / "runs"), name=a.version, exist_ok=True,
        # schedule: SGD (auto for long runs) + cosine decay, warmup 3 epochs, no mosaic for the last 10 epochs
        optimizer="auto", cos_lr=True, warmup_epochs=3, close_mosaic=10,
        # augmentation tuned for instance segmentation of road scenes with rare classes:
        mosaic=1.0, copy_paste=0.3,           # copy-paste pastes masked instances -> more rare-class examples
        mixup=0.0,                            # blending hurts mask edges at mAP@.5:.95 on a nano model
        scale=0.5, translate=0.1, degrees=0.0, shear=0.0, perspective=0.0,  # keep road geometry
        fliplr=0.5, flipud=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,          # day/night/rain colour variety
        overlap_mask=True, mask_ratio=4,
        amp=True,
    )
    for kv in a.set:
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)
        except ValueError:
            pass
        TRAIN_ARGS[k] = v
        print(f"[set] {k} = {v!r}")
    if a.bench:
        TRAIN_ARGS.update(fraction=0.05, plots=False, val=False, name="bench")
    g["seed_everything"](SEED)
    last = Path(TRAIN_ARGS["project"]) / TRAIN_ARGS["name"] / "weights" / "last.pt"
    if a.resume and last.exists():  # continue an interrupted run (e.g. Colab disconnect)
        model = YOLO(str(last))
        model.train(resume=True)
    else:
        model = YOLO(a.init_weights or g["MODEL_WEIGHTS"])
        if a.init_weights:
            print(f"[init] fine-tuning from {a.init_weights}")
        model.train(**TRAIN_ARGS)
    if a.bench:
        print(f"[bench] total {time.time() - t0:.0f}s")
        return
    BEST = Path(model.trainer.best)
    trained = [model.names[i] for i in sorted(model.names)]
    if trained != list(g["CLASSES"]):
        raise g["NotebookError"](f"Tên lớp {trained} khác {list(g['CLASSES'])}", "TRAIN")
    TRAIN_INFO = {"best_weights": str(BEST), "best_sha256": g["sha256_file"](BEST),
                  "best_tensors_sha256": g["weights_fingerprint"](BEST),
                  "train_args": {k: v for k, v in TRAIN_ARGS.items() if k != "project"}, "device": DEVICE,
                  "gpu": GPU_NAME, "rfs_train_list": not a.no_rfs}

    # ---- Step 4: export + platform self-check (notebook, unchanged) ----
    SUB = g["make_submission"](BEST, WORK, YOLO, SELFCHECK)

    # ---- Step 5: score like the platform on val (notebook, unchanged) ----
    VAL_IMAGES = g["sample_stems"](WORK / "dataset" / "images" / "val", g["SELFCHECK_MAX_IMAGES"], SEED)
    VAL_LABELS = WORK / "dataset" / "labels" / "val"
    PRED, INF = g["run_platform_inference"](SUB["zip"], VAL_IMAGES, WORK, SELFCHECK)
    VAL = g["print_score"](g["score_predictions"](PRED, VAL_LABELS, VAL_IMAGES, SELFCHECK), g["CLASSES"], g["CLASS_DISPLAY_VI"])
    try:
        rows, conf = g["analyze_errors"](PRED, VAL_LABELS, SELFCHECK, g["CONF_SHOW"])
        g["print_confusion"](conf, g["CLASSES"])
        g["render_error_images"](rows, PRED, WORK / "dataset" / "images" / "val", VAL_LABELS, g["CLASSES"], SELFCHECK,
                                 WORK / "report" / "loi", g["N_ERROR_IMAGES"], g["CONF_SHOW"])
    except Exception as e:  # analysis is informative only
        print("[warn] error analysis skipped:", e)

    # ---- Step 6: manifest (notebook, unchanged) ----
    EVAL = {"map50_95": VAL["map50_95"], "score_x100": VAL["score"], "map50": VAL["map50"],
            "per_class_ap": VAL["per_class_ap"], "n_images": VAL["n_images"], "latency_ms_p50": INF["latency_ms_p50"],
            "note": "tính trên val của đội bằng bộ chấm nhúng, không phải điểm trên bộ test ẩn"}
    MAN = g["build_manifest"](g["TEAM_ID"], SEED, P, DATASET, REPORT, TRAIN_INFO, SUB, EVAL, ENV, g["CLASSES"])
    man_path = g["write_manifest"](MAN, SUB["zip"].parent / "run_report.json")

    # ---- save the version ----
    dst = Path(a.save_dir or REPO / "models") / a.version
    dst.mkdir(parents=True, exist_ok=True)
    run_dir = BEST.parent.parent
    for src, name in [(BEST, "best.pt"), (SUB["zip"], "submission.zip"), (man_path, "run_report.json"),
                      (run_dir / "results.csv", "results.csv"), (run_dir / "args.yaml", "args.yaml"),
                      (run_dir / "results.png", "results.png")]:
        if Path(src).exists():
            shutil.copyfile(src, dst / name)
    json.dump(TRAIN_INFO, open(dst / "train_args.json", "w", encoding="utf-8"), indent=1, default=str, ensure_ascii=False)
    json.dump(EVAL, open(dst / "val_selfcheck.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"[done] {a.version}: val mask mAP50-95 = {VAL['map50_95']:.4f} (score {VAL['score']:.2f}) -> {dst}"
          f"  total {(time.time() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()
