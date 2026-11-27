"""Simulate the selection step on a LARGE, realistically skewed pool (counts only, no images) to
compare scoring formulas before downloading real data.

  python scripts/simulate_balance.py

Per-image instance rates per source are rough guesses shaped like real street datasets
(cars / pedestrians / motorbikes abundant, other_vehicle & cyclist rare). They are NOT measured
statistics — re-run on the real pool via run_pipeline once data is downloaded.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from vtb import balance as B  # noqa: E402
from vtb.stats import imbalance  # noqa: E402

C = ["car", "bus", "truck", "other_vehicle", "motorcycle", "motorcyclist", "bicycle", "cyclist",
     "pedestrian", "sign", "traffic_light"]
# mean instances / image (Poisson) and #images per simulated source
SOURCES = {
    "mapillary_like": (18000, "global", [7, .4, .9, .08, .6, .5, .4, .25, 3, 4, 1.5]),
    "coco_like":      (6000,  "global", [3, .5, .8, .03, .7, .4, .5, .2, 4, 0, .8]),
    "bdd_like":       (7000,  "global", [8, .3, 1, .05, .1, .1, .1, .1, 1.5, 0, 0]),
    "idd_like":       (14000, "asia",   [3, .5, 1, .6, 1.5, 3, .3, .3, 2, 1, .2]),
    "vn_manual":      (800,   "vn",     [2, .3, .5, .3, 3, 9, .2, .3, 1.5, 1, .3]),
}


def make_pool(seed=15):
    rng = np.random.default_rng(seed)
    recs = []
    for src, (n, dom, lam) in SOURCES.items():
        cnt = rng.poisson(lam, size=(n, len(C)))
        for i in range(n):
            recs.append({"uid": f"{src}/{i}", "source": src, "domain": dom, "split": "train",
                         "counts": dict(zip(C, map(int, cnt[i]))), "instances": [],
                         "missing_classes": [], "condition": "night" if rng.random() < .12 else "day",
                         "weather": "rain" if rng.random() < .06 else "clear_or_unknown"})
    return recs


def run(scoring: str, recs, target=20000, budget=60000):
    cfg = {"balance": {"target_instances_per_class": target, "max_images": budget,
                       "always_include_domains": ["vn"], "condition_bonus": {"night": .25, "rain": .25},
                       "overshoot_penalty": 0.15 if scoring == "legacy" else 0.05,
                       "overshoot_free_ratio": 1.5, "batch_size": 256, "background_fraction": 0,
                       "rfs_threshold": .3, "rfs_max_repeat": 4, "scoring": scoring}}
    for r in recs:
        r.pop("repeat", None)
    sel, log, _ = B.select(recs, C, cfg, {}, "allow", 15)
    B.repeat_factors(sel, C, cfg, 15)
    tot = {c: sum(r["counts"][c] for r in sel) for c in C}
    eff = {c: sum(r["counts"][c] * r["repeat"] for r in sel) for c in C}
    return sel, log, tot, eff


def main():
    recs = make_pool()
    pool = {c: sum(r["counts"][c] for r in recs) for c in C}
    ib = imbalance(list(pool.values()))
    print(f"POOL  {len(recs):>6,} ảnh | max/min {ib['max_min_ratio']:>6} | entropy {ib['norm_entropy']}")
    rows = {"pool": pool}
    for scoring in ("legacy", "deficit_weighted"):
        sel, log, tot, eff = run(scoring, recs)
        im, ie = imbalance(list(tot.values())), imbalance(list(eff.values()))
        n_eff = sum(r["repeat"] for r in sel)
        stop = [k for k in log if k.startswith("stop_reason")]
        print(f"{scoring:<17} {len(sel):>6,} ảnh | max/min {im['max_min_ratio']:>6} | entropy {im['norm_entropy']} | {stop}")
        print(f"{'  + RFS':<17} {n_eff:>6,} dòng| max/min {ie['max_min_ratio']:>6} | entropy {ie['norm_entropy']}")
        rows[scoring] = tot
        rows[scoring + "+RFS"] = eff
    print("\n" + f"{'class':<15}" + "".join(f"{k:>22}" for k in rows))
    for c in C:
        print(f"{c:<15}" + "".join(f"{rows[k][c]:>22,}" for k in rows))


if __name__ == "__main__":
    main()
