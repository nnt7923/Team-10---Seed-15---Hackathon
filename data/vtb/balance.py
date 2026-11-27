"""Class balancing = (1) deficit-driven greedy image selection + (2) repeat-factor sampling.

(1) Selection: every class has the same instance target T. One more instance of class c is worth
    deficit_c / T (the fraction of its target still missing), so rare classes dominate the score.
    Classes already beyond overshoot_free_ratio × T cost a small, capped penalty per instance, so an
    image with only surplus cars is rejected but 1 rare object + a few cars is still taken.
    Score is multiplied by a bonus for under-represented conditions (night / rain / fog).
    Vietnamese fully-labeled images are always kept. Per-source caps stop one big foreign
    dataset from dominating.
(2) RFS (Gupta et al., LVIS 2019): image repeat r_i = max_c max(1, sqrt(t / f_c)) where f_c is
    the fraction of selected images containing class c. Implemented as duplicated lines in
    train.txt, so it works with stock Ultralytics. Then VN images are up-weighted until they
    reach `min_domain_share` of the effective training stream.
"""
from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np


def eligible(rec: dict, policy: str) -> tuple[bool, str]:
    if rec.get("needs_mask"):
        return False, "needs_mask (run stage 'sam')"
    if rec.get("missing_classes") and not rec.get("completed") and policy == "require_completion":
        return False, "partial_labels (run stage 'complete')"
    return True, "ok"


def select(records: list[dict], class_names: list[str], cfg: dict, source_caps: dict[str, int | None],
           policy: str, seed: int) -> tuple[list[dict], Counter, list[dict]]:
    bc = cfg["balance"]
    rng = random.Random(seed)
    log = Counter()
    pool = []
    for r in records:
        if r["split"] != "train":
            continue
        ok, why = eligible(r, policy)
        if ok:
            pool.append(r)
        else:
            log[f"ineligible_{why.split()[0]}"] += 1
    C = len(class_names)
    N = np.array([[r["counts"][c] for c in class_names] for r in pool], np.float64).reshape(-1, C)
    has_obj = N.sum(1) > 0
    T = np.full(C, float(bc["target_instances_per_class"]))
    cond_bonus = bc.get("condition_bonus", {})
    bonus = np.array([1.0 + cond_bonus.get(r.get("condition"), 0.0) + cond_bonus.get(r.get("weather"), 0.0)
                      for r in pool])
    always = set(bc.get("always_include_domains") or [])
    chosen = np.zeros(len(pool), bool)
    src_count = Counter()
    for i, r in enumerate(pool):
        if r["domain"] in always and has_obj[i] and not r.get("missing_classes"):
            chosen[i] = True
            src_count[r["source"]] += 1
    log["always_included"] = int(chosen.sum())
    cur = N[chosen].sum(0)
    max_images = int(bc["max_images"])
    lam = float(bc.get("overshoot_penalty", 0.05))
    free = float(bc.get("overshoot_free_ratio", 1.5))
    B = int(bc.get("batch_size", 256))
    rounds = 0
    while chosen.sum() < max_images:
        deficit = np.maximum(T - cur, 0.0)
        if deficit.sum() <= 0:
            log["stop_reason_targets_met"] = 1
            break
        if bc.get("scoring", "deficit_weighted") == "legacy":  # v1 formula, kept for ablation only
            gain = (np.minimum(N, deficit) / T).sum(1)
            excess = (np.maximum(N - deficit, 0.0) / T).sum(1)
        else:
            # value of one more instance of class c = fraction of its target still missing (rarer -> worth more)
            need = deficit / T
            gain = (np.minimum(N, deficit) * need).sum(1)
            # penalty only for classes already far beyond target (> free × T), proportional to how far
            # capped at 1 so one instance of a still-missing class always outweighs a few surplus cars
            over = np.clip(cur / T - free, 0.0, 1.0)
            excess = (N * over).sum(1)
        score = (gain - lam * excess) * bonus
        score[chosen | ~has_obj] = -np.inf
        order = np.argsort(-score)[: B * 4]
        added = 0
        for i in order:
            if not np.isfinite(score[i]) or score[i] <= 0 or added >= B or chosen.sum() >= max_images:
                break
            src = pool[i]["source"]
            cap = source_caps.get(src)
            if cap is not None and src_count[src] >= cap:
                continue
            chosen[i] = True
            src_count[src] += 1
            cur += N[i]
            added += 1
        rounds += 1
        if added == 0:
            log["stop_reason_no_positive_gain"] = 1
            break
    else:
        log["stop_reason_budget"] = 1
    log["greedy_rounds"] = rounds
    sel = [pool[i] for i in np.flatnonzero(chosen)]
    # background images (no objects, fully labeled) to curb false positives
    bg = [pool[i] for i in np.flatnonzero(~has_obj) if not pool[i].get("missing_classes")]
    n_bg = min(len(bg), int(round(bc.get("background_fraction", 0.0) * len(sel))))
    sel += rng.sample(bg, n_bg)
    log["background_added"] = n_bg
    log["selected"] = len(sel)
    unchosen = [pool[i] for i in np.flatnonzero(~chosen & has_obj)]
    return sel, log, unchosen


def repeat_factors(selected: list[dict], class_names: list[str], cfg: dict, seed: int) -> Counter:
    bc = cfg["balance"]
    t, rmax = float(bc["rfs_threshold"]), int(bc["rfs_max_repeat"])
    rng = random.Random(seed)
    n = max(1, len(selected))
    f = {c: sum(1 for r in selected if r["counts"][c] > 0) / n for c in class_names}
    r_c = {c: (max(1.0, math.sqrt(t / f[c])) if f[c] > 0 else 1.0) for c in class_names}
    log = Counter()
    for r in selected:
        ri = max([r_c[c] for c in class_names if r["counts"][c] > 0] or [1.0])
        rep = int(math.floor(ri)) + (1 if rng.random() < ri - math.floor(ri) else 0)
        r["repeat"] = max(1, min(rmax, rep))
        r["rfs_factor"] = round(ri, 3)
    # domain share correction (repeat VN images until they reach the requested share)
    for dom, share in (bc.get("min_domain_share") or {}).items():
        d_sum = sum(r["repeat"] for r in selected if r["domain"] == dom)
        o_sum = sum(r["repeat"] for r in selected if r["domain"] != dom)
        if d_sum == 0 or d_sum / max(1, d_sum + o_sum) >= share:
            continue
        k = share * o_sum / ((1 - share) * d_sum)
        for r in selected:
            if r["domain"] == dom:
                r["repeat"] = min(rmax * 3, max(1, int(round(r["repeat"] * k))))
        log[f"domain_upweight_{dom}"] = round(k, 3)
    log["effective_images"] = sum(r["repeat"] for r in selected)
    for c in class_names:
        log[f"rfs_class_factor_{c}"] = round(r_c[c], 3)
    return log
