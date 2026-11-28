"""Image scan (pHash + brightness), condition tagging, near-duplicate removal, splitting."""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import imagehash
import numpy as np
from PIL import Image


# --------------------------------------------------------------------------- scan
def _scan_one(path: str) -> tuple[str | None, float | None]:
    try:
        with Image.open(path) as im:
            im.draft("RGB", (256, 256))  # fast JPEG downscale
            im = im.convert("RGB")
            ph = str(imagehash.phash(im))
            luma = float(np.asarray(im.convert("L").resize((64, 64))).mean())
        return ph, luma
    except Exception:
        return None, None


def scan(records: list[dict], workers: int = 8) -> Counter:
    log = Counter()
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(_scan_one, [r["image"] for r in records]))
    for r, (ph, luma) in zip(records, res):
        r["phash"], r["luma"] = ph, luma
        log["scan_ok" if ph else "scan_unreadable"] += 1
    return log


def tag_conditions(records: list[dict], night_thr: float) -> None:
    """timeofday/weather: trust source attributes; fall back to a brightness heuristic for time of day."""
    for r in records:
        a = r.setdefault("attrs", {})
        w = str(a.get("weather", "")).lower()
        tod = str(a.get("timeofday", "")).lower()
        if w == "night":  # ACDC encodes night as a condition folder
            tod, w = "night", ""
        if not tod or tod == "undefined":
            tod = "night" if (r.get("luma") is not None and r["luma"] < night_thr) else "day"
            r["attrs"]["timeofday_source"] = "luma_heuristic"
        cond = "night" if tod == "night" else "day"
        if "rain" in w:
            cond_w = "rain"
        elif "fog" in w:
            cond_w = "fog"
        elif "snow" in w:
            cond_w = "snow"
        else:
            cond_w = "clear_or_unknown"
        r["attrs"]["timeofday"] = tod
        r["condition"] = cond
        r["weather"] = cond_w


# --------------------------------------------------------------------------- dedupe
def dedupe(records: list[dict], priority: dict[str, int], hamming_thr: int = 4,
           max_per_group: int | None = None) -> tuple[list[dict], Counter]:
    """Near-duplicate removal with pHash + LSH (4 bands of 16 bits; pigeonhole => recall for thr<=3,
    near-complete for 4). Keeps the record with the highest (source priority, #instances)."""
    log = Counter()
    # copy-paste images are near-duplicates of their background BY DESIGN -> never clustered (v3)
    synth = [r for r in records if r.get("synthetic") and r.get("phash")]
    records = [r for r in records if not r.get("synthetic")]
    recs = [r for r in records if r.get("phash")]
    log["dropped_unreadable"] = len(records) - len(recs)
    ints = [int(r["phash"], 16) for r in recs]
    parent = list(range(len(recs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    buckets = defaultdict(list)
    for i, h in enumerate(ints):
        for b in range(4):
            buckets[(b, (h >> (16 * b)) & 0xFFFF)].append(i)
    for idx in buckets.values():
        if len(idx) < 2 or len(idx) > 5000:
            continue
        for a_i in range(len(idx)):
            for b_i in range(a_i + 1, len(idx)):
                a, b = idx[a_i], idx[b_i]
                if bin(ints[a] ^ ints[b]).count("1") <= hamming_thr:
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[rb] = ra
    clusters = defaultdict(list)
    for i in range(len(recs)):
        clusters[find(i)].append(i)
    kept = []
    for members in clusters.values():
        best = max(members, key=lambda i: (priority.get(recs[i]["source"], 0), len(recs[i]["instances"])))
        r = recs[best]
        r["dup_cluster_size"] = len(members)
        if len(members) > 1:
            r["dup_of"] = [recs[i]["uid"] for i in members if i != best][:20]
            log["dropped_near_duplicate"] += len(members) - 1
        kept.append(r)
    if max_per_group:
        per = defaultdict(list)
        for r in kept:
            per[r["group"]].append(r)
        kept = []
        for g, rs in per.items():
            rs.sort(key=lambda r: r["uid"])
            if len(rs) > max_per_group:  # uniform temporal subsample
                step = len(rs) / max_per_group
                rs_keep = [rs[int(i * step)] for i in range(max_per_group)]
                log["dropped_group_cap"] += len(rs) - len(rs_keep)
                rs = rs_keep
            kept.extend(rs)
    kept.extend(synth)
    log["kept_synthetic"] = len(synth)
    kept.sort(key=lambda r: r["uid"])
    log["kept"] = len(kept)
    return kept, log


# --------------------------------------------------------------------------- split
def _hash01(s: str, seed: int) -> float:
    return int(hashlib.md5(f"{seed}:{s}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def assign_splits(records: list[dict], val_sources: set[str], val_frac: float, test_frac: float,
                  seed: int, fallback: bool = True) -> Counter:
    """Group-aware split. val/test come ONLY from `use_for_val` sources (Vietnamese, fully labeled)
    so the metric mirrors the competition test set. Everything else is train-pool.

    Fallback (no fully-labeled VN source available yet): val/test are drawn by group from ALL fully
    labeled sources, and the log says so — the metric then measures foreign-street performance and
    must be cross-checked with the public leaderboard."""
    log = Counter()
    eligible_val = {r["source"] for r in records if r["source"] in val_sources and not r.get("missing_classes")}
    if not eligible_val and fallback:
        val_sources = {r["source"] for r in records if not r.get("missing_classes") and not r.get("completed")}
        log["val_fallback_all_complete_sources"] = 1
    for r in records:
        if r["source"] in val_sources and not r.get("missing_classes") and not r.get("completed"):
            u = _hash01(r["group"], seed)
            r["split"] = "test" if u < test_frac else ("val" if u < test_frac + val_frac else "train")
        else:
            r["split"] = "train"
        log[r["split"]] += 1
    return log
