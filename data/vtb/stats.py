"""Dataset statistics per pipeline stage + charts + markdown report."""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SMALL, MEDIUM = 32 ** 2, 96 ** 2  # COCO size buckets, measured at 640 long side

# validated reference palette (dataviz skill, light mode)
C_BEFORE, C_AFTER, C_THIRD = "#2a78d6", "#eb6834", "#1baf7a"
TEXT, TEXT2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ = ["#b9d3f2", "#6fa5e6", "#2a78d6"]  # one hue light->dark for size buckets


def imbalance(values: list[float]) -> dict:
    v = np.asarray(values, np.float64)
    nz = v[v > 0]
    tot = v.sum()
    if tot == 0:
        return {"max_min_ratio": None, "norm_entropy": 0.0, "gini": None, "cv": None, "zero_classes": int((v == 0).sum())}
    p = v / tot
    ent = -sum(x * math.log(x) for x in p if x > 0) / math.log(len(v))
    s = np.sort(v)
    n = len(s)
    gini = float((2 * np.arange(1, n + 1) - n - 1).dot(s) / (n * s.sum()))
    return {
        "max_min_ratio": round(float(nz.max() / nz.min()), 2) if len(nz) else None,
        "norm_entropy": round(float(ent), 4),   # 1.0 = perfectly balanced
        "gini": round(gini, 4),                 # 0.0 = perfectly balanced
        "cv": round(float(v.std() / v.mean()), 4),
        "zero_classes": int((v == 0).sum()),
    }


def compute(records: list[dict], class_names: list[str], weighted: bool = False) -> dict:
    """weighted=True counts each image `repeat` times (effective training distribution)."""
    inst = Counter()
    img_with = Counter()
    by_source = defaultdict(lambda: {"images": 0, "instances": 0})
    by_domain, by_cond, by_weather, by_origin = Counter(), Counter(), Counter(), Counter()
    sizes = {c: Counter() for c in class_names}
    n_img = 0
    for r in records:
        w = r.get("repeat", 1) if weighted else 1
        n_img += w
        cnt = r.get("counts") or Counter(i["cls"] for i in r["instances"])
        for c in class_names:
            k = cnt.get(c, 0)
            inst[c] += k * w
            if k:
                img_with[c] += w
        by_source[r["source"]]["images"] += w
        by_source[r["source"]]["instances"] += sum(cnt.get(c, 0) for c in class_names) * w
        by_domain[r.get("domain", "?")] += w
        by_cond[r.get("condition", "unknown")] += w
        by_weather[r.get("weather", "unknown")] += w
        for it in r["instances"]:
            if it["cls"] in sizes:
                a = it.get("area640", 0)
                sizes[it["cls"]]["small" if a < SMALL else "medium" if a < MEDIUM else "large"] += w
            by_origin[it.get("origin", "human")] += w
    tot = sum(inst.values())
    return {
        "images": n_img,
        "instances": tot,
        "instances_per_image": round(tot / max(1, n_img), 2),
        "per_class": {c: {"instances": inst[c], "share": round(inst[c] / max(1, tot), 4),
                          "images_with": img_with[c], "size": dict(sizes[c])} for c in class_names},
        "imbalance_instances": imbalance([inst[c] for c in class_names]),
        "imbalance_images_with": imbalance([img_with[c] for c in class_names]),
        "per_source": {k: dict(v) for k, v in sorted(by_source.items())},
        "per_domain": dict(by_domain),
        "per_condition": dict(by_cond),
        "per_weather": dict(by_weather),
        "label_origin": dict(by_origin),
    }


def raw_stats(records: list[dict]) -> dict:
    """Before resolve: tokens as mapped from source labels (riders not yet merged)."""
    tok = Counter()
    by_source = defaultdict(lambda: {"images": 0, "instances": 0})
    for r in records:
        by_source[r["source"]]["images"] += 1
        by_source[r["source"]]["instances"] += len(r["instances"])
        for it in r["instances"]:
            tok[it["cls"]] += 1
    return {"images": len(records), "instances": sum(tok.values()), "tokens": dict(tok.most_common()),
            "per_source": {k: dict(v) for k, v in sorted(by_source.items())}}


# --------------------------------------------------------------------------- charts
def _style(ax, title, xlabel=None):
    import matplotlib.ticker as mt  # noqa: F401
    ax.set_title(title, loc="left", fontsize=12, color=TEXT, pad=12, fontweight="bold")
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT2, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT2, fontsize=9)


def _grouped_barh(ax, labels, series: list[tuple[str, list[float], str]], fmt):
    n = len(series)
    h = 0.8 / n
    y = np.arange(len(labels))
    for k, (name, vals, color) in enumerate(series):
        pos = y - 0.4 + h * (k + 0.5)
        ax.barh(pos, vals, height=h * 0.88, color=color, label=name, edgecolor=SURFACE, linewidth=1.0)
        vmax = max(max(vals), 1e-9)
        for p, v in zip(pos, vals):
            ax.text(v + vmax * 0.01, p, fmt(v), va="center", fontsize=7.5, color=TEXT2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=TEXT, fontsize=9)
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=TEXT)


def charts(stages: dict, class_names: list[str], display: dict, out: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    before, after = stages["cleaned"], stages["train_effective"]
    labels = [f"{i} · {display.get(c, c)}" for i, c in enumerate(class_names)]
    files = []

    # 1. class share before vs after
    fig, ax = plt.subplots(figsize=(10, 6.2), facecolor=SURFACE)
    _grouped_barh(ax, labels, [
        ("Trước cân bằng (pool sạch)", [before["per_class"][c]["share"] * 100 for c in class_names], C_BEFORE),
        ("Sau cân bằng (train hiệu dụng)", [after["per_class"][c]["share"] * 100 for c in class_names], C_AFTER),
    ], lambda v: f"{v:.1f}%")
    ib, ia = before["imbalance_instances"], after["imbalance_instances"]
    _style(ax, "Tỉ lệ instance theo class (%) — đường lý tưởng 9.1% (1/11)", "% tổng số instance")
    ax.axvline(100 / len(class_names), color=TEXT2, linestyle=(0, (3, 3)), linewidth=1)
    fig.text(0.01, 0.01, f"Max/min: {ib['max_min_ratio']} → {ia['max_min_ratio']}   ·   "
             f"Entropy chuẩn hoá: {ib['norm_entropy']} → {ia['norm_entropy']}   ·   "
             f"Gini: {ib['gini']} → {ia['gini']}", fontsize=9, color=TEXT2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    files.append(out / "class_share_before_after.png")
    fig.savefig(files[-1], dpi=150)
    plt.close(fig)

    # 2. absolute instances (log scale)
    fig, ax = plt.subplots(figsize=(10, 6.2), facecolor=SURFACE)
    _grouped_barh(ax, labels, [
        ("Trước cân bằng", [max(before["per_class"][c]["instances"], 0.9) for c in class_names], C_BEFORE),
        ("Sau cân bằng (đã tính lặp RFS)", [max(after["per_class"][c]["instances"], 0.9) for c in class_names], C_AFTER),
    ], lambda v: f"{int(v):,}")
    ax.set_xscale("log")
    _style(ax, "Số instance theo class (thang log)", "số instance (log)")
    fig.tight_layout()
    files.append(out / "class_instances_before_after.png")
    fig.savefig(files[-1], dpi=150)
    plt.close(fig)

    # 3. source composition (images)
    srcs = sorted(set(before["per_source"]) | set(after["per_source"]))
    fig, ax = plt.subplots(figsize=(10, max(3.0, 0.55 * len(srcs) + 1.5)), facecolor=SURFACE)
    tb = max(1, before["images"])
    ta = max(1, after["images"])
    _grouped_barh(ax, srcs, [
        ("Trước", [before["per_source"].get(s, {}).get("images", 0) / tb * 100 for s in srcs], C_BEFORE),
        ("Sau (hiệu dụng)", [after["per_source"].get(s, {}).get("images", 0) / ta * 100 for s in srcs], C_AFTER),
    ], lambda v: f"{v:.1f}%")
    _style(ax, "Tỉ lệ ảnh theo nguồn (%)", "% số ảnh")
    fig.tight_layout()
    files.append(out / "source_share_before_after.png")
    fig.savefig(files[-1], dpi=150)
    plt.close(fig)

    # 4. conditions
    keys = ["day", "night"]
    wkeys = ["clear_or_unknown", "rain", "fog", "snow"]
    lab = ["Ban ngày", "Ban đêm", "Trời quang / không rõ", "Mưa", "Sương mù", "Tuyết"]
    vb = [before["per_condition"].get(k, 0) / tb * 100 for k in keys] + \
         [before["per_weather"].get(k, 0) / tb * 100 for k in wkeys]
    va = [after["per_condition"].get(k, 0) / ta * 100 for k in keys] + \
         [after["per_weather"].get(k, 0) / ta * 100 for k in wkeys]
    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor=SURFACE)
    _grouped_barh(ax, lab, [("Trước", vb, C_BEFORE), ("Sau (hiệu dụng)", va, C_AFTER)], lambda v: f"{v:.1f}%")
    _style(ax, "Điều kiện ánh sáng / thời tiết (% số ảnh)", "% số ảnh")
    fig.tight_layout()
    files.append(out / "conditions_before_after.png")
    fig.savefig(files[-1], dpi=150)
    plt.close(fig)

    # 5. object size buckets after balancing (stacked, sequential single hue)
    fig, ax = plt.subplots(figsize=(10, 6.2), facecolor=SURFACE)
    left = np.zeros(len(class_names))
    for k, (b, name) in enumerate([("small", "Nhỏ (<32²)"), ("medium", "Vừa (32²–96²)"), ("large", "Lớn (>96²)")]):
        vals = []
        for c in class_names:
            sz = after["per_class"][c]["size"]
            tot = max(1, sum(sz.values()))
            vals.append(sz.get(b, 0) / tot * 100)
        ax.barh(np.arange(len(class_names)), vals, left=left, color=SEQ[k], label=name,
                edgecolor=SURFACE, linewidth=2, height=0.7)
        left += np.asarray(vals)
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_yticklabels(labels, fontsize=9, color=TEXT)
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.08), labelcolor=TEXT)
    _style(ax, "Phân bố kích thước đối tượng sau cân bằng (đo ở cạnh dài 640px)")
    fig.tight_layout()
    files.append(out / "size_distribution_after.png")
    fig.savefig(files[-1], dpi=150)
    plt.close(fig)
    return files


# --------------------------------------------------------------------------- report
def report_md(stages: dict, logs: dict, class_names: list[str], display: dict, chart_dir_rel: str,
              synthetic: bool = False) -> str:
    L = []
    L.append("# Báo cáo thống kê bộ dữ liệu (tự sinh)\n")
    if synthetic:
        L.append("> ⚠️ **Đây là số liệu từ DỮ LIỆU DEMO TỔNG HỢP** (ảnh hình khối ngẫu nhiên), chỉ để kiểm tra "
                 "pipeline chạy đúng. Chạy lại với dữ liệu thật để có thống kê thật.\n")
    order = ["cleaned", "train_pool", "selected", "train_effective", "val", "test"]
    names = {"cleaned": "Pool sạch (trước cân bằng)", "train_pool": "Pool train hợp lệ",
             "selected": "Đã chọn (chưa lặp)", "train_effective": "Train hiệu dụng (sau RFS)",
             "val": "Val (chỉ VN)", "test": "Test (chỉ VN)"}
    present = [s for s in order if s in stages]
    L.append("## 1. Tổng quan theo giai đoạn\n")
    L.append("| Giai đoạn | Ảnh | Instance | Inst/ảnh | Max/min | Entropy chuẩn hoá ↑ | Gini ↓ | Class = 0 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    if "raw" in stages:
        r = stages["raw"]
        L.append(f"| Thô (sau chuyển đổi) | {r['images']:,} | {r['instances']:,} | – | – | – | – | – |")
    for s in present:
        st = stages[s]
        ib = st["imbalance_instances"]
        L.append(f"| {names[s]} | {st['images']:,} | {st['instances']:,} | {st['instances_per_image']} | "
                 f"{ib['max_min_ratio']} | {ib['norm_entropy']} | {ib['gini']} | {ib['zero_classes']} |")
    L.append("\n## 2. Instance theo class — trước và sau cân bằng\n")
    L.append("| ID | Class | Trước: instance | Trước: % | Trước: số ảnh có | Sau: instance (hiệu dụng) | Sau: % | Sau: số ảnh có |")
    L.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    b, a = stages["cleaned"], stages["train_effective"]
    for i, c in enumerate(class_names):
        pb, pa = b["per_class"][c], a["per_class"][c]
        L.append(f"| {i} | {display.get(c, c)} | {pb['instances']:,} | {pb['share']*100:.1f}% | {pb['images_with']:,} | "
                 f"{pa['instances']:,} | {pa['share']*100:.1f}% | {pa['images_with']:,} |")
    L.append("\n## 3. Theo nguồn\n")
    L.append("| Nguồn | Thô: ảnh | Thô: instance | Sạch: ảnh | Sạch: instance | Train hiệu dụng: ảnh | Train hiệu dụng: instance |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    raw_src = stages.get("raw", {}).get("per_source", {})
    for s in sorted(set(raw_src) | set(b["per_source"])):
        r0 = raw_src.get(s, {})
        r1 = b["per_source"].get(s, {})
        r2 = a["per_source"].get(s, {})
        L.append(f"| {s} | {r0.get('images', 0):,} | {r0.get('instances', 0):,} | {r1.get('images', 0):,} | "
                 f"{r1.get('instances', 0):,} | {r2.get('images', 0):,} | {r2.get('instances', 0):,} |")
    L.append("\n## 4. Miền & điều kiện\n")
    L.append("| Chỉ số | Trước | Sau (hiệu dụng) |")
    L.append("|---|---:|---:|")
    for grp, key in (("Miền", "per_domain"), ("Thời gian", "per_condition"), ("Thời tiết", "per_weather")):
        for k in sorted(set(b[key]) | set(a[key])):
            L.append(f"| {grp}: {k} | {b[key].get(k, 0)/max(1,b['images'])*100:.1f}% | "
                     f"{a[key].get(k, 0)/max(1,a['images'])*100:.1f}% |")
    L.append("\n## 5. Biểu đồ\n")
    for f, cap in (("class_share_before_after.png", "Tỉ lệ instance theo class"),
                   ("class_instances_before_after.png", "Số instance theo class (log)"),
                   ("source_share_before_after.png", "Tỉ lệ ảnh theo nguồn"),
                   ("conditions_before_after.png", "Điều kiện ánh sáng / thời tiết"),
                   ("size_distribution_after.png", "Kích thước đối tượng sau cân bằng")):
        L.append(f"**{cap}**\n\n![{cap}]({chart_dir_rel}/{f})\n")
    L.append("## 6. Nhật ký xử lý (số đối tượng/ảnh bị ghép, bỏ...)\n")
    for stage, lg in logs.items():
        L.append(f"<details><summary><b>{stage}</b></summary>\n\n```json\n{json.dumps(lg, ensure_ascii=False, indent=2)}\n```\n</details>\n")
    return "\n".join(L)
