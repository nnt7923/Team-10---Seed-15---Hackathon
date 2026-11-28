"""Minimalist HTML overview of the raw data: one file, a few lines of JS for the light/dark switch,
Be Vietnam Pro from Google Fonts with a system-font fallback when offline.

Visual language: warm paper background, black hairlines, square corners, large light headings with a
grey lead line, and "row lists" (label left, description right, separated by 1px rules).
"""
from __future__ import annotations

import html
import math
from datetime import datetime

SRC_COLORS = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)"]  # validated slots 1-4, light/dark steps in CSS
OTHER = "var(--s5)"

CSS = """
/* ---------- theme tokens (light default; dark is its own selected steps, not an inversion) ---------- */
:root,[data-theme=light]{color-scheme:light;--bg:#eeede9;--fg:#111110;--fg2:#55544f;--muted:#86857f;--line:#111110;
--rule:#d3d1ca;--accent:#2a78d6;--bad:#b3261e;--warn:#8a5a00;--good:#1b6b42;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#b7b6b0}
[data-theme=dark]{color-scheme:dark;--bg:#151514;--fg:#eeede8;--fg2:#b7b6ae;--muted:#85847d;--line:#eeede8;
--rule:#34342f;--accent:#3987e5;--bad:#ef6f66;--warn:#e0a050;--good:#5cc28d;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#6d6c66}
/* ---------- type scale: 1.25 ratio on a 16px base, fluid for headings ---------- */
:root{--f-xs:.8125rem;--f-sm:.875rem;--f-md:1rem;--f-lg:1.125rem;--f-xl:1.3125rem;
--f-h3:clamp(1.5rem,1.2rem + 1vw,1.875rem);--f-h2:clamp(2rem,1.4rem + 2vw,2.875rem);
--f-h1:clamp(2.5rem,1.6rem + 3.4vw,4rem);--f-kpi:clamp(1.875rem,1.5rem + 1.2vw,2.5rem);
--font:"Be Vietnam Pro","Segoe UI Variable Text","Segoe UI","Inter","Helvetica Neue",Arial,system-ui,sans-serif;
--mono:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
*{box-sizing:border-box}html{background:var(--bg);scroll-behavior:smooth;transition:background .2s}
body{margin:0;color:var(--fg);font:400 var(--f-md)/1.6 var(--font);font-feature-settings:"kern","liga";
-webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
a{color:inherit;text-decoration:none}
code{font:400 .85em var(--mono);color:var(--fg2)}
b,strong{font-weight:500}
/* ---------- top bar ---------- */
.bar{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line)}
.bar-in{max-width:1280px;margin:0 auto;padding:0 56px;height:76px;display:flex;align-items:center;gap:36px}
.brand{font-size:var(--f-xl);font-weight:500;letter-spacing:-.02em;margin-right:auto;white-space:nowrap}
.nav{display:flex;gap:32px;font-size:var(--f-lg)}
.nav a{padding:6px 0;border-bottom:1px solid transparent}.nav a:hover{border-bottom-color:var(--line)}
.btn{border:1px solid var(--line);padding:11px 24px;font-size:var(--f-md);white-space:nowrap;font-weight:500}
.btn:hover{background:var(--fg);color:var(--bg)}
/* theme switch: square track + knob, label states */
.switch{display:flex;align-items:center;gap:10px;background:none;border:0;padding:0;cursor:pointer;color:var(--fg);
font:500 var(--f-sm) var(--font)}
.switch .track{width:46px;height:24px;border:1px solid var(--line);position:relative;display:inline-block}
.switch .knob{position:absolute;top:3px;left:3px;width:16px;height:16px;background:var(--fg);transition:left .18s ease}
[data-theme=dark] .switch .knob{left:25px}
.switch .lbl-dark{display:none}[data-theme=dark] .switch .lbl-dark{display:inline}[data-theme=dark] .switch .lbl-light{display:none}
.switch:focus-visible{outline:1px solid var(--line);outline-offset:6px}
/* ---------- page ---------- */
main{max-width:1280px;margin:0 auto;padding:0 56px 140px}
section{padding-top:112px}
h1,h2,h3{font-weight:400;margin:0;text-wrap:balance}
h1{font-size:var(--f-h1);letter-spacing:-.04em;line-height:1.04;margin:88px 0 18px}
h2{font-size:var(--f-h2);letter-spacing:-.035em;line-height:1.08;margin-bottom:14px}
h3{font-size:var(--f-h3);letter-spacing:-.025em;line-height:1.2;margin-bottom:8px}
.lead{font-size:var(--f-xl);font-weight:300;line-height:1.5;color:var(--fg2);margin:0;padding-bottom:22px;
border-bottom:1px solid var(--line);text-wrap:pretty}
.meta{font-size:var(--f-sm);color:var(--muted)}
/* ---------- kpi strip ---------- */
.kpis{display:grid;grid-template-columns:repeat(6,1fr);border-top:1px solid var(--line);border-bottom:1px solid var(--line);
margin-top:56px}
.kpi{padding:22px 20px 20px 0;border-right:1px solid var(--rule)}.kpi+.kpi{padding-left:20px}.kpi:last-child{border-right:none}
.kpi b{display:block;font-size:var(--f-kpi);font-weight:300;letter-spacing:-.035em;font-variant-numeric:tabular-nums;line-height:1.1}
.kpi span{font-size:var(--f-sm);color:var(--fg2)}
/* row list (label left, description right) */
.rows{border-top:none}.row{display:flex;justify-content:space-between;align-items:baseline;gap:40px;
padding:24px 0;border-bottom:1px solid var(--line)}
.row .k{font-size:var(--f-lg)}.row .v{font-size:var(--f-lg);color:var(--fg2);text-align:right;font-variant-numeric:tabular-nums}
.row .k i{font-style:normal;display:inline-block;width:9px;height:9px;margin-right:12px;vertical-align:1px;background:var(--warn)}
.row.bad .k i{background:var(--bad)}.row.ok .k i{background:var(--good)}
.cols{display:grid;grid-template-columns:1fr 1fr;column-gap:56px}
.cols .row{padding:18px 0}.cols .row .k,.cols .row .v{font-size:var(--f-md)}
/* tables */
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:var(--f-sm);margin-top:8px}
th{font-weight:400;color:var(--fg2);text-align:right;padding:16px 12px;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left;padding-left:0}
td{padding:14px 12px;text-align:right;border-bottom:1px solid var(--rule)}
tr:last-child td{border-bottom:1px solid var(--line)}
.cell-miss{color:var(--muted)}
/* source block */
.src{padding-top:72px}
.src-head{display:flex;justify-content:space-between;align-items:flex-end;gap:32px;padding-bottom:18px;
border-bottom:1px solid var(--line)}
.src-head .meta a{border-bottom:1px solid var(--rule)}
.tags{display:flex;flex-wrap:wrap;gap:8px;justify-content:flex-end}
.tag{border:1px solid var(--line);padding:4px 12px;font-size:var(--f-xs);white-space:nowrap}
.thumbs{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:28px}
.thumbs img{width:100%;display:block;border:1px solid var(--line)}
details{border-bottom:1px solid var(--line)}
summary{cursor:pointer;list-style:none;display:flex;justify-content:space-between;padding:22px 0;font-size:var(--f-lg)}
summary::-webkit-details-marker{display:none}summary:after{content:"+";font-size:22px;line-height:1}
details[open] summary:after{content:"−"}
.flag{color:var(--bad)}
/* chart */
.legend{display:flex;flex-wrap:wrap;gap:28px;font-size:var(--f-sm);color:var(--fg2);margin:28px 0 18px}
.legend i{display:inline-block;width:12px;height:12px;margin-right:10px;vertical-align:-1px}
.chart{padding:8px 0 4px;border-bottom:1px solid var(--line)}
svg text{font-family:inherit}
.demo{border:1px solid var(--bad);color:var(--bad);padding:14px 18px;margin-top:40px;font-size:var(--f-sm)}
footer{margin-top:96px;padding-top:18px;border-top:1px solid var(--line);font-size:var(--f-sm);color:var(--muted)}
.flow{list-style:none;counter-reset:f;margin:24px 0 0;padding:0;border-top:1px solid var(--line)}
.flow li{counter-increment:f;display:grid;grid-template-columns:56px 160px 1fr;gap:16px;padding:14px 0;border-bottom:1px solid var(--rule)}
.flow li::before{content:counter(f,decimal-leading-zero);color:var(--muted);font-family:var(--mono)}
.flow b{font-family:var(--mono);font-weight:500}.flow span{color:var(--fg2)}
@media (max-width:900px){.flow li{grid-template-columns:40px 1fr}.flow span{grid-column:2}}
@media (max-width:1080px){.nav{display:none}}
@media (max-width:900px){.bar-in,main{padding-left:22px;padding-right:22px}
.kpis{grid-template-columns:repeat(2,1fr)}
.kpi,.kpi+.kpi{padding-left:0;border-right:none;border-bottom:1px solid var(--rule)}
.kpi:nth-child(even){padding-left:20px;border-left:1px solid var(--rule)}.kpi:nth-last-child(-n+2){border-bottom:none}.cols{grid-template-columns:1fr}
.src-head{flex-direction:column;align-items:flex-start}.tags{justify-content:flex-start}.thumbs{grid-template-columns:1fr}}
"""


def _n(x) -> str:
    return f"{int(x):,}".replace(",", " ")


def _gb(b) -> str:
    return f"{b / 1e9:.1f} GB" if b >= 1e9 else f"{b / 1e6:.0f} MB"


def _e(s) -> str:
    return html.escape(str(s if s is not None else "—"))


def _rows(items, cls="rows") -> str:
    """items: (label, value[, level]) -> label-left / value-right rows."""
    out = []
    for it in items:
        k, v = it[0], it[1]
        lvl = it[2] if len(it) > 2 else None
        dot = "<i></i>" if lvl is not None else ""
        out.append(f'<div class="row {lvl or ""}"><div class="k">{dot}{k}</div><div class="v">{v}</div></div>')
    return f'<div class="{cls}">{"".join(out)}</div>'


def coverage_chart(srcs: list[dict], class_names: list[str], display: dict, target: int) -> str:
    """Horizontal stacked bars: estimated instances per class after merge/filter, stacked by source."""
    keys = [s["id"] for s in srcs]
    colors = {k: SRC_COLORS[i] if i < len(SRC_COLORS) else OTHER for i, k in enumerate(keys)}
    tot = {c: sum(s["estimated_classes"].get(c, 0) for s in srcs) for c in class_names}
    vmax = max(max(tot.values() or [1]), target) * 1.1
    W, left, right, row = 1168, 240, 120, 36
    H = row * len(class_names) + 40
    sx = lambda v: left + (W - left - right) * v / vmax  # noqa: E731
    out = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Ước tính số instance theo lớp">']
    step = max(5000, int(round(vmax / 6 / 5000)) * 5000)
    for t in range(0, int(vmax) + 1, step):
        x = sx(t)
        out.append(f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{H - 28}" stroke="var(--rule)"/>'
                   f'<text x="{x:.1f}" y="{H - 8}" font-size="12" fill="var(--muted)" text-anchor="middle">{_n(t)}</text>')
    for i, c in enumerate(class_names):
        y = i * row + 8
        out.append(f'<text x="0" y="{y + 15}" font-size="14" fill="var(--fg)">{i}&#160;&#160;{_e(display.get(c, c))}</text>')
        x = left
        for k in keys:
            v = next(s for s in srcs if s["id"] == k)["estimated_classes"].get(c, 0)
            if v <= 0:
                continue
            w = sx(v) - left
            out.append(f'<rect x="{x:.1f}" y="{y + 2}" width="{max(w - 2, 0.5):.1f}" height="17" '
                       f'fill="{colors[k]}"><title>{_e(k)} · {_e(c)}: {_n(v)}</title></rect>')
            x += w
        low = tot[c] < target * 0.25
        out.append(f'<text x="{x + 10:.1f}" y="{y + 15}" font-size="14" fill="{"var(--bad)" if low else "var(--fg2)"}">'
                   f'{_n(tot[c])}{"  · thiếu" if low else ""}</text>')
    xt = sx(target)
    out.append(f'<line x1="{xt:.1f}" y1="0" x2="{xt:.1f}" y2="{H - 28}" stroke="var(--line)" stroke-dasharray="2 4"/>'
               f'<text x="{xt + 6:.1f}" y="11" font-size="12" fill="var(--fg)">mục tiêu {_n(target)}</text>')
    out.append("</svg>")
    legend = "".join(f'<span><i style="background:{colors[k]}"></i>{_e(k)}</span>' for k in keys)
    return f'<div class="legend">{legend}</div><div class="chart">{"".join(out)}</div>'


def _shade(v: int, vmax: int) -> str:
    if v <= 0:
        return ""
    a = 0.05 + 0.30 * math.log1p(v) / math.log1p(max(vmax, 1))
    return f' style="background:color-mix(in srgb, var(--accent) {a * 100:.0f}%, transparent)"'


def matrix(srcs: list[dict], class_names: list[str]) -> str:
    vmax = max([s["estimated_classes"].get(c, 0) for s in srcs for c in class_names] or [1])
    head = "".join(f'<th title="{_e(c)}">{i}</th>' for i, c in enumerate(class_names))
    rows = []
    for s in srcs:
        cells = []
        for c in class_names:
            v = s["estimated_classes"].get(c, 0)
            if c in s["missing_classes"] and c not in s["accept_missing"] and v == 0:
                cells.append('<td class="cell-miss" title="nguồn không gán lớp này">✗</td>')
            elif c in s["accept_missing"] and v == 0:
                cells.append('<td class="cell-miss" title="chấp nhận thiếu">–</td>')
            else:
                cells.append(f'<td{_shade(v, vmax)}>{_n(v) if v else "0"}</td>')
        rows.append(f"<tr><td>{_e(s['id'])}</td>{''.join(cells)}</tr>")
    return f"<table><tr><th>Nguồn</th>{head}</tr>{''.join(rows)}</table>"


def source_block(s: dict) -> str:
    if s.get("missing_on_disk"):
        return (f'<div class="src"><div class="src-head"><div><h3>{_e(s["name"])}</h3>'
                f'<div class="meta">{_e(s["id"])} · chưa có trên đĩa (bật trong sources.yaml nhưng chưa tải)</div>'
                f'</div></div></div>')
    tags = "".join(f'<span class="tag">{_e(t)}</span>' for t in
                   [f"tier {s['tier']}", s["domain"], s["format"], s["license"]])
    link = (f'<a href="{_e(s["homepage"].split(" ")[0])}">{_e(s["homepage"].split(" ")[0])}</a>'
            if s.get("homepage") else "—")
    left = [("Ảnh trên đĩa", _n(s["n_images"])), ("File nhãn", _n(s["n_label_files"])),
            ("Dung lượng", _gb(s["disk_bytes"])),
            ("Độ phân giải (trung vị)", f"{s['res_median'][0]} × {s['res_median'][1]}")]
    right = [("Instance / ảnh (trung vị · p90)", f"{s['inst_per_img_median']:.0f} · {s['inst_per_img_p90']:.0f}"),
             ("Ảnh tối (đêm, ước tính)", f"{s['night_share'] * 100:.0f}%"),
             ("Vật < 1.25% cạnh dài (bị lọc)", f"{s['short_side_pct']['below_1_25']:.0f}%"),
             ("Phân tích trên", f"{_n(s['n_analyzed'])} ảnh · {_e(s['analysis_mode'])}")]
    miss = ", ".join(s["missing_classes"]) or "không"
    if s["accept_missing"]:
        miss += f" (chấp nhận thiếu: {', '.join(s['accept_missing'])})"
    origin = ", ".join(f"{k} {_n(v)}" for k, v in sorted(s["label_origin"].items(), key=lambda x: -x[1])) or "—"
    unm = [v for v in s["unmapped"] if v["count"] > 0]
    info = [(f"Lớp chưa có nhãn{' (sau teacher)' if s.get('teacher_completed') else ''}", _e(miss)),
            ("Nguồn gốc nhãn", _e(origin))]
    if unm:
        info.append(("Nhãn gốc không có luật", f"{len(unm)} nhãn → bị bỏ (chủ yếu là nền: đường, cây, vạch sơn…)"))
    vocab = "".join(
        f'<tr><td>{_e(v["raw"])}</td><td>{_e(v["token"])}{"" if v["rule"] else " <span class=flag>· chưa có luật</span>"}'
        f'</td><td>{_n(v["count"])}</td></tr>' for v in s["vocab"][:80])
    thumbs = "".join(f'<img src="{t}" alt="ảnh mẫu {_e(s["id"])}" loading="lazy">' for t in s["thumbs"])
    return f"""<div class="src" id="src-{_e(s['id'])}">
<div class="src-head"><div><h3>{_e(s['name'])}</h3><div class="meta">{_e(s['id'])} · {link}</div></div>
<div class="tags">{tags}</div></div>
<div class="cols">{_rows(left)}{_rows(right)}</div>
{_rows(info)}
<details><summary>Từ vựng nhãn gốc → token · {len(s['vocab'])} nhãn</summary>
<table><tr><th>Nhãn gốc</th><th>Token</th><th>Instance</th></tr>{vocab}</table></details>
<div class="thumbs">{thumbs}</div></div>"""


def alert_rows(srcs: list[dict], class_names: list[str], display: dict, target: int, val_sources: set) -> list:
    items = []
    tot = {c: sum(s["estimated_classes"].get(c, 0) for s in srcs) for c in class_names}
    for i, c in enumerate(class_names):
        if tot[c] < target * 0.25:
            items.append((f"{i} · {_e(display.get(c, c))}", f"~{_n(tot[c])} instance — dưới 25% mục tiêu", "bad"))
        elif tot[c] < target:
            items.append((f"{i} · {_e(display.get(c, c))}", f"~{_n(tot[c])} instance — chưa đạt {_n(target)}", "warn"))
    if not any(s["id"] in val_sources for s in srcs):
        items.append(("Val / test", "Chưa có ảnh VN gán tay (<code>use_for_val</code>) — val sẽ lấy từ dữ liệu nước ngoài", "bad"))
    bg = ("construction--", "nature--", "void--", "flat", "lane", "area", "road", "sidewalk")
    kw = ("vehicle", "car", "bus", "truck", "moto", "bike", "bicycle", "person", "rider", "pedestrian", "sign",
          "light", "trailer", "rickshaw")
    for s in srcs:
        sus = [v for v in s["unmapped"] if any(k in str(v["raw"]).lower() for k in kw)
               and not any(b in str(v["raw"]).lower() for b in bg)]
        if sus:
            items.append((_e(s["id"]), "Nhãn giống đối tượng nhưng chưa có luật: " +
                          ", ".join(_e(v["raw"]) for v in sus[:5]), "bad"))
        if s.get("label_origin", {}).get("teacher"):
            items.append((_e(s["id"]), f"{_n(s['label_origin']['teacher'])} nhãn do teacher sinh — cần người duyệt "
                                       f"ảnh <code>needs_review</code>", "warn"))
        if s["short_side_pct"]["below_1_25"] > 30:
            items.append((_e(s["id"]), f"{s['short_side_pct']['below_1_25']:.0f}% đối tượng nhỏ hơn ngưỡng 1.25% "
                                       f"của BTC — sẽ bị lọc", "warn"))
    return items or [("Không có cảnh báo", "", "ok")]


STAGE_LABELS = [("raw", "Thô (sau chuyển đổi)"), ("cleaned", "Pool sạch"), ("train_pool", "Pool train hợp lệ"),
                ("selected", "Đã chọn (chưa lặp)"), ("train_effective", "Train hiệu dụng (sau RFS)"),
                ("val", "Val"), ("test", "Test")]


def flow_steps(cfg: dict, srcs: list[dict]) -> list:
    """The pipeline as run, with the numbers taken from the config (so the page matches the build)."""
    b, d, sp, ex = cfg.get("balance", {}), cfg.get("dedupe", {}), cfg.get("split", {}), cfg.get("export", {})
    teach = cfg.get("teacher_sources") or []
    tconf = ", ".join(f"{k} {v}" for k, v in (cfg.get("teacher_class_conf") or {}).items())
    full = [s["id"] for s in srcs if not s["missing_classes"]]
    steps = [
        ("Tải nguồn", f"{len(srcs)} bộ công khai (bản sao Hugging Face, Roboflow, Drive của đội) → <code>raw/</code>. "
                      "Chi tiết từng bộ ở mục Nguồn."),
        ("convert", "Đọc mọi định dạng (polygon JSON, COCO, YOLO-seg/detect) và ánh xạ nhãn gốc sang 11 lớp theo "
                    "<code>label_map</code> của từng nguồn. Người chưa rõ vai trò → <code>@person</code>, "
                    "vật BTC không gán → <code>@ignore</code>."),
        ("sam", f"Nguồn chỉ có box: SAM (<code>{_e(cfg.get('sam_model'))}</code>) vẽ mask từ box người gán."),
        ("complete", f"Teacher 11 lớp (<code>{_e(cfg.get('teacher_model'))}</code>) bù lớp nguồn không gán "
                     f"(ngưỡng: {_e(tconf)}; còn lại {cfg.get('teacher_conf')}) và thay có chọn lọc nhãn người / xe "
                     f"hai bánh của nguồn VN (<code>teacher_replace</code>). Áp dụng cho {len(teach)} nguồn: "
                     f"{_e(', '.join(teach))}."),
        ("resolve", "Ghép người lái + xe thành một instance (lớp 5/7), tách chồng lấn theo xa → gần (mask chỉ phần "
                    "nhìn thấy), bỏ vật có cạnh ngắn &lt; 1.25% cạnh dài và diện tích &lt; 64 px² ở 640 "
                    "(ngưỡng bộ chấm BTC bỏ qua)."),
        ("clean", f"Loại ảnh gần trùng (pHash, Hamming ≤ {d.get('hamming_thr')}), tối đa {d.get('max_per_group')} "
                  f"khung / chuỗi video, gắn ngày/đêm. Chia val {sp.get('val_fraction')} / test "
                  f"{sp.get('test_fraction')} theo group, chỉ từ nguồn nhãn người đủ 11 lớp "
                  f"({_e(', '.join(full) or 'IDD, Mapillary, Cityscapes')})."),
        ("balance", f"Chọn ảnh greedy theo thâm hụt (mục tiêu {_n(b.get('target_instances_per_class', 0))} "
                    f"instance / lớp, ngân sách {_n(b.get('max_images', 0))} ảnh), giữ mọi ảnh VN và ≥ "
                    f"{int((b.get('min_domain_share') or {}).get('vn', 0) * 100)}% ảnh VN trong luồng train, rồi "
                    f"repeat-factor sampling (ngưỡng {b.get('rfs_threshold')}, lặp tối đa {b.get('rfs_max_repeat')})."),
        ("export", f"Ảnh cạnh dài {ex.get('max_side')} px, JPEG {ex.get('jpeg_quality')}, nhãn YOLO-seg, metadata "
                   f"nguồn gốc từng ảnh → <code>{_e(ex.get('zip_name', 'du_lieu.zip'))}</code> đúng cấu trúc BTC."),
        ("train", "Notebook BTC (yolo26n-seg, 640, seed 15), danh sách train đã lặp RFS. Chọn checkpoint theo mask "
                  "mAP50 trên val, đối chiếu bảng public."),
    ]
    return steps


def flow_section(cfg: dict, srcs: list[dict], final: dict | None) -> str:
    items = "".join(f'<li><b>{_e(k)}</b><span>{v}</span></li>' for k, v in flow_steps(cfg, srcs))
    table = ""
    if final:
        rows = "".join(f"<tr><td>{lbl}</td><td>{_n(final[k]['images'])}</td><td>{_n(final[k]['instances'])}</td></tr>"
                       for k, lbl in STAGE_LABELS if k in final)
        table = (f'<h3 style="margin-top:40px">Số liệu bộ dữ liệu cuối</h3><table><tr><th>Giai đoạn</th><th>Ảnh</th>'
                 f'<th>Instance</th></tr>{rows}</table>')
    return f'<ol class="flow">{items}</ol>{table}'


def render(result: dict, classes, cfg: dict) -> str:
    srcs = [s for s in result["sources"] if not s.get("missing_on_disk")]
    missing = [s for s in result["sources"] if s.get("missing_on_disk")]
    target = int(cfg["balance"]["target_instances_per_class"])
    val_sources = set(result.get("val_sources", []))
    n_img = sum(s["n_images"] for s in srcs)
    n_inst = sum(sum(s["estimated_classes"].values()) for s in srcs)
    disk = sum(s["disk_bytes"] for s in srcs)
    vn = sum(s["n_images"] for s in srcs if s["domain"] == "vn")
    below = sum(1 for c in classes.names if sum(s["estimated_classes"].get(c, 0) for s in srcs) < target)
    kpis = [(_n(n_img), "ảnh"), (_n(n_inst), "instance ước tính"), (_gb(disk), "dung lượng thô"),
            (str(len(srcs)), "nguồn đã tải"), (f"{vn / max(n_img, 1) * 100:.0f}%", "ảnh miền VN"),
            (f"{below}/11", "lớp dưới mục tiêu")]
    kp = "".join(f'<div class="kpi"><b>{a}</b><span>{b}</span></div>' for a, b in kpis)
    now = datetime.now().strftime("%H:%M · %d/%m/%Y")
    blocks = "".join(source_block(s) for s in srcs + missing)
    legend_cls = " · ".join(f"{i} {_e(classes.display.get(c, c))}" for i, c in enumerate(classes.names))
    demo = ('<div class="demo"><b>DỮ LIỆU DEMO TỔNG HỢP</b> — ảnh hình khối do scripts/make_demo_data.py vẽ, '
            'chỉ để thử pipeline. Báo cáo dữ liệu thật: reports/raw_overview.html</div>') if result.get("synthetic") else ""
    alerts = alert_rows(srcs, classes.names, classes.display, target, val_sources)
    return f"""<!doctype html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Tổng quan dữ liệu thô</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@300;400;500;600&display=swap" rel="stylesheet">
<script>(function(){{var t;try{{t=localStorage.getItem("raw-report-theme")}}catch(e){{}}
if(!t)t=matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light";
document.documentElement.setAttribute("data-theme",t)}})();</script>
<style>{CSS}</style></head><body>
<header class="bar"><div class="bar-in">
<div class="brand">TEAM 10 &amp; SEED 15</div>
<nav class="nav"><a href="#tong-quan">Tổng quan</a><a href="#luong">Luồng</a><a href="#do-phu">Độ phủ</a><a href="#ma-tran">Ma trận</a>
<a href="#canh-bao">Cảnh báo</a><a href="#nguon">Nguồn</a></nav>
<button class="switch" id="theme" type="button" role="switch" aria-label="Chế độ tối">
<span class="track"><span class="knob"></span></span><span class="lbl-light">Sáng</span><span class="lbl-dark">Tối</span></button>
<a class="btn" href="raw_overview.json">Dữ liệu JSON</a>
</div></header>
<main>
{demo}
<section id="tong-quan" style="padding-top:0">
<h1>Tổng quan dữ liệu thô{' (demo)' if result.get("synthetic") else ''}</h1>
<p class="lead">Những gì đang có trong <code>raw/</code> trước khi chạy các bước chung — ước tính theo đúng luật của pipeline.</p>
<p class="meta" style="margin-top:14px">Sinh lúc {now} bởi <code>scripts/analyze_raw.py</code></p>
<div class="kpis">{kp}</div>
</section>

<section id="luong">
<h2>Luồng xử lý</h2>
<p class="lead">Các bước <code>scripts/run_pipeline.py</code> chạy, theo đúng thứ tự, với tham số lấy từ cấu hình.</p>
{flow_section(cfg, srcs, result.get("final_stats"))}
</section>

<section id="do-phu">
<h2>Độ phủ 11 lớp</h2>
<p class="lead">Instance ước tính sau ghép người lái + xe, tách chồng lấn và bỏ vật &lt; 1.25%, xếp chồng theo nguồn.</p>
{coverage_chart(srcs, classes.names, classes.display, target)}
</section>

<section id="ma-tran">
<h2>Ma trận nguồn × lớp</h2>
<p class="lead">{legend_cls}</p>
{matrix(srcs, classes.names)}
<p class="meta" style="margin-top:14px">✗ nguồn không gán lớp này · – chấp nhận thiếu · đậm hơn = nhiều hơn (thang log)</p>
</section>

<section id="canh-bao">
<h2>Cảnh báo</h2>
<p class="lead">Lỗ hổng cần xử lý trước khi chạy các bước chung.</p>
{_rows(alerts)}
</section>

<section id="nguon">
<h2>Chi tiết từng nguồn</h2>
<p class="lead">Mỗi nguồn: quy mô, chất lượng ảnh, độ phủ nhãn, từ vựng nhãn gốc và ảnh mẫu có vẽ mask.</p>
{blocks}
</section>
<footer>Số liệu là ước tính. Số cuối cùng nằm trong <code>stats/REPORT.md</code> sau khi chạy pipeline.</footer>
</main>
<script>(function(){{var r=document.documentElement,b=document.getElementById("theme");
function sync(){{b.setAttribute("aria-checked",r.getAttribute("data-theme")==="dark")}}sync();
b.addEventListener("click",function(){{var t=r.getAttribute("data-theme")==="dark"?"light":"dark";
r.setAttribute("data-theme",t);try{{localStorage.setItem("raw-report-theme",t)}}catch(e){{}}sync()}});}})();</script>
</body></html>"""
