"""Analyse the raw downloaded sources and write a minimalist HTML overview.

  python scripts/analyze_raw.py                              # all enabled sources -> reports/raw_overview.html
  python scripts/analyze_raw.py --sample 500 --out reports/raw.html
  python scripts/analyze_raw.py --config configs/pipeline_demo.yaml

Read-only: it never modifies raw/ or work/. Sources already converted (work/converted/<id>.jsonl, e.g.
after SAM + teacher) are analysed in full; others are estimated from a random sample.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from vtb.common import load_pipeline  # noqa: E402
from vtb.html_report import render  # noqa: E402
from vtb.rawstats import analyze  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--sample", type=int, default=300, help="images per source when not converted yet")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg, classes, sources, base = load_pipeline(a.config)
    out = Path(a.out) if a.out else base / "reports" / "raw_overview.html"
    res = analyze(sources, Path(cfg["data_root"]), classes, Path(cfg["work_dir"]) / "converted", a.sample)
    res["val_sources"] = [s["id"] for s in sources if s.get("use_for_val")]
    res["synthetic"] = "demo" in Path(cfg["sources"]).name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(res, classes, cfg), encoding="utf-8")
    slim = {"sources": [{k: v for k, v in s.items() if k != "thumbs"} for s in res["sources"]]}
    out.with_suffix(".json").write_text(json.dumps(slim, ensure_ascii=False, indent=1), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
