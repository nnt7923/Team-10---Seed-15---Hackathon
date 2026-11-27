"""Fast parallel download of many SMALL files from a HF dataset repo (per-file latency bound).

  python scripts/download_hf_files.py ali-sh07/Mapillary-dataset training/v2.0/panoptic/ raw/mapillary --reverse

Files already present locally are skipped, so it can run next to `download_hf.py` on the same folder
(e.g. one process from the start of the list, this one from the end).
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("prefix")
    ap.add_argument("local_dir")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--reverse", action="store_true")
    a = ap.parse_args()
    from huggingface_hub import HfApi, hf_hub_download
    files = [f for f in HfApi().list_repo_files(a.repo, repo_type="dataset") if f.startswith(a.prefix)]
    if a.reverse:
        files = files[::-1]
    todo = [f for f in files if not (Path(a.local_dir) / f).exists()]
    print(f"{len(files)} files, {len(todo)} missing", flush=True)
    done = 0

    def get(f):
        if (Path(a.local_dir) / f).exists():
            return
        hf_hub_download(a.repo, f, repo_type="dataset", local_dir=a.local_dir)

    with ThreadPoolExecutor(a.workers) as ex:
        for _ in ex.map(get, todo):
            done += 1
            if done % 500 == 0:
                print(f"{done}/{len(todo)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
