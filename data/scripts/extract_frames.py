"""Cut dashcam videos into frames for the `vn_dashcam_self` source.

  python scripts/extract_frames.py --videos D:/dashcam --out raw/vn_dashcam_self/images --fps 1

One sub-folder per video => the folder name becomes the leakage `group` (all frames of one
video always land in the same split). Near-identical frames are dropped later by the
`clean` stage (pHash) and `dedupe.max_per_group`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".ts"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=1.0, help="frames kept per second of video")
    ap.add_argument("--max-side", type=int, default=1920)
    a = ap.parse_args()
    out = Path(a.out)
    for v in sorted(p for p in Path(a.videos).rglob("*") if p.suffix.lower() in VIDEO_EXT):
        cap = cv2.VideoCapture(str(v))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        step = max(1, round(src_fps / a.fps))
        d = out / v.stem
        d.mkdir(parents=True, exist_ok=True)
        i = kept = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if i % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    h, w = frame.shape[:2]
                    s = min(1.0, a.max_side / max(h, w))
                    if s < 1:
                        frame = cv2.resize(frame, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(str(d / f"{v.stem}_{i:07d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    kept += 1
            i += 1
        cap.release()
        print(f"{v.name}: {kept} frames")


if __name__ == "__main__":
    main()
