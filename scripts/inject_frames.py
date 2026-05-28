"""
Build synthetic injected videos from clean base videos + ad frames.

For each clean video, randomly select injection points and insert
a burst of ad frames (9–30 frames, ~300ms–1s at 30fps).

Outputs:
  - data/synthetic/videos/<name>_injected.mp4
  - data/synthetic/annotations/<name>_annotations.json

Run:
  python scripts/inject_frames.py \
    --videos data/raw_videos \
    --ad_frames data/ad_frames \
    --output data/synthetic \
    --injections_per_video 3
"""

import argparse
import json
import os
import random
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def get_video_info(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"fps": fps, "total_frames": total, "width": w, "height": h}


def load_ad_frames(ad_frames_dir: str, width: int, height: int) -> list[np.ndarray]:
    """Load and resize all ad frames to match video resolution."""
    frames = []
    for fname in sorted(os.listdir(ad_frames_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(ad_frames_dir, fname)
        img = cv2.imread(path)
        if img is not None:
            if img.shape[1] != width or img.shape[0] != height:
                img = cv2.resize(img, (width, height))
            frames.append(img)
    return frames


def inject_into_video(
    video_path: str,
    ad_frames: list[np.ndarray],
    output_path: str,
    injections: list[dict],
    fps: float,
    width: int,
    height: int,
):
    """
    Re-encode video with injected ad frame bursts.

    injections: list of {insert_after_frame: int, ad_indices: [int, ...]}
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Build an injection map: frame_idx → list of ad_frame arrays to insert before
    inject_map: dict[int, list[np.ndarray]] = {}
    for inj in injections:
        pos = inj["insert_after_frame"]
        burst = [ad_frames[i] for i in inj["ad_indices"]]
        inject_map[pos] = burst

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in inject_map:
            for ad_frame in inject_map[frame_idx]:
                out.write(ad_frame)

        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()


def plan_injections(
    total_frames: int,
    fps: float,
    ad_frames: list[np.ndarray],
    n_injections: int,
    min_burst_frames: int = 9,
    max_burst_frames: int = 30,
    min_gap_frames: int = 90,  # at least 3s between injections
) -> list[dict]:
    """
    Randomly plan injection positions.

    Returns list of injection specs with ground truth metadata.
    """
    # Keep injections away from very start/end
    safe_start = int(fps * 5)
    safe_end = total_frames - int(fps * 5)

    injections = []
    used_positions = []

    attempts = 0
    while len(injections) < n_injections and attempts < 200:
        attempts += 1
        pos = random.randint(safe_start, safe_end)

        # Check minimum gap from other injections
        if any(abs(pos - p) < min_gap_frames for p in used_positions):
            continue

        burst_len = random.randint(min_burst_frames, max_burst_frames)
        # Pick a contiguous range from the same ad frame type (realistic burst)
        ad_start = random.randint(0, max(0, len(ad_frames) - burst_len))
        ad_indices = list(range(ad_start, min(ad_start + burst_len, len(ad_frames))))
        if len(ad_indices) < min_burst_frames:
            continue

        injections.append({
            "insert_after_frame": pos,
            "num_injected_frames": len(ad_indices),
            "ad_indices": ad_indices,
            "duration_ms": len(ad_indices) / fps * 1000,
            "start_ms_in_output": (pos + len(ad_indices) * 0) / fps * 1000,  # approx
        })
        used_positions.append(pos)

    return injections


def process_video(
    video_path: str,
    ad_frames_dir: str,
    output_dir: str,
    n_injections: int = 3,
    min_burst: int = 9,
    max_burst: int = 30,
    also_create_clean_copy: bool = True,
):
    """
    Process one video: create injected version + annotation file.
    Optionally also create a clean (no-injection) copy with null annotations.
    """
    info = get_video_info(video_path)
    fps, total, w, h = info["fps"], info["total_frames"], info["width"], info["height"]

    if total < int(fps * 20):
        print(f"  Skipping {Path(video_path).name}: too short ({total / fps:.1f}s)")
        return

    ad_frames = load_ad_frames(ad_frames_dir, w, h)
    if not ad_frames:
        raise ValueError(f"No ad frames found in {ad_frames_dir}")

    videos_out = os.path.join(output_dir, "videos")
    annots_out = os.path.join(output_dir, "annotations")
    os.makedirs(videos_out, exist_ok=True)
    os.makedirs(annots_out, exist_ok=True)

    stem = Path(video_path).stem

    # ── Injected version ──
    injections = plan_injections(total, fps, ad_frames, n_injections, min_burst, max_burst)
    inj_path = os.path.join(videos_out, f"{stem}_injected.mp4")
    print(f"  Injecting {len(injections)} burst(s) into {stem}...")
    inject_into_video(video_path, ad_frames, inj_path, injections, fps, w, h)

    # Compute actual output frame positions (account for inserted frames)
    cumulative_offset = 0
    for inj in injections:
        inj["output_start_frame"] = inj["insert_after_frame"] + cumulative_offset
        inj["output_end_frame"] = inj["output_start_frame"] + inj["num_injected_frames"] - 1
        inj["output_start_ms"] = inj["output_start_frame"] / fps * 1000
        inj["output_end_ms"] = inj["output_end_frame"] / fps * 1000
        cumulative_offset += inj["num_injected_frames"]

    annotation = {
        "video_path": inj_path,
        "source_video": video_path,
        "fps": fps,
        "label": "injected",
        "injections": [
            {
                "output_start_frame": inj["output_start_frame"],
                "output_end_frame": inj["output_end_frame"],
                "output_start_ms": inj["output_start_ms"],
                "output_end_ms": inj["output_end_ms"],
                "duration_ms": inj["duration_ms"],
                "num_frames": inj["num_injected_frames"],
            }
            for inj in injections
        ],
    }
    annot_path = os.path.join(annots_out, f"{stem}_injected.json")
    with open(annot_path, "w") as f:
        json.dump(annotation, f, indent=2)

    # ── Clean version (negative samples) ──
    if also_create_clean_copy:
        clean_path = os.path.join(videos_out, f"{stem}_clean.mp4")
        shutil.copy2(video_path, clean_path)
        clean_annot = {
            "video_path": clean_path,
            "source_video": video_path,
            "fps": fps,
            "label": "clean",
            "injections": [],
        }
        clean_annot_path = os.path.join(annots_out, f"{stem}_clean.json")
        with open(clean_annot_path, "w") as f:
            json.dump(clean_annot, f, indent=2)

    print(f"  Done: {inj_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", default="data/raw_videos")
    parser.add_argument("--ad_frames", default="data/ad_frames")
    parser.add_argument("--output", default="data/synthetic")
    parser.add_argument("--injections_per_video", type=int, default=3)
    parser.add_argument("--min_burst", type=int, default=9)
    parser.add_argument("--max_burst", type=int, default=30)
    args = parser.parse_args()

    video_files = [
        os.path.join(args.videos, f)
        for f in os.listdir(args.videos)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    ]
    if not video_files:
        print(f"No videos found in {args.videos}")
        return

    print(f"Processing {len(video_files)} video(s)...")
    for vf in tqdm(video_files, desc="Videos"):
        try:
            process_video(
                vf, args.ad_frames, args.output,
                n_injections=args.injections_per_video,
                min_burst=args.min_burst,
                max_burst=args.max_burst,
            )
        except Exception as e:
            print(f"  Error on {vf}: {e}")


if __name__ == "__main__":
    main()
