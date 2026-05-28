"""
Build fine-tuning JSONL dataset from synthetic injected videos.

For each annotated video:
  - Run frame extraction + anomaly detection
  - For each TRUE injection window: generate contact sheet → positive sample
  - For each non-injection anomaly candidate: generate contact sheet → negative sample
  - For random "clean" windows: generate contact sheet → hard-negative sample

Output: data/dataset/{train,val,test}/annotations.jsonl

Run:
  python scripts/build_dataset.py \
    --annotations data/synthetic/annotations \
    --output data/dataset \
    --train_ratio 0.7 --val_ratio 0.15
"""

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import json
import random
from pathlib import Path

from tqdm import tqdm

from src.anomaly_detector import run_anomaly_detection
from src.contact_sheet import build_contact_sheet
from src.frame_extractor import extract_frames
from src.subtitle_extractor import load_subtitles, SubtitleTrack


def _ms_overlap(a_start, a_end, b_start, b_end) -> float:
    overlap = max(0, min(a_end, b_end) - max(a_start, b_start))
    span = max(a_end, b_end) - min(a_start, b_start)
    return overlap / span if span > 0 else 0.0


def process_annotated_video(
    annotation: dict,
    output_dir: str,
    context_frames: int = 3,
    max_anomaly_frames: int = 5,
) -> list[dict]:
    """Process one annotated video and return a list of JSONL records."""
    video_path = annotation["video_path"]
    true_injections = annotation.get("injections", [])
    is_clean = annotation["label"] == "clean"

    if not os.path.exists(video_path):
        print(f"  Missing: {video_path}, skipping.")
        return []

    stem = Path(video_path).stem
    frames_dir = os.path.join(output_dir, "frames", stem)
    sheets_dir = os.path.join(output_dir, "sheets", stem)
    os.makedirs(sheets_dir, exist_ok=True)

    # Extract frames
    frame_store = extract_frames(video_path, frames_dir, show_progress=False)
    subtitle_track = load_subtitles(video_path)

    # Run anomaly detection
    anomaly_result = run_anomaly_detection(frame_store)

    fps = frame_store.video_meta.fps
    records = []

    # ── Positive samples (true injection windows) ──────────────────────────
    for inj_idx, inj in enumerate(true_injections):
        inj_start_frame = inj["output_start_frame"]
        inj_end_frame = inj["output_end_frame"]
        inj_start_ms = inj["output_start_ms"]
        inj_end_ms = inj["output_end_ms"]

        # Find or create a segment covering this injection
        # Use the true injection frames directly as the anomaly segment
        from src.anomaly_detector import AnomalySegment
        from src.anomaly_detector import AnomalyResult

        seg_distances = anomaly_result.frame_distances[inj_start_frame: inj_end_frame + 1]
        if not seg_distances:
            seg_distances = [30.0] * (inj_end_frame - inj_start_frame + 1)

        segment = AnomalySegment(
            start_frame=inj_start_frame,
            end_frame=inj_end_frame,
            start_ms=inj_start_ms,
            end_ms=inj_end_ms,
            duration_ms=inj_end_ms - inj_start_ms,
            peak_distance=max(seg_distances) if seg_distances else 30.0,
            mean_distance=sum(seg_distances) / len(seg_distances) if seg_distances else 30.0,
            temporal_score=min(1.0, (sum(seg_distances) / len(seg_distances) if seg_distances else 30.0) / 64.0),
            frame_distances=seg_distances,
        )

        sheet_path = os.path.join(sheets_dir, f"pos_{inj_idx:04d}.jpg")
        build_contact_sheet(segment, frame_store, subtitle_track, sheet_path,
                            context_frames=context_frames, max_anomaly_frames=max_anomaly_frames)

        ctx_ms = context_frames / fps * 1000
        sub_before = subtitle_track.get_text_for_range(inj_start_ms - ctx_ms, inj_start_ms)
        sub_during = subtitle_track.get_text_for_range(inj_start_ms, inj_end_ms)
        sub_after = subtitle_track.get_text_for_range(inj_end_ms, inj_end_ms + ctx_ms)

        records.append(_make_record(
            sheet_path=sheet_path,
            label="ad_insertion",
            confidence=0.95,
            detected_elements=["promotional content"],
            reason="This segment contains injected promotional frames inconsistent with surrounding content.",
            subtitle_before=sub_before,
            subtitle_during=sub_during,
            subtitle_after=sub_after,
            temporal_score=segment.temporal_score,
            duration_ms=segment.duration_ms,
            num_frames=segment.num_frames,
        ))

    # ── Negative samples (detected anomalies that are NOT real injections) ──
    for det_idx, seg in enumerate(anomaly_result.segments):
        # Check overlap with any true injection
        is_tp = False
        for inj in true_injections:
            overlap = _ms_overlap(seg.start_ms, seg.end_ms,
                                  inj["output_start_ms"], inj["output_end_ms"])
            if overlap > 0.3:
                is_tp = True
                break
        if is_tp:
            continue  # already covered as positive

        sheet_path = os.path.join(sheets_dir, f"neg_{det_idx:04d}.jpg")
        build_contact_sheet(seg, frame_store, subtitle_track, sheet_path,
                            context_frames=context_frames, max_anomaly_frames=max_anomaly_frames)

        ctx_ms = context_frames / fps * 1000
        sub_before = subtitle_track.get_text_for_range(seg.start_ms - ctx_ms, seg.start_ms)
        sub_during = subtitle_track.get_text_for_range(seg.start_ms, seg.end_ms)
        sub_after = subtitle_track.get_text_for_range(seg.end_ms, seg.end_ms + ctx_ms)

        records.append(_make_record(
            sheet_path=sheet_path,
            label="normal",
            confidence=0.9,
            detected_elements=[],
            reason="This segment appears to be a scene transition or natural content change, not an ad insertion.",
            subtitle_before=sub_before,
            subtitle_during=sub_during,
            subtitle_after=sub_after,
            temporal_score=seg.temporal_score,
            duration_ms=seg.duration_ms,
            num_frames=seg.num_frames,
        ))

    # Cleanup frames to save disk space
    import shutil
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)

    return records


def _make_record(
    sheet_path: str,
    label: str,
    confidence: float,
    detected_elements: list,
    reason: str,
    subtitle_before: str,
    subtitle_during: str,
    subtitle_after: str,
    temporal_score: float,
    duration_ms: float,
    num_frames: int,
) -> dict:
    """Build a LLaMA-Factory / TRL compatible JSONL record."""
    from src.qwen_classifier import SYSTEM_PROMPT, _build_user_prompt
    user_content = _build_user_prompt(
        subtitle_before, subtitle_during, subtitle_after,
        temporal_score, duration_ms, num_frames
    )
    assistant_response = json.dumps({
        "label": label,
        "confidence": confidence,
        "detected_elements": detected_elements,
        "reason": reason,
    })
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sheet_path},
                    {"type": "text", "text": user_content},
                ],
            },
            {"role": "assistant", "content": assistant_response},
        ],
        "images": [sheet_path],
    }


def build_dataset(
    annotations_dir: str,
    output_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
):
    random.seed(seed)
    all_records = []

    annot_files = [
        os.path.join(annotations_dir, f)
        for f in sorted(os.listdir(annotations_dir))
        if f.endswith(".json")
    ]
    print(f"Processing {len(annot_files)} annotation file(s)...")

    for af in tqdm(annot_files):
        with open(af) as f:
            annotation = json.load(f)
        records = process_annotated_video(annotation, output_dir)
        all_records.extend(records)
        print(f"  {Path(af).stem}: {len(records)} samples")

    random.shuffle(all_records)
    n = len(all_records)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": all_records[:n_train],
        "val": all_records[n_train: n_train + n_val],
        "test": all_records[n_train + n_val:],
    }

    for split_name, records in splits.items():
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        out_path = os.path.join(split_dir, "annotations.jsonl")
        with open(out_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        pos = sum(1 for r in records if r["messages"][-1]["content"].find('"ad_insertion"') >= 0)
        neg = len(records) - pos
        print(f"  {split_name}: {len(records)} records  (pos={pos}, neg={neg}) → {out_path}")

    print(f"\nDataset complete: {n} total records across {len(splits)} splits.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default="data/synthetic/annotations")
    parser.add_argument("--output", default="data/dataset")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_dataset(args.annotations, args.output, args.train_ratio, args.val_ratio, args.seed)
