"""
Full detection pipeline orchestrator.

Ties together all steps:
  1. Input video
  2. Extract frames at full FPS
  3. Compute visual distances between neighboring frames
  4. Detect A–X–A temporal anomaly candidates
  5. Merge nearby suspicious frames into segments
  6. Generate contact sheets with before/center/after context
  7. Feed contact sheets into Qwen2.5-VL
  8. Qwen-VL predicts label, confidence, elements, reason
  9. Combine temporal score + VLM score
 10. Output timestamps, frame ranges, confidence scores, labels, and visual report
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import yaml
from rich.console import Console
from rich.table import Table

from src.anomaly_detector import run_anomaly_detection
from src.contact_sheet import build_all_contact_sheets
from src.frame_extractor import extract_frames
from src.qwen_classifier import QwenVLClassifier
from src.report_generator import generate_report
from src.score_combiner import combine_all
from src.subtitle_extractor import load_subtitles

console = Console()


def load_config(config_path: str = "configs/pipeline_config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_pipeline(
    video_path: str,
    config: Optional[dict] = None,
    config_path: str = "configs/pipeline_config.yaml",
    work_dir: Optional[str] = None,
    srt_path: Optional[str] = None,
    keep_frames: bool = False,
) -> dict:
    """
    Run the full detection pipeline on a single video.

    Args:
        video_path:  Path to input video file.
        config:      Config dict (overrides config_path if provided).
        config_path: Path to YAML config file.
        work_dir:    Directory for intermediate files. Auto-generated if None.
        srt_path:    Optional explicit SRT subtitle path.
        keep_frames: Keep extracted frames on disk after pipeline finishes.

    Returns:
        dict with keys: report_paths, detections, video_meta
    """
    cfg = config or load_config(config_path)

    video_path = str(Path(video_path).resolve())
    video_name = Path(video_path).stem

    # Work directory for intermediate files
    auto_work_dir = work_dir is None
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix=f"vlm_{video_name}_")
    frames_dir = os.path.join(work_dir, "frames")
    sheets_dir = os.path.join(work_dir, "sheets")
    report_dir = os.path.join(cfg["report"]["output_dir"], video_name)

    console.rule(f"[bold blue]Processing: {video_name}")

    # ── Step 2: Extract frames ─────────────────────────────────────────────
    console.print("[1/8] Extracting frames...")
    frame_store = extract_frames(
        video_path=video_path,
        output_dir=frames_dir,
        target_fps=None,  # native FPS
        save_frames=True,
        frame_quality=cfg["video"]["frame_quality"],
    )
    fps = frame_store.video_meta.fps
    console.print(f"      {len(frame_store.frames)} frames @ {fps:.1f} fps  "
                  f"({frame_store.video_meta.duration_ms / 1000:.1f}s)")

    if fps < cfg["video"]["min_fps"]:
        console.print(f"[yellow]Warning: video FPS ({fps:.1f}) below minimum "
                      f"({cfg['video']['min_fps']}). Short insertions may be missed.")

    # ── Steps 3–5: Anomaly detection ──────────────────────────────────────
    console.print("[2/8] Running temporal anomaly detection...")
    anom_cfg = cfg["anomaly_detection"]
    anomaly_result = run_anomaly_detection(
        frame_store,
        phash_threshold=anom_cfg["phash_threshold"],
        scene_cut_threshold=anom_cfg["scene_cut_threshold"],
        context_window=anom_cfg["context_window"],
        min_segment_frames=anom_cfg["min_segment_frames"],
        max_segment_frames=anom_cfg["max_segment_frames"],
        merge_gap_frames=anom_cfg["merge_gap_frames"],
        hash_size=anom_cfg["phash_hash_size"],
    )
    console.print(f"      {len(anomaly_result.segments)} candidate segment(s)  |  "
                  f"{len(anomaly_result.scene_cut_frames)} scene cut(s)")

    if not anomaly_result.segments:
        console.print("[green]No suspicious segments found. Generating clean report.")
        report_paths = generate_report(
            video_meta=frame_store.video_meta,
            anomaly_result=anomaly_result,
            results=[],
            output_dir=report_dir,
            model_path=cfg["classifier"]["model_path"],
            formats=tuple(cfg["report"]["format"].split(",")) if isinstance(cfg["report"]["format"], str)
                    else (cfg["report"]["format"],),
        )
        _cleanup(work_dir, frames_dir, keep_frames, auto_work_dir)
        return {"report_paths": report_paths, "detections": [], "video_meta": frame_store.video_meta}

    # ── Load subtitles ─────────────────────────────────────────────────────
    console.print("[3/8] Loading subtitles...")
    subtitle_track = load_subtitles(video_path, srt_path=srt_path)
    sub_info = f"{len(subtitle_track.entries)} entries" if not subtitle_track.is_empty() else "none found"
    console.print(f"      Subtitles: {sub_info}")

    # ── Step 6: Contact sheets ─────────────────────────────────────────────
    console.print("[4/8] Building contact sheets...")
    cs_cfg = cfg["contact_sheet"]
    sheet_items = build_all_contact_sheets(
        anomaly_result=anomaly_result,
        frame_store=frame_store,
        subtitle_track=subtitle_track,
        output_dir=sheets_dir,
        context_frames=cs_cfg["context_frames"],
        max_anomaly_frames=cs_cfg["max_anomaly_frames"],
        tile_w=cs_cfg["tile_width"],
        tile_h=cs_cfg["tile_height"],
        border_px=cs_cfg["border_px"],
        font_size=cs_cfg["font_size"],
        subtitle_strip_h=cs_cfg["subtitle_strip_height"],
    )
    console.print(f"      {len(sheet_items)} contact sheet(s) saved.")

    # ── Steps 7–8: Qwen-VL classification ─────────────────────────────────
    console.print("[5/8] Running Qwen2.5-VL classifier...")
    clf_cfg = cfg["classifier"]
    model_path = (
        clf_cfg["finetuned_path"] if clf_cfg.get("use_finetuned") else clf_cfg["model_path"]
    )
    classifier = QwenVLClassifier(
        model_path=model_path,
        use_4bit=clf_cfg["use_4bit"],
        max_new_tokens=clf_cfg["max_new_tokens"],
        temperature=clf_cfg["temperature"],
    )
    classify_inputs = [
        {
            "sheet_path": item["sheet_path"],
            "subtitle_before": item["subtitle_before"],
            "subtitle_during": item["subtitle_during"],
            "subtitle_after": item["subtitle_after"],
            "temporal_score": item["segment"].temporal_score,
            "duration_ms": item["segment"].duration_ms,
            "num_frames": item["segment"].num_frames,
        }
        for item in sheet_items
    ]
    clf_outputs = classifier.classify_batch(classify_inputs)
    console.print(f"      Classification complete.")

    # ── Step 9: Score fusion ───────────────────────────────────────────────
    console.print("[6/8] Fusing scores...")
    sc_cfg = cfg["scoring"]
    detection_results = combine_all(
        sheet_items=sheet_items,
        classifier_outputs=clf_outputs,
        temporal_weight=sc_cfg["temporal_weight"],
        vlm_weight=sc_cfg["vlm_weight"],
        decision_threshold=sc_cfg["decision_threshold"],
    )

    # ── Step 10: Report ────────────────────────────────────────────────────
    console.print("[7/8] Generating report...")
    fmt = cfg["report"]["format"]
    formats = tuple(fmt.split(",")) if isinstance(fmt, str) else (fmt,)
    report_paths = generate_report(
        video_meta=frame_store.video_meta,
        anomaly_result=anomaly_result,
        results=detection_results,
        output_dir=report_dir,
        model_path=model_path,
        formats=formats,
    )

    # ── Summary table ──────────────────────────────────────────────────────
    flagged = [r for r in detection_results if r.is_flagged]
    console.print(f"\n[8/8] Done. {len(flagged)}/{len(detection_results)} segment(s) flagged.\n")
    _print_summary_table(detection_results)

    for fmt_key, path in report_paths.items():
        console.print(f"[bold]Report ({fmt_key}):[/bold] {path}")

    _cleanup(work_dir, frames_dir, keep_frames, auto_work_dir)

    return {
        "report_paths": report_paths,
        "detections": detection_results,
        "video_meta": frame_store.video_meta,
    }


def _print_summary_table(results):
    table = Table(title="Detection Summary", show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=4)
    table.add_column("Start", width=10)
    table.add_column("End", width=10)
    table.add_column("Duration", width=10)
    table.add_column("Temporal", width=10)
    table.add_column("VLM", width=8)
    table.add_column("Final", width=8)
    table.add_column("Status", width=20)
    table.add_column("Elements")

    for i, r in enumerate(results):
        status = "[red]AD DETECTED[/red]" if r.is_flagged else "[green]Normal[/green]"
        table.add_row(
            str(i + 1),
            f"{r.start_ms / 1000:.2f}s",
            f"{r.end_ms / 1000:.2f}s",
            f"{r.duration_ms:.0f}ms",
            f"{r.temporal_score:.3f}",
            f"{r.vlm_confidence:.3f}",
            f"{r.final_score:.3f}",
            status,
            ", ".join(r.detected_elements) if r.detected_elements else "—",
        )
    console.print(table)


def _cleanup(work_dir: str, frames_dir: str, keep_frames: bool, auto_work_dir: bool):
    if not keep_frames and os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    if auto_work_dir and os.path.exists(work_dir) and not os.listdir(work_dir):
        shutil.rmtree(work_dir)
