"""
Step 10: Generate detection report.

Outputs:
  - <video_name>_report.json   — machine-readable full results
  - <video_name>_report.html   — human-readable visual report with embedded contact sheets
"""

import base64
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.anomaly_detector import AnomalyResult
from src.frame_extractor import VideoMeta
from src.score_combiner import DetectionResult


def _img_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _build_timeline_plot(
    video_meta: VideoMeta,
    anomaly_result: AnomalyResult,
    results: list[DetectionResult],
    output_path: str,
):
    """Build a timeline showing frame distances and flagged segments."""
    distances = anomaly_result.frame_distances
    fps = video_meta.fps
    times = [i / fps for i in range(len(distances))]

    fig, ax = plt.subplots(figsize=(14, 3), facecolor="#1e1e1e")
    ax.set_facecolor("#1e1e1e")
    ax.plot(times, distances, color="#5599ff", linewidth=0.8, alpha=0.8, label="pHash distance")

    # Scene cuts
    for fc in anomaly_result.scene_cut_frames:
        ax.axvline(fc / fps, color="#ffaa00", linewidth=0.5, alpha=0.5)

    # Highlight flagged segments
    for r in results:
        color = "#ff4444" if r.is_flagged else "#44aa44"
        ax.axvspan(r.start_ms / 1000, r.end_ms / 1000, alpha=0.3, color=color)

    ax.set_xlabel("Time (s)", color="#aaaaaa")
    ax.set_ylabel("pHash distance", color="#aaaaaa")
    ax.tick_params(colors="#aaaaaa")
    ax.spines["bottom"].set_color("#555555")
    ax.spines["left"].set_color("#555555")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    red_patch = mpatches.Patch(color="#ff4444", alpha=0.5, label="Flagged")
    green_patch = mpatches.Patch(color="#44aa44", alpha=0.5, label="Cleared")
    ax.legend(handles=[red_patch, green_patch], facecolor="#333333", labelcolor="#cccccc")
    ax.set_title("Temporal Anomaly Timeline", color="#dddddd", pad=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight", facecolor="#1e1e1e")
    plt.close()


HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Ad Insertion Detection Report — {video_name}</title>
<style>
  body {{ background: #1a1a2e; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; margin: 0; padding: 20px; }}
  h1 {{ color: #a78bfa; border-bottom: 1px solid #333; padding-bottom: 8px; }}
  h2 {{ color: #60a5fa; margin-top: 30px; }}
  .meta {{ background: #16213e; border-radius: 8px; padding: 14px; margin-bottom: 20px; }}
  .meta table {{ border-collapse: collapse; width: 100%; }}
  .meta td {{ padding: 4px 12px; }}
  .meta td:first-child {{ color: #94a3b8; width: 180px; }}
  .summary {{ display: flex; gap: 16px; margin-bottom: 24px; }}
  .stat {{ background: #16213e; border-radius: 8px; padding: 14px 20px; flex: 1; text-align: center; }}
  .stat .val {{ font-size: 2em; font-weight: bold; }}
  .flagged {{ color: #f87171; }}
  .normal  {{ color: #4ade80; }}
  .uncertain {{ color: #fbbf24; }}
  .timeline {{ margin: 20px 0; }}
  .timeline img {{ width: 100%; border-radius: 8px; }}
  .segment {{ background: #16213e; border-radius: 10px; padding: 16px; margin-bottom: 24px; border-left: 4px solid #888; }}
  .segment.flagged {{ border-left-color: #f87171; }}
  .segment.normal {{ border-left-color: #4ade80; }}
  .segment h3 {{ margin: 0 0 10px; }}
  .segment img {{ width: 100%; border-radius: 6px; margin: 10px 0; }}
  .scores {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 8px 0; }}
  .score-pill {{ background: #0f3460; border-radius: 20px; padding: 4px 12px; font-size: 0.85em; }}
  .elements {{ margin: 6px 0; }}
  .el-tag {{ display: inline-block; background: #374151; border-radius: 4px; padding: 2px 8px; margin: 2px; font-size: 0.82em; }}
  .reason {{ font-style: italic; color: #94a3b8; margin: 8px 0; }}
  .sub {{ background: #0d1117; border-radius: 6px; padding: 10px 14px; font-size: 0.85em; margin-top: 8px; }}
  .sub span {{ color: #60a5fa; font-weight: bold; }}
</style>
</head>
<body>
<h1>Ad Insertion Detection Report</h1>
<div class="meta">
  <table>
    <tr><td>Video</td><td>{video_name}</td></tr>
    <tr><td>Duration</td><td>{duration}</td></tr>
    <tr><td>FPS</td><td>{fps}</td></tr>
    <tr><td>Resolution</td><td>{resolution}</td></tr>
    <tr><td>Generated</td><td>{generated_at}</td></tr>
    <tr><td>Model</td><td>{model_path}</td></tr>
  </table>
</div>
<div class="summary">
  <div class="stat"><div class="val">{total_candidates}</div>Candidates</div>
  <div class="stat flagged"><div class="val flagged">{flagged_count}</div>Flagged</div>
  <div class="stat normal"><div class="val normal">{cleared_count}</div>Cleared</div>
  <div class="stat"><div class="val">{scene_cuts}</div>Scene Cuts</div>
</div>
<h2>Anomaly Timeline</h2>
<div class="timeline"><img src="data:image/png;base64,{timeline_b64}" /></div>
<h2>Detected Segments</h2>
{segments_html}
</body>
</html>
"""

SEGMENT_TEMPLATE = """\
<div class="segment {css_class}">
  <h3>Segment {idx}: {label_display}
    <small style="font-weight:normal; color:#94a3b8;">
      {start_ts} – {end_ts} ({duration_ms:.0f}ms, {num_frames} frames)
    </small>
  </h3>
  <div class="scores">
    <span class="score-pill">Temporal: {temporal:.3f}</span>
    <span class="score-pill">VLM: {vlm:.3f}</span>
    <span class="score-pill"><b>Final: {final:.3f}</b></span>
  </div>
  {elements_html}
  <p class="reason">{reason}</p>
  <img src="data:image/jpeg;base64,{sheet_b64}" />
  <div class="sub">
    <span>Before:</span> {sub_before}<br>
    <span>During:</span> {sub_during}<br>
    <span>After:</span>  {sub_after}
  </div>
</div>
"""


def _ms_to_ts(ms: float) -> str:
    total_s = ms / 1000
    m = int(total_s // 60)
    s = total_s % 60
    return f"{m:02d}:{s:06.3f}"


def generate_report(
    video_meta: VideoMeta,
    anomaly_result: AnomalyResult,
    results: list[DetectionResult],
    output_dir: str,
    model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    formats: tuple = ("json", "html"),
) -> dict[str, str]:
    """
    Generate JSON and/or HTML report.
    Returns dict of {format: output_path}.
    """
    os.makedirs(output_dir, exist_ok=True)
    video_name = Path(video_meta.video_path).stem
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out_paths = {}

    flagged = [r for r in results if r.is_flagged]
    cleared = [r for r in results if not r.is_flagged]

    # Build JSON
    if "json" in formats:
        json_data = {
            "video": {
                "path": video_meta.video_path,
                "fps": video_meta.fps,
                "duration_ms": video_meta.duration_ms,
                "width": video_meta.width,
                "height": video_meta.height,
            },
            "summary": {
                "total_candidates": len(results),
                "flagged": len(flagged),
                "cleared": len(cleared),
                "scene_cuts": len(anomaly_result.scene_cut_frames),
                "model": model_path,
                "generated_at": generated_at,
            },
            "detections": [
                {
                    "idx": i,
                    "start_frame": r.start_frame,
                    "end_frame": r.end_frame,
                    "start_ms": r.start_ms,
                    "end_ms": r.end_ms,
                    "duration_ms": r.duration_ms,
                    "temporal_score": r.temporal_score,
                    "vlm_confidence": r.vlm_confidence,
                    "final_score": r.final_score,
                    "is_flagged": r.is_flagged,
                    "label": r.label,
                    "detected_elements": r.detected_elements,
                    "reason": r.reason,
                    "subtitle_during": r.subtitle_during,
                    "sheet_path": r.sheet_path,
                }
                for i, r in enumerate(results)
            ],
        }
        json_path = os.path.join(output_dir, f"{video_name}_report.json")
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)
        out_paths["json"] = json_path

    # Build HTML
    if "html" in formats:
        # Timeline plot
        timeline_path = os.path.join(output_dir, f"{video_name}_timeline.png")
        _build_timeline_plot(video_meta, anomaly_result, results, timeline_path)
        timeline_b64 = _img_to_b64(timeline_path)

        # Segments HTML
        segments_html_parts = []
        for i, r in enumerate(results):
            css_class = "flagged" if r.is_flagged else "normal"
            label_display = (
                f'<span class="flagged">AD INSERTION DETECTED</span>'
                if r.is_flagged
                else '<span class="normal">Normal / Cleared</span>'
            )
            elements_html = ""
            if r.detected_elements:
                tags = "".join(f'<span class="el-tag">{e}</span>' for e in r.detected_elements)
                elements_html = f'<div class="elements">Detected: {tags}</div>'

            sheet_b64 = _img_to_b64(r.sheet_path) if os.path.exists(r.sheet_path) else ""

            segments_html_parts.append(
                SEGMENT_TEMPLATE.format(
                    idx=i + 1,
                    css_class=css_class,
                    label_display=label_display,
                    start_ts=_ms_to_ts(r.start_ms),
                    end_ts=_ms_to_ts(r.end_ms),
                    duration_ms=r.duration_ms,
                    num_frames=r.segment.num_frames,
                    temporal=r.temporal_score,
                    vlm=r.vlm_confidence,
                    final=r.final_score,
                    elements_html=elements_html,
                    reason=r.reason or "(no reason provided)",
                    sheet_b64=sheet_b64,
                    sub_before=r.subtitle_before or "(none)",
                    sub_during=r.subtitle_during or "(none)",
                    sub_after=r.subtitle_after or "(none)",
                )
            )

        duration_str = _ms_to_ts(video_meta.duration_ms)
        html = HTML_TEMPLATE.format(
            video_name=video_name,
            duration=duration_str,
            fps=f"{video_meta.fps:.2f}",
            resolution=f"{video_meta.width}×{video_meta.height}",
            generated_at=generated_at,
            model_path=model_path,
            total_candidates=len(results),
            flagged_count=len(flagged),
            cleared_count=len(cleared),
            scene_cuts=len(anomaly_result.scene_cut_frames),
            timeline_b64=timeline_b64,
            segments_html="\n".join(segments_html_parts) or "<p>No candidate segments detected.</p>",
        )
        html_path = os.path.join(output_dir, f"{video_name}_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        out_paths["html"] = html_path

    return out_paths
