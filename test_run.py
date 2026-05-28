"""
Test run: exercises the full pipeline with a mock VLM classifier.
Validates steps 1–6 with real data, then stubs steps 7–8 to complete report generation.

Run: python test_run.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import random
from pathlib import Path
from unittest.mock import patch

from rich.console import Console
from rich.panel import Panel

console = Console()


# ── Mock Qwen classifier (replaces GPU inference) ──────────────────────────
def mock_classify(self, sheet_path, subtitle_before="", subtitle_during="",
                  subtitle_after="", temporal_score=0.0, duration_ms=0.0, num_frames=0):
    from src.qwen_classifier import ClassifierOutput

    # Simulate model: high temporal score → likely ad; also check subtitle cues
    is_ad = temporal_score > 0.15
    ad_keywords = ["sponsor", "promo", "discount", "code", "offer", "visit", "buy", "shop"]
    if any(k in (subtitle_during or "").lower() for k in ad_keywords):
        is_ad = True

    if is_ad:
        elements = random.choice([
            ["logo", "promotional text"],
            ["QR code", "URL"],
            ["product image", "price tag"],
            ["brand name", "call-to-action button"],
        ])
        return ClassifierOutput(
            label="ad_insertion",
            confidence=round(random.uniform(0.78, 0.97), 3),
            detected_elements=elements,
            reason="Frame content is visually inconsistent with surrounding video — contains promotional elements.",
            raw_response='{"label": "ad_insertion", "confidence": 0.92, ...}',
        )
    else:
        return ClassifierOutput(
            label="normal",
            confidence=round(random.uniform(0.75, 0.92), 3),
            detected_elements=[],
            reason="Frames appear to be a natural scene transition consistent with surrounding content.",
            raw_response='{"label": "normal", "confidence": 0.85, ...}',
        )


def main():
    console.print(Panel.fit(
        "[bold cyan]VLM Ad Insertion Detector — Test Run[/bold cyan]\n"
        "[dim]VLM step mocked (no GPU required)[/dim]",
        border_style="cyan",
    ))

    video_path = "video_input/test_scene_injected.mp4"
    if not os.path.exists(video_path):
        console.print(f"[red]Video not found: {video_path}[/red]")
        return

    # Load ground truth to compare against
    gt_path = "data/synthetic/annotations/test_scene_injected.json"
    with open(gt_path) as f:
        ground_truth = json.load(f)

    console.print(f"\n[bold]Ground truth injections:[/bold]")
    for i, inj in enumerate(ground_truth["injections"]):
        console.print(
            f"  #{i+1}  frame {inj['output_start_frame']}–{inj['output_end_frame']}  "
            f"({inj['output_start_ms']/1000:.2f}s–{inj['output_end_ms']/1000:.2f}s)  "
            f"{inj['duration_ms']:.0f}ms"
        )

    console.print()

    # Patch the VLM classifier
    import src.qwen_classifier  # ensure module is imported before patching
    with patch.object(src.qwen_classifier.QwenVLClassifier, "classify", mock_classify):
        from src.pipeline import load_config, run_pipeline

        config = load_config("configs/pipeline_config.yaml")
        config["classifier"]["use_finetuned"] = False

        results = run_pipeline(
            video_path=video_path,
            config=config,
            keep_frames=False,
        )

    # ── Evaluation vs ground truth ──────────────────────────────────────────
    console.print("\n[bold cyan]Evaluation vs Ground Truth[/bold cyan]")
    gt_windows = [
        (inj["output_start_ms"], inj["output_end_ms"])
        for inj in ground_truth["injections"]
    ]
    detections = results["detections"]
    flagged = [d for d in detections if d.is_flagged]

    tp, fp, fn = 0, 0, 0
    matched_gt = set()

    for det in flagged:
        matched = False
        for gi, (gt_start, gt_end) in enumerate(gt_windows):
            # Check overlap
            overlap = max(0, min(det.end_ms, gt_end) - max(det.start_ms, gt_start))
            span = max(det.end_ms, gt_end) - min(det.start_ms, gt_start)
            iou = overlap / span if span > 0 else 0
            if iou > 0.2 and gi not in matched_gt:
                tp += 1
                matched_gt.add(gi)
                matched = True
                break
        if not matched:
            fp += 1

    fn = len(gt_windows) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    from rich.table import Table
    eval_table = Table(show_header=True, header_style="bold yellow")
    eval_table.add_column("Metric", width=25)
    eval_table.add_column("Value", width=10)
    eval_table.add_row("Ground truth injections", str(len(gt_windows)))
    eval_table.add_row("Detected (all candidates)", str(len(detections)))
    eval_table.add_row("Flagged", str(len(flagged)))
    eval_table.add_row("True Positives", f"[green]{tp}[/green]")
    eval_table.add_row("False Positives", f"[red]{fp}[/red]")
    eval_table.add_row("False Negatives (missed)", f"[yellow]{fn}[/yellow]")
    eval_table.add_row("Precision", f"{precision:.2f}")
    eval_table.add_row("Recall", f"{recall:.2f}")
    eval_table.add_row("F1 Score", f"[bold]{f1:.2f}[/bold]")
    console.print(eval_table)

    # Report paths
    console.print("\n[bold]Output reports:[/bold]")
    for fmt, path in results["report_paths"].items():
        console.print(f"  [{fmt}] {path}")

    console.print("\n[dim]Note: VLM was mocked. For real inference, install torch + "
                  "Qwen2.5-VL-7B-Instruct and run: python main.py --video <path>[/dim]")


if __name__ == "__main__":
    main()
