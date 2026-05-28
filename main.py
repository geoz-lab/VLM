"""
Entry point: watch ./video_input and run detection pipeline on any video dropped in.

Usage:
  # Single video:
  python main.py --video path/to/video.mp4

  # Watch mode (process all videos in ./video_input):
  python main.py --watch

  # Batch mode:
  python main.py --input_dir video_input
"""

import argparse
import os
import sys
import time
from pathlib import Path

from rich.console import Console

from src.pipeline import load_config, run_pipeline

console = Console()

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
PROCESSED_MARKER = ".processed"


def is_video(path: str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTS


def mark_processed(video_path: str):
    marker = video_path + PROCESSED_MARKER
    Path(marker).touch()


def is_already_processed(video_path: str) -> bool:
    return os.path.exists(video_path + PROCESSED_MARKER)


def process_single(video_path: str, config: dict, keep_frames: bool = False):
    try:
        results = run_pipeline(
            video_path=video_path,
            config=config,
            keep_frames=keep_frames,
        )
        mark_processed(video_path)
        flagged = [r for r in results["detections"] if r.is_flagged]
        if flagged:
            console.print(f"[bold red]ALERT: {len(flagged)} ad insertion(s) detected in {Path(video_path).name}[/bold red]")
        else:
            console.print(f"[green]Clean: no ad insertions detected in {Path(video_path).name}[/green]")
        return results
    except Exception as e:
        console.print(f"[red]Error processing {video_path}: {e}[/red]")
        import traceback
        traceback.print_exc()
        return None


def watch_directory(watch_dir: str, config: dict, poll_interval: int = 5):
    console.print(f"[bold]Watching {watch_dir} for new videos (Ctrl+C to stop)...[/bold]")
    os.makedirs(watch_dir, exist_ok=True)

    while True:
        for fname in sorted(os.listdir(watch_dir)):
            fpath = os.path.join(watch_dir, fname)
            if is_video(fpath) and not is_already_processed(fpath):
                console.print(f"\n[yellow]New video detected: {fname}[/yellow]")
                process_single(fpath, config)
        time.sleep(poll_interval)


def batch_process(input_dir: str, config: dict, keep_frames: bool = False):
    videos = [
        os.path.join(input_dir, f)
        for f in sorted(os.listdir(input_dir))
        if is_video(os.path.join(input_dir, f))
    ]
    if not videos:
        console.print(f"[yellow]No videos found in {input_dir}[/yellow]")
        return

    console.print(f"[bold]Processing {len(videos)} video(s) from {input_dir}[/bold]")
    for vpath in videos:
        process_single(vpath, config, keep_frames=keep_frames)


def main():
    parser = argparse.ArgumentParser(
        description="VLM Ad Insertion Detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --video my_video.mp4
  python main.py --watch
  python main.py --input_dir video_input --keep_frames
        """,
    )
    parser.add_argument("--video", help="Path to a single video to analyze")
    parser.add_argument("--input_dir", default="video_input", help="Batch: directory of videos")
    parser.add_argument("--watch", action="store_true", help="Watch ./video_input for new files")
    parser.add_argument("--config", default="configs/pipeline_config.yaml", help="Pipeline config")
    parser.add_argument("--keep_frames", action="store_true", help="Keep extracted frames on disk")
    parser.add_argument("--use_finetuned", action="store_true",
                        help="Use fine-tuned model (set finetuned_path in config)")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        console.print(f"[red]Config not found: {args.config}[/red]")
        sys.exit(1)

    config = load_config(args.config)

    if args.use_finetuned:
        config["classifier"]["use_finetuned"] = True

    if args.video:
        process_single(args.video, config, keep_frames=args.keep_frames)
    elif args.watch:
        watch_directory("video_input", config)
    else:
        batch_process(args.input_dir, config, keep_frames=args.keep_frames)


if __name__ == "__main__":
    main()
