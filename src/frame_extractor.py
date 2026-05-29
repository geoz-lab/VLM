"""
Step 2: Extract frames from video at full FPS.

Outputs a structured FrameStore: list of FrameMeta objects + raw frame cache.
"""

import os
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm


@dataclass
class FrameMeta:
    frame_idx: int           # 0-based absolute frame index
    timestamp_ms: float      # Milliseconds from video start
    file_path: str           # Path to saved JPEG (or None if in-memory only)
    width: int
    height: int


@dataclass
class VideoMeta:
    video_path: str
    fps: float
    total_frames: int
    duration_ms: float
    width: int
    height: int
    has_audio: bool


@dataclass
class FrameStore:
    video_meta: VideoMeta
    frames: list[FrameMeta] = field(default_factory=list)
    output_dir: Optional[str] = None

    def get_frame_image(self, frame_idx: int) -> Optional[np.ndarray]:
        """Load frame image from disk (lazy load)."""
        meta = self.frames[frame_idx]
        if meta.file_path and os.path.exists(meta.file_path):
            return cv2.imread(meta.file_path)
        return None

    def get_timestamp_ms(self, frame_idx: int) -> float:
        return self.frames[frame_idx].timestamp_ms

    def save_metadata(self):
        if self.output_dir:
            meta_path = os.path.join(self.output_dir, "frame_meta.json")
            data = {
                "video_meta": asdict(self.video_meta),
                "frames": [asdict(f) for f in self.frames],
            }
            with open(meta_path, "w") as f:
                json.dump(data, f, indent=2)


def extract_frames(
    video_path: str,
    output_dir: str,
    target_fps: Optional[float] = None,
    save_frames: bool = True,
    frame_quality: int = 95,
    show_progress: bool = True,
) -> FrameStore:
    """
    Extract frames from a video file.

    Args:
        video_path:   Path to input video.
        output_dir:   Directory to save extracted frames.
        target_fps:   If None, extract at native FPS. Otherwise resample.
        save_frames:  Whether to write JPEG files to disk.
        frame_quality: JPEG quality (1–100).
        show_progress: Show tqdm progress bar.

    Returns:
        FrameStore with all frame metadata.
    """
    video_path = str(video_path)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_native_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_ms = (total_native_frames / native_fps) * 1000 if native_fps > 0 else 0

    if native_fps <= 0:
        raise RuntimeError(f"Invalid FPS ({native_fps}) for video: {video_path}")

    effective_fps = target_fps if target_fps else native_fps
    frame_step = max(1, round(native_fps / effective_fps))

    video_meta = VideoMeta(
        video_path=video_path,
        fps=effective_fps,
        total_frames=total_native_frames,
        duration_ms=duration_ms,
        width=width,
        height=height,
        has_audio=True,  # assume; ffprobe would be more accurate
    )

    if save_frames:
        os.makedirs(output_dir, exist_ok=True)

    store = FrameStore(video_meta=video_meta, output_dir=output_dir if save_frames else None)

    native_idx = 0
    extracted_idx = 0
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, frame_quality]

    pbar = tqdm(total=total_native_frames, desc="Extracting frames", disable=not show_progress)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if native_idx % frame_step == 0:
            timestamp_ms = (native_idx / native_fps) * 1000
            frame_path = None

            if save_frames:
                frame_path = os.path.join(output_dir, f"frame_{extracted_idx:06d}.jpg")
                cv2.imwrite(frame_path, frame, encode_params)

            store.frames.append(
                FrameMeta(
                    frame_idx=extracted_idx,
                    timestamp_ms=timestamp_ms,
                    file_path=frame_path,
                    width=width,
                    height=height,
                )
            )
            extracted_idx += 1

        native_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    if save_frames:
        store.save_metadata()

    return store
