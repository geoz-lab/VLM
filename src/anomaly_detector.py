"""
Steps 3–5: Temporal anomaly detection.

Computes per-frame visual distance, detects A–X–A pattern anomalies,
and merges nearby suspicious frames into segments.

A–X–A pattern: frame[i] is very different from frame[i-window] AND
               frame[i+window], but frame[i-window] ≈ frame[i+window].
This catches inserted content surrounded by continuous original footage.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

import imagehash
import numpy as np
from PIL import Image
from scipy.signal import find_peaks
from tqdm import tqdm

from src.frame_extractor import FrameStore


@dataclass
class AnomalySegment:
    """A suspicious segment of consecutive frames."""
    start_frame: int
    end_frame: int
    start_ms: float
    end_ms: float
    duration_ms: float
    peak_distance: float        # Max pHash distance within segment
    mean_distance: float        # Mean pHash distance within segment
    temporal_score: float       # Normalized 0–1 anomaly score
    frame_distances: list[float] = field(default_factory=list)

    @property
    def num_frames(self) -> int:
        return self.end_frame - self.start_frame + 1


@dataclass
class AnomalyResult:
    segments: list[AnomalySegment]
    frame_distances: list[float]       # Per-frame pHash distance from context
    scene_cut_frames: list[int]        # Hard scene cuts (not insertions)
    fps: float


def _frame_to_pil(frame_store: FrameStore, idx: int) -> Optional[Image.Image]:
    img_bgr = frame_store.get_frame_image(idx)
    if img_bgr is None:
        return None
    import cv2
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def compute_phash_distances(
    frame_store: FrameStore,
    hash_size: int = 16,
    context_window: int = 15,
    show_progress: bool = True,
) -> list[float]:
    """
    For each frame i, compute:
        distance(i) = max(phash(i, i-window), phash(i, i+window))

    This captures how much frame i differs from its temporal neighborhood,
    independent of gradual scene changes.
    """
    n = len(frame_store.frames)
    hashes = []

    # Compute all hashes first
    for idx in tqdm(range(n), desc="Computing pHashes", disable=not show_progress):
        pil = _frame_to_pil(frame_store, idx)
        if pil is None:
            hashes.append(None)
        else:
            hashes.append(imagehash.phash(pil, hash_size=hash_size))

    distances = []
    for i in range(n):
        if hashes[i] is None:
            distances.append(0.0)
            continue

        # Compare against left context
        left_idx = max(0, i - context_window)
        right_idx = min(n - 1, i + context_window)

        left_dist = 0.0
        right_dist = 0.0

        if hashes[left_idx] is not None and left_idx != i:
            left_dist = float(hashes[i] - hashes[left_idx])
        if hashes[right_idx] is not None and right_idx != i:
            right_dist = float(hashes[i] - hashes[right_idx])

        # A–X–A: both neighbors are far from i, and similar to each other
        if hashes[left_idx] is not None and hashes[right_idx] is not None:
            neighbor_similarity = float(hashes[left_idx] - hashes[right_idx])
            # Boost score if neighbors are similar to each other (true insertion)
            # but lower score if neighbors are also different (scene transition)
            if neighbor_similarity < 20:  # neighbors are similar = true insertion
                distance = (left_dist + right_dist) / 2.0 * 1.2  # boost
            else:
                distance = (left_dist + right_dist) / 2.0 * 0.6  # dampen
        else:
            distance = (left_dist + right_dist) / 2.0

        distances.append(distance)

    return distances


def detect_scene_cuts(distances: list[float], scene_cut_threshold: float = 80.0) -> list[int]:
    """Frames where distance exceeds scene_cut_threshold are hard scene cuts."""
    return [i for i, d in enumerate(distances) if d >= scene_cut_threshold]


def detect_anomaly_segments(
    frame_store: FrameStore,
    distances: list[float],
    phash_threshold: float = 25.0,
    scene_cut_threshold: float = 80.0,
    min_segment_frames: int = 5,
    max_segment_frames: int = 90,
    merge_gap_frames: int = 10,
) -> AnomalyResult:
    """
    Identify suspicious segments using the computed per-frame distances.

    Steps:
    1. Threshold distances to find anomalous frame indices.
    2. Exclude hard scene cuts (too abrupt = legitimate cut, not insertion).
    3. Merge nearby anomalous frames into segments.
    4. Filter segments by min/max length.
    5. Compute temporal_score per segment.
    """
    n = len(distances)
    max_possible_distance = float(frame_store.video_meta.fps * 16 * 16)  # rough upper bound
    # In practice pHash max distance ≈ hash_size^2 bits; for 16x16 = 256
    MAX_HASH_DIST = 256.0

    scene_cuts = set(detect_scene_cuts(distances, scene_cut_threshold))

    # Step 1: Find anomalous frames (above threshold, not scene cuts)
    anomalous = [
        i for i, d in enumerate(distances)
        if d >= phash_threshold and i not in scene_cuts
    ]

    if not anomalous:
        return AnomalyResult(
            segments=[],
            frame_distances=distances,
            scene_cut_frames=sorted(scene_cuts),
            fps=frame_store.video_meta.fps,
        )

    # Step 2: Merge nearby anomalous frames
    segments_raw = []
    group_start = anomalous[0]
    group_end = anomalous[0]

    for idx in anomalous[1:]:
        if idx - group_end <= merge_gap_frames:
            group_end = idx
        else:
            segments_raw.append((group_start, group_end))
            group_start = idx
            group_end = idx
    segments_raw.append((group_start, group_end))

    # Step 3: Filter by length and build AnomalySegment objects
    fps = frame_store.video_meta.fps
    segments = []

    for start, end in segments_raw:
        length = end - start + 1
        if length < min_segment_frames or length > max_segment_frames:
            continue

        seg_distances = distances[start: end + 1]
        peak_dist = max(seg_distances)
        mean_dist = float(np.mean(seg_distances))
        temporal_score = min(1.0, mean_dist / MAX_HASH_DIST * 4.0)  # scale to 0-1

        segments.append(
            AnomalySegment(
                start_frame=start,
                end_frame=end,
                start_ms=frame_store.get_timestamp_ms(start),
                end_ms=frame_store.get_timestamp_ms(end),
                duration_ms=frame_store.get_timestamp_ms(end) - frame_store.get_timestamp_ms(start),
                peak_distance=peak_dist,
                mean_distance=mean_dist,
                temporal_score=temporal_score,
                frame_distances=seg_distances,
            )
        )

    return AnomalyResult(
        segments=segments,
        frame_distances=distances,
        scene_cut_frames=sorted(scene_cuts),
        fps=fps,
    )


def run_anomaly_detection(
    frame_store: FrameStore,
    phash_threshold: float = 25.0,
    scene_cut_threshold: float = 80.0,
    context_window: int = 15,
    min_segment_frames: int = 5,
    max_segment_frames: int = 90,
    merge_gap_frames: int = 10,
    hash_size: int = 16,
) -> AnomalyResult:
    """Full anomaly detection pipeline. Entry point for pipeline.py."""
    distances = compute_phash_distances(
        frame_store,
        hash_size=hash_size,
        context_window=context_window,
    )
    return detect_anomaly_segments(
        frame_store,
        distances,
        phash_threshold=phash_threshold,
        scene_cut_threshold=scene_cut_threshold,
        min_segment_frames=min_segment_frames,
        max_segment_frames=max_segment_frames,
        merge_gap_frames=merge_gap_frames,
    )
