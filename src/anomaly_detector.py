"""
Steps 3–5: Temporal anomaly detection.

Two-signal approach:

Signal A — consecutive distance:
  consec[i] = pHash(frame[i], frame[i-1])
  An injection creates TWO spikes: one entering the inserted segment,
  one leaving it. A legitimate scene cut creates only ONE spike.

Signal B — A–X–A context distance (original):
  dist[i] = how different frame[i] is from frame[i±window]
  Good for catching the halo frames around the injection.

We merge both signals: within N frames of a consec spike pair,
expand to cover the full injection segment.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

import imagehash
import numpy as np
from PIL import Image
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


def compute_all_hashes(
    frame_store: FrameStore,
    hash_size: int = 16,
    show_progress: bool = True,
) -> list:
    """Compute pHash for every frame. Returns list of hash objects (or None)."""
    n = len(frame_store.frames)
    hashes = []
    for idx in tqdm(range(n), desc="Computing pHashes", disable=not show_progress):
        pil = _frame_to_pil(frame_store, idx)
        hashes.append(imagehash.phash(pil, hash_size=hash_size) if pil is not None else None)
    return hashes


def compute_consecutive_distances(hashes: list) -> list[float]:
    """
    consec[i] = pHash(frame[i], frame[i-1]).
    consec[0] = 0.

    A sudden spike = visual boundary. An injection creates TWO spikes
    (enter + exit). A legitimate scene cut creates ONE spike.
    """
    n = len(hashes)
    dists = [0.0]
    for i in range(1, n):
        if hashes[i] is None or hashes[i - 1] is None:
            dists.append(0.0)
        else:
            dists.append(float(hashes[i] - hashes[i - 1]))
    return dists


def compute_phash_distances(
    frame_store: FrameStore,
    hash_size: int = 16,
    context_window: int = 15,
    show_progress: bool = True,
) -> list[float]:
    """
    A–X–A context distance: for each frame i,
        distance(i) = avg(pHash(i, i-window), pHash(i, i+window))
    with boost/dampen based on whether the two neighbors are similar.
    Used as Signal B for temporal_score computation.
    """
    hashes = compute_all_hashes(frame_store, hash_size, show_progress)
    n = len(hashes)
    distances = []
    for i in range(n):
        if hashes[i] is None:
            distances.append(0.0)
            continue
        left_idx = max(0, i - context_window)
        right_idx = min(n - 1, i + context_window)
        left_dist = float(hashes[i] - hashes[left_idx]) if hashes[left_idx] is not None and left_idx != i else 0.0
        right_dist = float(hashes[i] - hashes[right_idx]) if hashes[right_idx] is not None and right_idx != i else 0.0
        if hashes[left_idx] is not None and hashes[right_idx] is not None:
            neighbor_sim = float(hashes[left_idx] - hashes[right_idx])
            factor = 1.2 if neighbor_sim < 20 else 0.6
            distance = (left_dist + right_dist) / 2.0 * factor
        else:
            distance = (left_dist + right_dist) / 2.0
        distances.append(distance)
    return distances


def detect_scene_cuts(distances: list[float], scene_cut_threshold: float = 80.0) -> list[int]:
    """Frames where A-X-A distance exceeds threshold are likely scene cuts."""
    return [i for i, d in enumerate(distances) if d >= scene_cut_threshold]


def _find_injection_segments_from_consecutive(
    consec: list[float],
    spike_threshold: float,
    min_segment_frames: int,
    max_segment_frames: int,
    merge_gap_frames: int,
) -> list[tuple[int, int]]:
    """
    Find injection patterns using consecutive distances.

    Key insight: an injection is a BLOCK of high consecutive distances
    (each ad frame is different from the next because varied ad content).
    A single scene cut creates a BRIEF spike (1–4 frames).

    Detection: find runs of frames where consec > spike_threshold.
    - Block length in [min_segment, max_segment] → injection candidate
    - Block length < min_segment → scene cut (excluded)
    - Block length > max_segment → too long, excluded

    Blocks within merge_gap_frames of each other are merged first.
    """
    spikes = [i for i, d in enumerate(consec) if d > spike_threshold]
    if not spikes:
        return []

    # Merge nearby spikes into blocks
    blocks: list[tuple[int, int]] = []
    g_start, g_end = spikes[0], spikes[0]
    for s in spikes[1:]:
        if s - g_end <= merge_gap_frames:
            g_end = s
        else:
            blocks.append((g_start, g_end))
            g_start, g_end = s, s
    blocks.append((g_start, g_end))

    # Filter blocks by size
    candidates = []
    for start, end in blocks:
        length = end - start + 1
        if min_segment_frames <= length <= max_segment_frames:
            candidates.append((start, end))
        # Short blocks (< min_segment) are scene cuts → ignored
        # Long blocks (> max_segment) are ignored (too long to be an ad burst)

    return candidates


def detect_anomaly_segments(
    frame_store: FrameStore,
    distances: list[float],
    phash_threshold: float = 25.0,
    scene_cut_threshold: float = 80.0,
    min_segment_frames: int = 5,
    max_segment_frames: int = 90,
    merge_gap_frames: int = 10,
    hashes: list = None,
) -> AnomalyResult:
    """
    Identify suspicious segments using both consecutive and A–X–A signals.

    If hashes are provided (pre-computed), uses them for consecutive distance.
    Falls back to A-X-A-only if hashes not provided.
    """
    n = len(distances)
    MAX_HASH_DIST = 256.0
    fps = frame_store.video_meta.fps

    # Signal B: A-X-A based scene cut detection
    scene_cuts = set(detect_scene_cuts(distances, scene_cut_threshold))

    candidate_ranges: list[tuple[int, int]] = []

    if hashes is not None:
        # Signal A: consecutive distance → find paired spikes (A-X-A pattern)
        consec = compute_consecutive_distances(hashes)
        # Spike threshold: use 3× the median of the top-25% to adapt to video content
        nonzero = sorted([d for d in consec if d > 0])
        if nonzero:
            p75 = nonzero[int(len(nonzero) * 0.75)]
            spike_thresh = max(phash_threshold, p75 * 3.0)
        else:
            spike_thresh = phash_threshold

        candidate_ranges = _find_injection_segments_from_consecutive(
            consec, spike_thresh, min_segment_frames, max_segment_frames, merge_gap_frames
        )

    # Signal B fallback: A-X-A based anomalous frame groups
    if not candidate_ranges:
        anomalous = [i for i, d in enumerate(distances) if d >= phash_threshold and i not in scene_cuts]
        if anomalous:
            segments_raw = []
            g_start, g_end = anomalous[0], anomalous[0]
            for idx in anomalous[1:]:
                if idx - g_end <= merge_gap_frames:
                    g_end = idx
                else:
                    segments_raw.append((g_start, g_end))
                    g_start, g_end = idx, idx
            segments_raw.append((g_start, g_end))
            candidate_ranges = [(s, e) for s, e in segments_raw
                                if min_segment_frames <= (e - s + 1) <= max_segment_frames]

    # Build AnomalySegment objects
    segments = []
    for start, end in candidate_ranges:
        start = max(0, start)
        end = min(n - 1, end)
        length = end - start + 1
        if length < min_segment_frames or length > max_segment_frames:
            continue

        seg_distances = distances[start: end + 1]
        peak_dist = max(seg_distances) if seg_distances else 0.0
        mean_dist = float(np.mean(seg_distances)) if seg_distances else 0.0
        temporal_score = min(1.0, peak_dist / MAX_HASH_DIST * 2.0)

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
    """
    Full anomaly detection pipeline. Entry point for pipeline.py.
    Computes hashes once, then runs both Signal A (consecutive) and
    Signal B (A-X-A context) detection.
    """
    # Compute hashes once
    hashes = compute_all_hashes(frame_store, hash_size=hash_size)

    # Signal B: A-X-A context distances (used for temporal_score + scene cut detection)
    n = len(hashes)
    distances = []
    for i in range(n):
        if hashes[i] is None:
            distances.append(0.0)
            continue
        left_idx = max(0, i - context_window)
        right_idx = min(n - 1, i + context_window)
        left_dist = float(hashes[i] - hashes[left_idx]) if hashes[left_idx] is not None and left_idx != i else 0.0
        right_dist = float(hashes[i] - hashes[right_idx]) if hashes[right_idx] is not None and right_idx != i else 0.0
        if hashes[left_idx] is not None and hashes[right_idx] is not None:
            neighbor_sim = float(hashes[left_idx] - hashes[right_idx])
            factor = 1.2 if neighbor_sim < 20 else 0.6
            distance = (left_dist + right_dist) / 2.0 * factor
        else:
            distance = (left_dist + right_dist) / 2.0
        distances.append(distance)

    return detect_anomaly_segments(
        frame_store,
        distances,
        phash_threshold=phash_threshold,
        scene_cut_threshold=scene_cut_threshold,
        min_segment_frames=min_segment_frames,
        max_segment_frames=max_segment_frames,
        merge_gap_frames=merge_gap_frames,
        hashes=hashes,  # pass for Signal A
    )
