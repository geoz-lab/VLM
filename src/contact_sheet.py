"""
Step 6: Generate contact sheets for each anomaly segment.

Layout (3 rows, variable columns):
  ROW 0 (green border):  context_frames BEFORE the segment
  ROW 1 (red border):    up to max_anomaly_frames FROM the segment (most anomalous)
  ROW 2 (green border):  context_frames AFTER the segment

Below each row: timestamp labels.
Below the entire sheet: subtitle strip (before / during / after).
"""

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.anomaly_detector import AnomalyResult, AnomalySegment
from src.frame_extractor import FrameStore
from src.subtitle_extractor import SubtitleTrack


# Colors (RGB)
COLOR_BEFORE = (60, 179, 113)    # medium sea green
COLOR_ANOMALY = (220, 50, 47)    # red
COLOR_AFTER = (60, 179, 113)
COLOR_BG = (30, 30, 30)
COLOR_TEXT = (240, 240, 240)
COLOR_SUBTITLE_BG = (20, 20, 20)


def _load_pil(frame_store: FrameStore, idx: int, tile_w: int, tile_h: int) -> Image.Image:
    bgr = frame_store.get_frame_image(idx)
    if bgr is None:
        # Return blank tile
        return Image.new("RGB", (tile_w, tile_h), (50, 50, 50))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    pil = pil.resize((tile_w, tile_h), Image.LANCZOS)
    return pil


def _add_border(img: Image.Image, color: tuple, width: int = 4) -> Image.Image:
    bordered = Image.new("RGB", (img.width + 2 * width, img.height + 2 * width), color)
    bordered.paste(img, (width, width))
    return bordered


def _get_font(size: int):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except Exception:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except Exception:
            return ImageFont.load_default()


def build_contact_sheet(
    segment: AnomalySegment,
    frame_store: FrameStore,
    subtitle_track: SubtitleTrack,
    output_path: str,
    context_frames: int = 3,
    max_anomaly_frames: int = 5,
    tile_w: int = 320,
    tile_h: int = 180,
    border_px: int = 4,
    font_size: int = 13,
    subtitle_strip_h: int = 70,
) -> str:
    """
    Build and save a contact sheet image for one anomaly segment.
    Returns the saved path.
    """
    total_frames = len(frame_store.frames)

    # Collect frame indices for each row
    before_indices = list(range(
        max(0, segment.start_frame - context_frames),
        segment.start_frame,
    ))
    # Pick most anomalous frames from segment (highest frame_distances)
    seg_indices = list(range(segment.start_frame, segment.end_frame + 1))
    if len(seg_indices) > max_anomaly_frames:
        # Sort by distance descending, pick top N, then re-sort by index
        ranked = sorted(
            enumerate(segment.frame_distances),
            key=lambda x: x[1],
            reverse=True,
        )[:max_anomaly_frames]
        selected_offsets = sorted([r[0] for r in ranked])
        seg_indices = [segment.start_frame + o for o in selected_offsets]

    after_indices = list(range(
        segment.end_frame + 1,
        min(total_frames, segment.end_frame + 1 + context_frames),
    ))

    # Pad shorter rows with blank markers (-1)
    max_cols = max(len(before_indices), len(seg_indices), len(after_indices), 1)
    rows_data = [
        (before_indices, COLOR_BEFORE, "BEFORE"),
        (seg_indices, COLOR_ANOMALY, "SUSPICIOUS"),
        (after_indices, COLOR_AFTER, "AFTER"),
    ]

    tile_with_border = tile_w + 2 * border_px
    tile_h_with_border = tile_h + 2 * border_px
    label_h = 20  # height of timestamp label under each tile

    sheet_w = max_cols * tile_with_border
    sheet_h = 3 * (tile_h_with_border + label_h) + subtitle_strip_h
    sheet = Image.new("RGB", (sheet_w, sheet_h), COLOR_BG)
    draw = ImageDraw.Draw(sheet)
    font_small = _get_font(font_size - 2)
    font_label = _get_font(font_size)
    font_sub = _get_font(font_size + 1)

    for row_idx, (indices, border_color, row_label) in enumerate(rows_data):
        y_tile = row_idx * (tile_h_with_border + label_h)
        for col_idx in range(max_cols):
            x = col_idx * tile_with_border
            if col_idx < len(indices):
                fidx = indices[col_idx]
                tile = _load_pil(frame_store, fidx, tile_w, tile_h)
                tile = _add_border(tile, border_color, border_px)
                sheet.paste(tile, (x, y_tile))
                # Timestamp label
                ts_ms = frame_store.get_timestamp_ms(fidx)
                ts_str = f"{ts_ms / 1000:.2f}s  #{fidx}"
                draw.text(
                    (x + border_px, y_tile + tile_h_with_border + 2),
                    ts_str,
                    font=font_small,
                    fill=COLOR_TEXT,
                )
            else:
                # Blank tile
                blank = Image.new("RGB", (tile_with_border, tile_h_with_border), (40, 40, 40))
                sheet.paste(blank, (x, y_tile))

        # Row label on the left edge (rotated)
        draw.text((2, y_tile + tile_h // 2), row_label, font=font_label, fill=border_color)

    # Subtitle strip
    sub_y = 3 * (tile_h_with_border + label_h)
    sheet_sub = Image.new("RGB", (sheet_w, subtitle_strip_h), COLOR_SUBTITLE_BG)
    draw_sub = ImageDraw.Draw(sheet_sub)

    ctx_ms = context_frames / frame_store.video_meta.fps * 1000
    sub_before = subtitle_track.get_text_for_range(
        segment.start_ms - ctx_ms, segment.start_ms
    ) or "(no subtitle)"
    sub_during = subtitle_track.get_text_for_range(
        segment.start_ms, segment.end_ms
    ) or "(no subtitle)"
    sub_after = subtitle_track.get_text_for_range(
        segment.end_ms, segment.end_ms + ctx_ms
    ) or "(no subtitle)"

    sub_text = (
        f"BEFORE: {sub_before[:80]}   |   "
        f"DURING: {sub_during[:80]}   |   "
        f"AFTER: {sub_after[:80]}"
    )
    draw_sub.text((8, 8), sub_text, font=font_sub, fill=COLOR_TEXT)
    draw_sub.text(
        (8, subtitle_strip_h - 20),
        f"Segment: frame {segment.start_frame}–{segment.end_frame}  "
        f"({segment.start_ms / 1000:.2f}s – {segment.end_ms / 1000:.2f}s)  "
        f"duration: {segment.duration_ms:.0f}ms  "
        f"temporal_score: {segment.temporal_score:.3f}",
        font=font_small,
        fill=(180, 180, 100),
    )
    sheet.paste(sheet_sub, (0, sub_y))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sheet.save(output_path, quality=95)
    return output_path


def build_all_contact_sheets(
    anomaly_result: AnomalyResult,
    frame_store: FrameStore,
    subtitle_track: SubtitleTrack,
    output_dir: str,
    **kwargs,
) -> list[dict]:
    """
    Build contact sheets for all detected segments.
    Returns list of dicts: {segment, sheet_path, subtitle_before, subtitle_during, subtitle_after}
    """
    os.makedirs(output_dir, exist_ok=True)
    sheets = []
    context_frames = kwargs.get("context_frames", 3)

    for i, segment in enumerate(anomaly_result.segments):
        sheet_path = os.path.join(output_dir, f"segment_{i:04d}.jpg")
        build_contact_sheet(segment, frame_store, subtitle_track, sheet_path, **kwargs)

        ctx_ms = context_frames / frame_store.video_meta.fps * 1000
        sheets.append({
            "segment": segment,
            "sheet_path": sheet_path,
            "subtitle_before": subtitle_track.get_text_for_range(
                segment.start_ms - ctx_ms, segment.start_ms
            ),
            "subtitle_during": subtitle_track.get_text_for_range(
                segment.start_ms, segment.end_ms
            ),
            "subtitle_after": subtitle_track.get_text_for_range(
                segment.end_ms, segment.end_ms + ctx_ms
            ),
        })

    return sheets
