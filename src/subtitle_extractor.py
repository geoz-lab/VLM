"""
Subtitle / caption extraction.

Priority:
1. Load a .srt file with the same basename as the video.
2. Fall back to Whisper transcription from audio.

Returns subtitle text mapped to time ranges, with a helper to
look up what was being said during a given time window.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pysrt


@dataclass
class SubtitleEntry:
    start_ms: float
    end_ms: float
    text: str


class SubtitleTrack:
    def __init__(self, entries: list[SubtitleEntry]):
        self.entries = sorted(entries, key=lambda e: e.start_ms)

    def get_text_for_range(self, start_ms: float, end_ms: float) -> str:
        """Return concatenated subtitle text overlapping [start_ms, end_ms]."""
        texts = []
        for entry in self.entries:
            if entry.end_ms < start_ms:
                continue
            if entry.start_ms > end_ms:
                break
            texts.append(entry.text.strip())
        return " ".join(texts) if texts else ""

    def is_empty(self) -> bool:
        return len(self.entries) == 0


def _pysrt_time_to_ms(t) -> float:
    return (t.hours * 3600 + t.minutes * 60 + t.seconds) * 1000 + t.milliseconds


def load_srt(srt_path: str) -> SubtitleTrack:
    subs = pysrt.open(srt_path, encoding="utf-8")
    entries = [
        SubtitleEntry(
            start_ms=_pysrt_time_to_ms(s.start),
            end_ms=_pysrt_time_to_ms(s.end),
            text=s.text,
        )
        for s in subs
    ]
    return SubtitleTrack(entries)


def _transcribe_with_whisper(video_path: str) -> SubtitleTrack:
    """Use OpenAI Whisper to transcribe audio from video."""
    try:
        import whisper
    except ImportError:
        raise ImportError("whisper not installed. Run: pip install openai-whisper")

    model = whisper.load_model("base")
    result = model.transcribe(video_path, word_timestamps=False)

    entries = []
    for seg in result.get("segments", []):
        entries.append(
            SubtitleEntry(
                start_ms=seg["start"] * 1000,
                end_ms=seg["end"] * 1000,
                text=seg["text"].strip(),
            )
        )
    return SubtitleTrack(entries)


def load_subtitles(video_path: str, srt_path: Optional[str] = None) -> SubtitleTrack:
    """
    Load subtitles for a video.

    Checks (in order):
      1. Explicit srt_path argument.
      2. <video_basename>.srt in same directory.
      3. <video_basename>.en.srt in same directory.
      4. Whisper transcription (slower, requires ffmpeg).
    """
    # Explicit path
    if srt_path and os.path.exists(srt_path):
        return load_srt(srt_path)

    # Auto-discover srt next to video
    video_stem = Path(video_path).stem
    video_dir = Path(video_path).parent
    for suffix in [".srt", ".en.srt", ".eng.srt"]:
        candidate = video_dir / (video_stem + suffix)
        if candidate.exists():
            return load_srt(str(candidate))

    # Whisper fallback
    print("[subtitle_extractor] No SRT found — running Whisper transcription (may be slow)")
    try:
        return _transcribe_with_whisper(video_path)
    except Exception as e:
        print(f"[subtitle_extractor] Whisper failed: {e}. Proceeding without subtitles.")
        return SubtitleTrack([])
