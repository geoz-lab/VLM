# VLM Ad Insertion Detector

Detects unauthorized inserted frames in videos — short promotional bursts (≥300ms) that creators inject to show ads, QR codes, or sponsor content that bypasses platform moderation.

---

## Problem

At 30 fps, a 300 ms insertion = **9 consecutive frames**. These are:
- Invisible during normal playback (too fast to consciously process)
- Visible when paused — making them a common vector for smuggling unapproved content
- Distinguishable from legitimate scene cuts by their A–X–A temporal pattern (surrounding footage is continuous; only the inserted segment is visually foreign)

---

## Detection Pipeline

```
Input Video
    │
    ▼
[Step 1] Frame Extraction (full FPS, ~30fps)
    │  src/frame_extractor.py
    │  → FrameStore: per-frame JPEG + timestamp metadata
    │
    ▼
[Steps 2–4] Temporal Anomaly Detection
    │  src/anomaly_detector.py
    │  • Compute pHash for every frame
    │  • For each frame i: distance = avg(hash[i] vs hash[i−15], hash[i] vs hash[i+15])
    │  • A–X–A boost: if neighbors are similar to each other → true insertion signal
    │  • Scene cut filter: distance > 80 → hard cut, not insertion
    │
    ▼
[Step 5] Segment Merging
    │  src/anomaly_detector.py
    │  • Merge anomalous frames within 10-frame gap
    │  • Filter: 5–90 frames (167ms – 3s at 30fps)
    │  → AnomalyResult: list of AnomalySegments with temporal_score
    │
    ▼
[Step 6] Contact Sheet Generation
    │  src/contact_sheet.py
    │  • 3-row grid: [BEFORE context] [SUSPICIOUS frames] [AFTER context]
    │  • Green border = normal context, Red border = suspicious segment
    │  • Subtitle text strip below (before / during / after)
    │  → One JPEG per segment
    │
    ▼
[Steps 7–8] Qwen2.5-VL-7B-Instruct Classification
    │  src/qwen_classifier.py
    │  • Input: contact sheet image + subtitle context + temporal metadata
    │  • Output: { label, confidence, detected_elements, reason }
    │  • Structured JSON output with low-temperature sampling
    │
    ▼
[Step 9] Score Fusion
    │  src/score_combiner.py
    │  final_score = 0.35 × temporal_score + 0.65 × vlm_confidence
    │  is_flagged = final_score ≥ 0.55
    │
    ▼
[Step 10] Report Generation
       src/report_generator.py
       → reports/<video_name>/<video_name>_report.html  (visual, embeds contact sheets)
       → reports/<video_name>/<video_name>_report.json  (machine-readable)
```

---

## Dataset Construction (Synthetic Ground Truth)

Since no public dataset exists for this specific attack, we build synthetic labeled data:

```
Clean base videos  +  Synthetic ad frames
        │                     │
        └──────────┬──────────┘
                   ▼
    scripts/inject_frames.py
    • Inject 9–30 frame bursts at random positions (≥3s apart)
    • Save injected .mp4 + annotation JSON with exact frame ranges
    • Also save clean copy (negative samples)
                   │
                   ▼
    scripts/build_dataset.py
    • Run full frame extraction + anomaly detection on each video
    • True injection windows → positive contact sheets
    • Non-injection anomaly candidates → hard-negative contact sheets
    • Split 70% train / 15% val / 15% test
    • Output: data/dataset/{train,val,test}/annotations.jsonl
```

Ad frames are generated programmatically (PIL) in four archetypes:
- **Logo banner** — gradient background + brand name + CTA button
- **QR code** — fake QR grid + URL text
- **Text promo** — promotional copy with brand name
- **Product placeholder** — product box + price tag

---

## Fine-Tuning Qwen2.5-VL

Model: **Qwen2.5-VL-7B-Instruct** with QLoRA (4-bit, r=16)

```
Training task: Given contact sheet + subtitles → predict structured JSON
Loss: only on assistant response tokens (system + user turns masked)
Effective batch: 8 (1 GPU × 8 gradient accumulation steps)
Hardware: ≥24GB VRAM recommended (RTX 4090 / A100)
```

Training config: `configs/training_config.yaml`

---

## Project Structure

```
VLM/
├── video_input/              # Drop any .mp4/.mov here for auto-processing
├── reports/                  # HTML + JSON detection reports (output)
├── models/                   # Fine-tuned Qwen checkpoints
├── data/
│   ├── raw_videos/           # Clean base clips for synthetic dataset
│   ├── ad_frames/            # Generated promo images
│   ├── synthetic/
│   │   ├── videos/           # Injected videos
│   │   └── annotations/      # Ground truth JSON per video
│   └── dataset/
│       ├── train/annotations.jsonl
│       ├── val/annotations.jsonl
│       └── test/annotations.jsonl
├── src/
│   ├── frame_extractor.py    # Full-FPS frame extraction
│   ├── anomaly_detector.py   # pHash A–X–A temporal detection
│   ├── contact_sheet.py      # Contact sheet builder
│   ├── subtitle_extractor.py # SRT / Whisper subtitle loader
│   ├── qwen_classifier.py    # Qwen2.5-VL inference wrapper
│   ├── score_combiner.py     # Score fusion
│   ├── report_generator.py   # HTML + JSON report generation
│   └── pipeline.py           # Full orchestrator
├── scripts/
│   ├── generate_ad_frames.py # Synthetic promo image generator
│   ├── inject_frames.py      # Synthetic video dataset creator
│   ├── build_dataset.py      # JSONL dataset builder for fine-tuning
│   └── fine_tune_qwen.py     # QLoRA fine-tuning script
├── configs/
│   ├── pipeline_config.yaml  # Detection thresholds, model paths
│   └── training_config.yaml  # LoRA + training hyperparameters
├── main.py                   # CLI entry point
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
# Flash attention (optional but recommended):
pip install flash-attn --no-build-isolation
```

### 2. Run detection on a video (zero-shot, base model)

```bash
python main.py --video path/to/your_video.mp4
```

### 3. Watch mode (auto-process any video dropped into ./video_input)

```bash
python main.py --watch
```

### 4. Build synthetic dataset

```bash
# Step A: generate synthetic ad frames
python scripts/generate_ad_frames.py --count 200

# Step B: get some clean base videos (example: download from YouTube)
# yt-dlp -o "data/raw_videos/%(title)s.%(ext)s" <URL>

# Step C: inject ad frames into base videos
python scripts/inject_frames.py --injections_per_video 3

# Step D: build fine-tuning dataset
python scripts/build_dataset.py
```

### 5. Fine-tune

```bash
python scripts/fine_tune_qwen.py --config configs/training_config.yaml
```

### 6. Run with fine-tuned model

```bash
python main.py --video path/to/video.mp4 --use_finetuned
```

---

## Configuration

### Detection thresholds (`configs/pipeline_config.yaml`)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `phash_threshold` | 25 | Min pHash distance to flag a frame as anomalous |
| `scene_cut_threshold` | 80 | Distance above which = hard scene cut (excluded) |
| `context_window` | 15 | Frames before/after used in A–X–A comparison |
| `min_segment_frames` | 5 | ~167ms minimum segment at 30fps |
| `merge_gap_frames` | 10 | Merge segments within this gap |
| `decision_threshold` | 0.55 | Final score to flag as ad insertion |
| `temporal_weight` | 0.35 | Weight of temporal anomaly score |
| `vlm_weight` | 0.65 | Weight of VLM confidence |

### Key design choice: 300ms minimum

At 30fps, 300ms = 9 frames. `min_segment_frames: 5` (167ms) is intentionally slightly lower to catch edge cases and borderline attacks, relying on the VLM to filter false positives.

---

## Output Report

Each processed video produces an HTML report with:
- **Video metadata** (fps, duration, resolution)
- **Summary stats** (candidates, flagged, cleared, scene cuts)
- **Anomaly timeline plot** (pHash distances over time, with flagged segments highlighted)
- **Per-segment cards** showing:
  - Contact sheet (before / suspicious / after)
  - Temporal score, VLM confidence, final score
  - Detected elements (e.g., "logo", "QR code")
  - Model reasoning
  - Subtitle context (before / during / after)

---

## Subtitle Integration

Subtitles provide powerful context for the VLM:
- If a video says "...and now a word from our sponsor..." right before an anomalous segment — high confidence it's an insertion
- If the subtitle context is continuous cooking instructions and the anomaly frames show a QR code — definitive signal

Priority order:
1. `<video_name>.srt` in same directory
2. `<video_name>.en.srt`
3. Whisper auto-transcription (fallback, requires ffmpeg)
