# VLM Ad Insertion Detector

Detects unauthorized inserted frames in videos — short promotional bursts (≥300ms at 30fps) that creators inject to show ads, QR codes, or sponsor content that bypasses platform moderation.

---

## Problem

At 30 fps, a 300 ms insertion = **9 consecutive frames**. These are:

- Invisible during normal playback
- Visible when paused — a common vector for smuggling unapproved promotional content
- Detectable by their **block of high consecutive pHash distances** followed by a return to normal content

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
    │  • Consecutive distance: consec[i] = pHash(frame[i], frame[i-1])
    │  • Injection = block of N consecutive high-distance frames (5–90)
    │  • Scene cut = single spike (< min_segment_frames) → excluded
    │
    ▼
[Step 5] Segment Merging
    │  src/anomaly_detector.py
    │  • Merge spike blocks within merge_gap_frames
    │  • Filter: 5–90 frames (167ms – 3s at 30fps)
    │  → AnomalyResult: list of AnomalySegments with temporal_score
    │
    ▼
[Step 6] Contact Sheet Generation
    │  src/contact_sheet.py
    │  • 3-row grid: [BEFORE context] [SUSPICIOUS frames] [AFTER context]
    │  • Green border = normal, Red border = suspicious
    │  • Subtitle text strip (before / during / after)
    │
    ▼
[Steps 7–8] Qwen2.5-VL-7B-Instruct Classification
    │  src/qwen_classifier.py
    │  • Input: contact sheet image + subtitle context + temporal score
    │  • Output: { label, confidence, detected_elements, reason }
    │  • Auto-detects base / LoRA adapter / merged model
    │
    ▼
[Step 9] Score Fusion
    │  src/score_combiner.py
    │  final_score = 0.35 × temporal_score + 0.65 × vlm_confidence
    │  is_flagged  = final_score ≥ 0.55
    │
    ▼
[Step 10] Report Generation
       src/report_generator.py
       → reports/<video>/<video>_report.html  (visual, embedded contact sheets + timeline)
       → reports/<video>/<video>_report.json  (machine-readable)
```

---

## Project Structure

```text
VLM/
├── video_input/              # Drop any video here for auto-detection
├── reports/                  # HTML + JSON output reports
├── models/                   # Fine-tuned LoRA adapters or merged checkpoints
├── data/
│   ├── raw_videos/           # Clean base clips (generate with scripts)
│   ├── ad_frames/            # Synthetic promo images (generate with scripts)
│   ├── synthetic/
│   │   ├── videos/           # Injected videos
│   │   └── annotations/      # Ground truth JSON per video
│   └── dataset/
│       ├── train/annotations.jsonl
│       ├── val/annotations.jsonl
│       └── test/annotations.jsonl
├── src/
│   ├── frame_extractor.py
│   ├── anomaly_detector.py
│   ├── contact_sheet.py
│   ├── subtitle_extractor.py
│   ├── qwen_classifier.py
│   ├── score_combiner.py
│   ├── report_generator.py
│   └── pipeline.py
├── scripts/
│   ├── create_base_videos.py     # Generate 5 diverse synthetic base videos
│   ├── create_test_video.py      # Generate minimal test video
│   ├── generate_ad_frames.py     # Synthetic promo image generator (4 archetypes)
│   ├── inject_frames.py          # Inject ad bursts + write ground truth JSON
│   ├── build_dataset.py          # Build JSONL fine-tuning dataset
│   ├── fine_tune_qwen.py         # QLoRA / LoRA fine-tuning (A100 / H100)
│   └── merge_adapter.py          # Merge LoRA adapter into base model
├── configs/
│   ├── pipeline_config.yaml      # Detection thresholds, model paths
│   ├── training_config.yaml      # LoRA hyperparameters
│   └── deepspeed_zero3.json      # DeepSpeed ZeRO-3 config for multi-GPU
├── test_run.py                   # End-to-end test with mocked VLM
├── main.py                       # CLI entry point
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt

# Flash Attention (recommended for A100/H100):
pip install flash-attn --no-build-isolation
```

### 2. Run detection on a video (base model, zero-shot)

```bash
python main.py --video path/to/video.mp4
```

### 3. Watch mode — auto-process videos dropped into ./video_input

```bash
python main.py --watch
```

---

## Building the Synthetic Dataset

```bash
# Step 1: Generate synthetic ad frames (logos, QR codes, banners, product images)
python scripts/generate_ad_frames.py --count 200 --width 640 --height 360

# Step 2: Generate diverse base videos (or bring your own .mp4 files)
python scripts/create_base_videos.py --duration 20

# Step 3: Inject ad bursts and create ground truth annotations
python scripts/inject_frames.py --injections_per_video 3

# Step 4: Build fine-tuning JSONL dataset (70% train / 15% val / 15% test)
python scripts/build_dataset.py
```

Output: `data/dataset/{train,val,test}/annotations.jsonl`

Each JSONL record is a chat-format sample:

```json
{
  "messages": [
    {"role": "system", "content": "...moderation assistant prompt..."},
    {"role": "user",   "content": [{"type": "image", "image": "path/to/sheet.jpg"}, {"type": "text", "text": "..."}]},
    {"role": "assistant", "content": "{\"label\": \"ad_insertion\", \"confidence\": 0.95, ...}"}
  ],
  "images": ["path/to/contact_sheet.jpg"]
}
```

---

## Fine-Tuning on A100 / H100

### GPU tier selection

| Flag | GPU | Config | Notes |
| --- | --- | --- | --- |
| `--gpu_tier a100` | A100 80GB / H100 | bf16, LoRA r=64, batch=8 | Recommended |
| `--gpu_tier h100` | H100 80GB | bf16, LoRA r=128, batch=8 | Max quality |
| `--gpu_tier a100_4bit` | A100 40GB or multi-GPU | QLoRA 4-bit, LoRA r=16 | Memory-constrained |

### Single GPU

```bash
python scripts/fine_tune_qwen.py --gpu_tier a100
```

### Multi-GPU (DDP, same node)

```bash
torchrun --nproc_per_node=2 scripts/fine_tune_qwen.py --gpu_tier a100
```

### Multi-GPU (DeepSpeed ZeRO-3)

```bash
deepspeed --num_gpus=4 scripts/fine_tune_qwen.py \
  --gpu_tier a100 \
  --deepspeed configs/deepspeed_zero3.json
```

### Resume from checkpoint

```bash
python scripts/fine_tune_qwen.py \
  --gpu_tier a100 \
  --resume_from_checkpoint models/qwen_ad_detector/checkpoint-200
```

Training metrics tracked: **precision**, **recall**, **F1** on the val set.  
Best checkpoint (highest F1) is saved automatically.

The adapter is saved to `models/qwen_ad_detector/` with:

- `adapter_config.json` — LoRA config
- `adapter_model.safetensors` — trained weights
- `training_info.json` — tier + base model used

---

## Merging the LoRA Adapter (for clean deployment)

After fine-tuning, merge the adapter into the base model weights for
faster inference (no PEFT overhead, single `from_pretrained` call):

```bash
python scripts/merge_adapter.py \
  --adapter models/qwen_ad_detector \
  --output  models/qwen_ad_detector_merged
```

Optional: push directly to HuggingFace Hub:

```bash
python scripts/merge_adapter.py \
  --adapter models/qwen_ad_detector \
  --output  models/qwen_ad_detector_merged \
  --push_to_hub your-username/qwen-ad-detector
```

---

## Running Inference with Fine-Tuned Weights

### Option A — Use the LoRA adapter (loads base + adapter at runtime)

```yaml
# configs/pipeline_config.yaml
classifier:
  finetuned_path: "./models/qwen_ad_detector"
  use_finetuned: true
  use_4bit: true       # set false on A100/H100
```

```bash
python main.py --video path/to/video.mp4 --use_finetuned
```

### Option B — Use the merged model (faster, no PEFT dependency)

```yaml
# configs/pipeline_config.yaml
classifier:
  finetuned_path: "./models/qwen_ad_detector_merged"
  use_finetuned: true
  use_4bit: false      # merged model loaded in bf16
```

```bash
python main.py --video path/to/video.mp4 --use_finetuned
```

The classifier auto-detects which mode to use based on whether
`adapter_config.json` is present in the path:

- **Adapter dir** → loads base model + applies LoRA weights via PEFT
- **Merged dir** → loads directly as a standalone model

---

## Configuration Reference

### Detection thresholds (`configs/pipeline_config.yaml`)

| Parameter | Default | Effect |
| --- | --- | --- |
| `phash_threshold` | 25 | Min consecutive pHash distance to flag a frame |
| `scene_cut_threshold` | 80 | A-X-A distance above which = hard scene cut |
| `min_segment_frames` | 5 | ~167ms minimum at 30fps |
| `max_segment_frames` | 90 | ~3s maximum |
| `merge_gap_frames` | 10 | Merge spike blocks within this gap |
| `decision_threshold` | 0.55 | Final fused score to flag as ad insertion |
| `temporal_weight` | 0.35 | Weight of temporal anomaly score |
| `vlm_weight` | 0.65 | Weight of VLM classifier confidence |

### Training (`configs/training_config.yaml`)

Key fields — override per GPU tier via `--gpu_tier` flag:

| Parameter | a100 | h100 | a100_4bit |
| --- | --- | --- | --- |
| LoRA r | 64 | 128 | 16 |
| batch (effective) | 8 | 8 | 8 |
| quantization | none | none | 4-bit NF4 |
| flash attention | yes | yes | yes |

---

## Output Report

Each processed video produces:

- **HTML report** — embedded contact sheets, timeline plot, per-segment cards with:
  - Before / suspicious / after frame rows
  - Temporal score + VLM confidence + final score
  - Detected elements (e.g. "logo", "QR code", "price tag")
  - Model reasoning
  - Subtitle context

- **JSON report** — machine-readable detections with all scores and metadata

---

## Subtitle Integration

Subtitles give the VLM strong contextual signal:

1. `<video_name>.srt` in same directory as video
2. `<video_name>.en.srt` fallback
3. Whisper auto-transcription fallback (`pip install openai-whisper`)
