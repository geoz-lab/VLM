# VLM Ad Insertion Detector

Detects unauthorized inserted frames in videos — short promotional bursts (≥300 ms at 30 fps) that creators inject to show ads, QR codes, or sponsor content that bypasses platform moderation.

---

## Table of Contents

1. [Problem](#problem)
2. [Detection Pipeline](#detection-pipeline)
3. [Project Structure](#project-structure)
4. [Environment Setup](#environment-setup)
5. [Sherlock (Stanford HPC) Setup](#sherlock-stanford-hpc-setup)
6. [Creating the Dataset](#creating-the-dataset)
7. [Fine-Tuning](#fine-tuning)
8. [Saving and Merging Weights](#saving-and-merging-weights)
9. [Running Inference with Fine-Tuned Weights](#running-inference-with-fine-tuned-weights)
10. [Running a Test (No GPU Required)](#running-a-test-no-gpu-required)
11. [Running Detection on Real Videos](#running-detection-on-real-videos)
12. [Configuration Reference](#configuration-reference)
13. [Output Report Format](#output-report-format)
14. [Subtitle Integration](#subtitle-integration)

---

## Problem

At 30 fps, a 300 ms insertion = **9 consecutive frames**. These are:

- Invisible during normal playback
- Visible when paused — a common vector for smuggling unapproved promotional content
- Detectable by their **block of high consecutive pHash distances** followed by a return to normal content

---

## Detection Pipeline

```text
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
├── video_input/              # Drop any video here for auto-detection (watch mode)
├── reports/                  # HTML + JSON output reports (auto-created)
├── models/                   # Fine-tuned LoRA adapters or merged checkpoints
│   ├── qwen_ad_detector/         # LoRA adapter output (from fine_tune_qwen.py)
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   └── training_info.json
│   └── qwen_ad_detector_merged/  # Merged weights (from merge_adapter.py)
├── data/
│   ├── raw_videos/               # Clean base clips (generate with scripts)
│   ├── ad_frames/                # Synthetic promo images (generate with scripts)
│   ├── synthetic/
│   │   ├── videos/               # Injected videos
│   │   └── annotations/          # Ground truth JSON per video
│   └── dataset/
│       ├── train/annotations.jsonl
│       ├── val/annotations.jsonl
│       └── test/annotations.jsonl
├── src/
│   ├── frame_extractor.py        # Step 1: extract frames at full FPS
│   ├── anomaly_detector.py       # Steps 2–5: pHash + segment detection
│   ├── contact_sheet.py          # Step 6: 3-row visual context grid
│   ├── subtitle_extractor.py     # Load SRT or Whisper transcription
│   ├── qwen_classifier.py        # Steps 7–8: Qwen2.5-VL inference
│   ├── score_combiner.py         # Step 9: fuse temporal + VLM scores
│   ├── report_generator.py       # Step 10: HTML + JSON reports
│   └── pipeline.py               # Orchestrates all steps
├── scripts/
│   ├── create_base_videos.py     # Generate 5 diverse synthetic base videos
│   ├── create_test_video.py      # Generate a minimal test video
│   ├── generate_ad_frames.py     # Synthetic promo image generator (4 archetypes)
│   ├── inject_frames.py          # Inject ad bursts + write ground truth JSON
│   ├── build_dataset.py          # Build JSONL fine-tuning dataset
│   ├── fine_tune_qwen.py         # QLoRA / LoRA fine-tuning (A100 / H100)
│   └── merge_adapter.py          # Merge LoRA adapter into base model
├── configs/
│   ├── pipeline_config.yaml      # Detection thresholds, model paths
│   ├── training_config.yaml      # LoRA hyperparameters, dataset paths
│   └── deepspeed_zero3.json      # DeepSpeed ZeRO-3 config for multi-GPU
├── test_run.py                   # End-to-end test with mocked VLM (no GPU needed)
├── main.py                       # CLI entry point
└── requirements.txt
```

---

## Environment Setup

### Local machine

```bash
# 1. Clone and enter repo
git clone <repo-url> && cd VLM

# 2. Create environment (Python 3.10+)
conda create -n vlm python=3.10 -y
conda activate vlm

# 3. Install dependencies
pip install -r requirements.txt

# 4. Flash Attention — significant speed-up on A100/H100 (optional on other GPUs)
pip install flash-attn --no-build-isolation
```

> **Note:** `flash-attn` requires CUDA 11.8+ and a matching PyTorch build. If install fails, the code automatically falls back to `eager` attention.

---

## Sherlock (Stanford HPC) Setup

Repo lives at `/home/groups/kovscek/gmzhang/VLM/VLM-main`.
Conda env lives at `/home/groups/kovscek/gmzhang/miniconda3/envs/vlm`.

### Every new login session

```bash
ssh gmzhang@login.sherlock.stanford.edu
tmux new -s vlm   # or: tmux attach -t vlm

cd /home/groups/kovscek/gmzhang/VLM/VLM-main
source /home/groups/kovscek/gmzhang/miniconda3/etc/profile.d/conda.sh
conda activate vlm
export PATH="/home/groups/kovscek/gmzhang/miniconda3/envs/vlm/bin:$PATH"
```

### Request an interactive A100 GPU

Run this from the login node:

```bash
srun --ntasks=1 -G 1 --mem-per-cpu=64g --time=6:00:00 \
  --partition=serc --constraint="GPU_SKU:A100_SXM4" --pty bash
```

After landing on the compute node, re-activate the environment and load CUDA:

```bash
source /home/groups/kovscek/gmzhang/miniconda3/etc/profile.d/conda.sh
conda activate vlm
export PATH="/home/groups/kovscek/gmzhang/miniconda3/envs/vlm/bin:$PATH"
module load cuda/12.1.1
export CUDA_HOME=$CUDA_DIR
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/groups/kovscek/gmzhang/VLM/VLM-main
```

> These env vars suppress tokenizer warnings and enable PyTorch's memory-efficient allocator, which matters on A100s with large batches.

---

## Creating the Dataset

The dataset pipeline has four sequential steps. Run them in order.

### Step 1 — Generate synthetic ad frames

Creates diverse promotional images (logos, QR codes, banners, product placeholders) used as injection material.

```bash
python scripts/generate_ad_frames.py \
  --output data/ad_frames \
  --count 200 \
  --width 1280 \
  --height 720
```

Output: `data/ad_frames/ad_0000.jpg` … `ad_0199.jpg`

Four frame archetypes are generated in rotation:

| Archetype | Description |
| --- | --- |
| Logo banner | Brand name + gradient background + CTA button |
| QR code | Fake QR grid pattern + URL text |
| Text promo | "Use code SAVE20 for 20% off!" style |
| Product placeholder | Product box + price tag + brand name |

### Step 2 — Generate synthetic base videos

Creates clean 20-second videos with diverse content categories (news, sports, tutorial, nature, talking head).

```bash
python scripts/create_base_videos.py --duration 20
```

Output: `data/raw_videos/*.mp4`

Or bring your own `.mp4` files — place them in `data/raw_videos/`.

### Step 3 — Inject ad bursts and write ground truth

Randomly injects 3 bursts per video (9–30 frames each) and records exact frame positions.

```bash
python scripts/inject_frames.py \
  --videos data/raw_videos \
  --ad_frames data/ad_frames \
  --output data/synthetic \
  --injections_per_video 3 \
  --min_burst 9 \
  --max_burst 30
```

Outputs:

- `data/synthetic/videos/<stem>_injected.mp4` — video with injected frames
- `data/synthetic/videos/<stem>_clean.mp4` — copy of original (negative samples)
- `data/synthetic/annotations/<stem>_injected.json` — ground truth with exact frame/ms positions

Example annotation file:

```json
{
  "video_path": "data/synthetic/videos/tutorial_injected.mp4",
  "fps": 30.0,
  "label": "injected",
  "injections": [
    {
      "output_start_frame": 312,
      "output_end_frame": 327,
      "output_start_ms": 10400.0,
      "output_end_ms": 10933.0,
      "duration_ms": 533.0,
      "num_frames": 16
    }
  ]
}
```

### Step 4 — Build the JSONL fine-tuning dataset

Runs anomaly detection on every annotated video, generates contact sheets, and writes chat-format JSONL records split 70% train / 15% val / 15% test.

```bash
python scripts/build_dataset.py \
  --annotations data/synthetic/annotations \
  --output data/dataset \
  --train_ratio 0.70 \
  --val_ratio 0.15 \
  --seed 42
```

Output:

- `data/dataset/train/annotations.jsonl`
- `data/dataset/val/annotations.jsonl`
- `data/dataset/test/annotations.jsonl`

Each JSONL line is one training sample in chat format:

```json
{
  "messages": [
    {"role": "system", "content": "...moderation assistant prompt..."},
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "data/dataset/sheets/tutorial_injected/pos_0000.jpg"},
        {"type": "text",  "text": "Analyze the contact sheet...\nTemporal anomaly score: 0.813\n..."}
      ]
    },
    {
      "role": "assistant",
      "content": "{\"label\": \"ad_insertion\", \"confidence\": 0.95, \"detected_elements\": [\"promotional content\"], \"reason\": \"...\"}"
    }
  ],
  "images": ["data/dataset/sheets/tutorial_injected/pos_0000.jpg"]
}
```

The dataset has two sample types:

- **Positive** (`label: ad_insertion`) — true injection windows from ground truth
- **Negative** (`label: normal`) — detected anomalies that are NOT real injections (scene transitions, etc.)

---

## Fine-Tuning

Fine-tuning uses QLoRA or LoRA on Qwen2.5-VL-7B-Instruct. The GPU tier flag sets all memory-sensitive hyperparameters automatically.

### GPU tier selection

| `--gpu_tier` | Target GPU | LoRA r | LoRA alpha | Batch/GPU | Grad accum | Effective batch | Quantization |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `a100` | A100 80GB or H100 80GB | 32 | 128 | 2 | 4 | 8 | none (bf16) |
| `h100` | H100 80GB | 128 | 256 | 4 | 2 | 8 | none (bf16) |
| `a100_4bit` | A100 40GB or multi-GPU | 16 | 32 | 1 | 8 | 8 | QLoRA 4-bit NF4 |

**Which tier to pick:**

- **A100 80GB** → `--gpu_tier a100` — good quality, reasonable training time
- **H100 80GB** → `--gpu_tier h100` — highest LoRA rank, fastest throughput
- **A100 40GB or memory-constrained** → `--gpu_tier a100_4bit` — 4-bit quantization reduces VRAM by ~50%, slight quality tradeoff
- **Multiple GPUs** → use `a100` or `h100` with DeepSpeed; each GPU sees 2–4 samples per step

### Single GPU

```bash
python scripts/fine_tune_qwen.py --gpu_tier a100
```

### Multi-GPU (DDP, single node)

```bash
torchrun --nproc_per_node=2 scripts/fine_tune_qwen.py --gpu_tier a100
```

### Multi-GPU (DeepSpeed ZeRO-3)

```bash
deepspeed --num_gpus=4 scripts/fine_tune_qwen.py \
  --gpu_tier a100 \
  --deepspeed configs/deepspeed_zero3.json
```

DeepSpeed ZeRO-3 shards optimizer state, gradients, and parameters across all GPUs — use this when the model doesn't fit on a single card even in 4-bit mode.

### Resume from checkpoint

```bash
python scripts/fine_tune_qwen.py \
  --gpu_tier a100 \
  --resume_from_checkpoint models/qwen_ad_detector/checkpoint-200
```

### Where to change hyperparameters

**GPU-tier hyperparameters** (LoRA rank, batch size, quantization) — edit `GPU_TIERS` in `scripts/fine_tune_qwen.py:36–67`:

```python
GPU_TIERS = {
    "a100": dict(
        use_4bit=False,
        lora_r=32,          # ← LoRA rank (higher = more capacity, more VRAM)
        lora_alpha=128,     # ← typically 4× lora_r
        per_device_batch=2, # ← samples per GPU per step
        gradient_accumulation=4,   # effective batch = per_device_batch × this
        flash_attn=True,
        dtype=torch.bfloat16,
    ),
    ...
}
```

**All other hyperparameters** — edit `configs/training_config.yaml`:

```yaml
lora:
  r: 16                    # default (overridden by GPU tier)
  alpha: 32                # default (overridden by GPU tier)
  dropout: 0.05            # ← regularization; increase to 0.1 if overfitting
  target_modules:          # ← which layers to train (all attention + MLP)
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"

training:
  num_train_epochs: 5      # ← total passes over training data
  learning_rate: 2.0e-4    # ← typical LoRA LR; halve if loss is unstable
  lr_scheduler_type: "cosine"  # ← cosine decay; try "linear" for shorter runs
  warmup_ratio: 0.1        # ← 10% of steps for LR warmup
  weight_decay: 0.01
  max_grad_norm: 1.0       # ← gradient clipping
  logging_steps: 10        # ← print loss every N steps
  save_steps: 100          # ← checkpoint every N steps
  eval_steps: 100          # ← run val metrics every N steps
  save_total_limit: 3      # ← keep only the 3 most recent checkpoints
  report_to: "none"        # ← set "wandb" for Weights & Biases tracking

dataset:
  max_seq_length: 1536     # ← max tokens per sample (longer = more VRAM)
  image_max_pixels: 401408 # ← ~1M pixels max per image tile (~1024×392)
```

Training metrics tracked: **precision**, **recall**, **F1** on the val set. The checkpoint with the highest F1 is loaded at the end automatically.

### Expected training time (approximate)

| GPU | Tier | ~100 samples | ~1000 samples |
| --- | --- | --- | --- |
| A100 80GB | a100 | 5–10 min | 45–90 min |
| H100 80GB | h100 | 3–6 min | 25–50 min |
| A100 40GB | a100_4bit | 8–15 min | 70–120 min |

---

## Saving and Merging Weights

### What gets saved after fine-tuning

Fine-tuning saves a **LoRA adapter** (not the full model) to `models/qwen_ad_detector/`:

```text
models/qwen_ad_detector/
├── adapter_config.json          # LoRA config + base model name
├── adapter_model.safetensors    # Trained delta weights (~100–500 MB)
├── preprocessor_config.json     # Processor / tokenizer config
├── tokenizer.json
├── tokenizer_config.json
└── training_info.json           # GPU tier, LoRA r, base model used
```

At inference time, the base model (~15 GB) is loaded from HuggingFace Hub and the adapter is applied on top.

### Merging the adapter (recommended for production)

Merging bakes the adapter weights into the base model, producing a standalone model with no PEFT dependency and faster inference:

```bash
python scripts/merge_adapter.py \
  --adapter models/qwen_ad_detector \
  --output  models/qwen_ad_detector_merged
```

The merged model is saved to `models/qwen_ad_detector_merged/` (~15 GB safetensors shards) and can be loaded with a plain `from_pretrained` call.

### Push merged model to HuggingFace Hub (optional)

```bash
python scripts/merge_adapter.py \
  --adapter models/qwen_ad_detector \
  --output  models/qwen_ad_detector_merged \
  --push_to_hub your-hf-username/qwen-ad-detector
```

---

## Running Inference with Fine-Tuned Weights

### Step 1 — Point the pipeline config at your weights

Edit `configs/pipeline_config.yaml`:

```yaml
classifier:
  model_path: "Qwen/Qwen2.5-VL-7B-Instruct"     # base model (HF Hub)
  finetuned_path: "./models/qwen_ad_detector"     # ← your adapter path
  use_finetuned: true                              # ← set to true
  use_4bit: false     # false on A100/H100 bf16; true for 4-bit inference
```

**Option A — LoRA adapter** (loads base model + applies adapter at runtime):

```yaml
finetuned_path: "./models/qwen_ad_detector"    # contains adapter_config.json
use_finetuned: true
use_4bit: true   # optional: reduces VRAM usage
```

**Option B — Merged model** (faster, single load, no PEFT):

```yaml
finetuned_path: "./models/qwen_ad_detector_merged"   # contains config.json, no adapter_config.json
use_finetuned: true
use_4bit: false   # merged model is bf16; 4-bit has no effect here
```

The classifier (`src/qwen_classifier.py`) auto-detects which mode to use:

- If `adapter_config.json` is present → loads base + adapter via PEFT
- If only `config.json` is present → loads as a standalone merged model
- Otherwise → loads the HF Hub base model (zero-shot)

### Step 2 — Run detection

```bash
# Single video with fine-tuned weights
python main.py --video path/to/video.mp4 --use_finetuned

# Batch process a directory
python main.py --input_dir video_input --use_finetuned

# Watch mode (auto-process videos dropped into ./video_input)
python main.py --watch --use_finetuned
```

The `--use_finetuned` flag sets `config["classifier"]["use_finetuned"] = True` at runtime. Alternatively, set `use_finetuned: true` directly in the YAML and omit the flag.

---

## Running a Test (No GPU Required)

`test_run.py` exercises the complete pipeline (steps 1–10) with a **mocked VLM classifier**. No GPU or model download needed — useful for verifying the pipeline logic end-to-end.

### Prerequisites

```bash
# Create a minimal test video with one injected segment
python scripts/create_test_video.py
```

This writes:

- `video_input/test_scene_injected.mp4`
- `data/synthetic/annotations/test_scene_injected.json`

### Run the test

```bash
python test_run.py
```

### What it does

1. Loads `video_input/test_scene_injected.mp4` and the ground truth annotation
2. Runs steps 1–6 (frame extraction → contact sheets) with real data
3. Replaces steps 7–8 (Qwen inference) with a deterministic mock:
   - Segments with `temporal_score > 0.15` → `ad_insertion`
   - Subtitle keywords (`sponsor`, `promo`, `discount`, …) → `ad_insertion`
   - Otherwise → `normal`
4. Runs steps 9–10 (score fusion + report generation)
5. Compares detections against ground truth and prints precision / recall / F1

### Expected output

```
╭─────────────────────────────────────────────╮
│  VLM Ad Insertion Detector — Test Run        │
│  VLM step mocked (no GPU required)           │
╰─────────────────────────────────────────────╯

Ground truth injections:
  #1  frame 150–165  (5.00s–5.50s)  500ms
  #2  frame 420–440  (14.00s–14.67s)  667ms

┌─────────────────────────────┬────────────┐
│ Metric                      │ Value      │
├─────────────────────────────┼────────────┤
│ Ground truth injections     │ 2          │
│ Detected (all candidates)   │ 3          │
│ Flagged                     │ 2          │
│ True Positives              │ 2          │
│ False Positives             │ 0          │
│ False Negatives (missed)    │ 0          │
│ Precision                   │ 1.00       │
│ Recall                      │ 1.00       │
│ F1 Score                    │ 1.00       │
└─────────────────────────────┴────────────┘

Output reports:
  [html] reports/test_scene_injected/test_scene_injected_report.html
  [json] reports/test_scene_injected/test_scene_injected_report.json
```

---

## Demo Detection

Run the fine-tuned model on one of the synthetic injected videos generated during dataset creation:

```bash
mkdir -p video_input
cp data/synthetic/videos/nature_doc_injected.mp4 video_input/
python main.py --video video_input/nature_doc_injected.mp4 --use_finetuned
```

Result — 3/3 ad segments correctly detected:

| Segment | Time | Duration | Final Score |
| --- | --- | --- | --- |
| 1 | 5.93s – 6.40s | 467ms | 0.967 ✅ |
| 2 | 12.33s – 13.33s | 1000ms | 0.844 ✅ |
| 3 | 16.37s – 17.17s | 800ms | 0.967 ✅ |

Reports saved to `reports/nature_doc_injected/`.

---

## Running Detection on Real Videos

### Single video

```bash
python main.py --video my_video.mp4
```

### Batch process a directory

```bash
python main.py --input_dir video_input
```

### Watch mode (auto-process new files)

```bash
python main.py --watch
```

Polls `./video_input/` every 5 seconds. Processed videos are marked with a `.processed` file so they are not re-analyzed.

### Keep extracted frames on disk

```bash
python main.py --video my_video.mp4 --keep_frames
```

By default, extracted frames are deleted after the pipeline finishes to save disk space.

### All CLI flags

```text
python main.py [--video PATH] [--input_dir DIR] [--watch]
               [--config YAML] [--keep_frames] [--use_finetuned]

--video PATH          Analyze a single video file
--input_dir DIR       Batch-process all videos in a directory (default: video_input)
--watch               Watch video_input/ and auto-process new files
--config YAML         Pipeline config file (default: configs/pipeline_config.yaml)
--keep_frames         Keep extracted JPEG frames after processing
--use_finetuned       Use fine-tuned model (reads finetuned_path from config)
```

---

## Configuration Reference

### Detection thresholds — `configs/pipeline_config.yaml`

```yaml
anomaly_detection:
  phash_threshold: 25        # Hamming distance above which a frame is "anomalous" (0–256)
                             # Lower → more sensitive, more false positives
                             # Higher → less sensitive, fewer false positives
  scene_cut_threshold: 80    # Distance above which = hard scene cut (not an insertion)
  min_segment_frames: 5      # ~167ms minimum at 30fps — shorter bursts are ignored
  max_segment_frames: 90     # ~3s maximum — longer anomalies are treated as scene changes
  merge_gap_frames: 10       # Merge two detected blocks if gap < this many frames
  context_window: 15         # Frames before/after used for A-X-A pattern comparison

scoring:
  temporal_weight: 0.35      # How much the temporal anomaly score contributes
  vlm_weight: 0.65           # How much the VLM confidence contributes
  decision_threshold: 0.55   # Final fused score ≥ this → flagged as ad insertion
```

**Tuning tips:**

- Reduce `phash_threshold` (e.g. 20) to catch subtler insertions at the cost of more false positives
- Raise `decision_threshold` (e.g. 0.65) to require higher confidence before flagging
- Adjust `temporal_weight` / `vlm_weight` to trust temporal signal more or less than the VLM

### Model paths — `configs/pipeline_config.yaml`

```yaml
classifier:
  model_path: "Qwen/Qwen2.5-VL-7B-Instruct"  # HF Hub ID for base model
  finetuned_path: "./models/qwen_ad_detector"  # Local adapter or merged model
  use_finetuned: false   # Set true to use fine-tuned weights
  max_new_tokens: 256    # Max tokens generated per classification
  temperature: 0.1       # Low temp = more deterministic structured output
  use_4bit: false        # QLoRA 4-bit inference (saves ~8GB VRAM)
  batch_size: 1          # Number of contact sheets per inference call
```

### Training — `configs/training_config.yaml`

Key fields (see [Fine-Tuning](#fine-tuning) for the full table):

| Parameter | Default | Notes |
| --- | --- | --- |
| `lora.r` | 16 | Overridden by `--gpu_tier` |
| `lora.alpha` | 32 | Overridden by `--gpu_tier` |
| `lora.dropout` | 0.05 | Regularization |
| `training.num_train_epochs` | 5 | Total dataset passes |
| `training.learning_rate` | 2e-4 | LoRA default; halve if unstable |
| `training.lr_scheduler_type` | cosine | `linear` also works |
| `training.warmup_ratio` | 0.1 | 10% warmup |
| `training.save_steps` | 100 | Checkpoint frequency |
| `training.eval_steps` | 100 | Val evaluation frequency |
| `training.report_to` | none | Set `wandb` for experiment tracking |
| `dataset.max_seq_length` | 1536 | Truncation length in tokens |
| `dataset.image_max_pixels` | 401408 | Max pixels per image tile (~1MP) |

### DeepSpeed ZeRO-3 — `configs/deepspeed_zero3.json`

Used automatically when `--deepspeed configs/deepspeed_zero3.json` is passed. Shards parameters, gradients, and optimizer state across all GPUs. Gradient accumulation and batch sizes are controlled by the `TrainingArguments` and set to `"auto"`.

---

## Output Report Format

Each processed video produces two files in `reports/<video_name>/`:

### HTML report (`<video>_report.html`)

A self-contained visual report with embedded images (no external dependencies). Contents:

- **Header** — video name, duration, FPS, resolution, model used, timestamp
- **Summary stats** — total candidates, flagged count, cleared count, scene cuts detected
- **Anomaly timeline** — matplotlib plot of frame-to-frame pHash distances across the entire video, with red/green shading for flagged/cleared segments and orange tick marks for scene cuts
- **Per-segment cards** — one card per candidate segment, each showing:
  - Timestamp range and duration
  - Score pills: `Temporal: 0.812 | VLM: 0.940 | Final: 0.896`
  - Detected elements (e.g. `logo`, `QR code`, `promo text`)
  - Model reasoning sentence
  - Contact sheet image (3-row grid: before / suspicious / after)
  - Subtitle text for each section

Flagged segments have a red left border and `AD INSERTION DETECTED` label. Cleared segments have a green left border and `Normal / Cleared` label.

### JSON report (`<video>_report.json`)

Machine-readable output for downstream processing:

```json
{
  "video": {
    "path": "video_input/my_video.mp4",
    "fps": 29.97,
    "duration_ms": 120000.0,
    "width": 1920,
    "height": 1080
  },
  "summary": {
    "total_candidates": 3,
    "flagged": 2,
    "cleared": 1,
    "scene_cuts": 4,
    "model": "./models/qwen_ad_detector",
    "generated_at": "2026-05-28 14:23:01"
  },
  "detections": [
    {
      "idx": 0,
      "start_frame": 312,
      "end_frame": 328,
      "start_ms": 10410.0,
      "end_ms": 10946.0,
      "duration_ms": 536.0,
      "temporal_score": 0.812,
      "vlm_confidence": 0.940,
      "final_score": 0.896,
      "is_flagged": true,
      "label": "ad_insertion",
      "detected_elements": ["logo", "promotional text"],
      "reason": "Frame content is visually inconsistent with surrounding video — contains promotional elements.",
      "subtitle_during": "(none)",
      "sheet_path": "/tmp/vlm_my_video_abc123/sheets/segment_000.jpg"
    }
  ]
}
```

---

## Subtitle Integration

Subtitles give the VLM strong contextual signal — promotional keywords in the subtitle during a suspicious segment raise confidence significantly.

The pipeline tries three subtitle sources in order:

1. `<video_name>.srt` — in the same directory as the video
2. `<video_name>.en.srt` — English subtitle fallback
3. Whisper auto-transcription — if no SRT is found and `openai-whisper` is installed

To install Whisper:

```bash
pip install openai-whisper
```

To provide a subtitle file manually:

```bash
python main.py --video my_video.mp4
# Place my_video.srt alongside my_video.mp4 — it is picked up automatically
```
