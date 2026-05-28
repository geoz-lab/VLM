# Running on Sherlock (Stanford HPC)

This guide covers the full workflow for training and running the VLM ad detector on Stanford's Sherlock cluster.

## Step 1 — Navigate to the project

```bash
cd /home/groups/kovscek/gmzhang/VLM-main
ls
```

## Step 2 — Set up the environment

```bash
conda create -n vlm python=3.10 -y
conda activate vlm
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## Step 3 — Request an interactive GPU session

```bash
# H100 (preferred for fine-tuning)
srun --ntasks=1 -G 1 --mem-per-cpu=64g --time=6:00:00 --partition=serc --constraint="GPU_SKU:H100_SXM5" --pty bash

# A100 (alternative)
srun --ntasks=1 -G 1 --mem-per-cpu=64g --time=6:00:00 --partition=serc --constraint="GPU_SKU:A100_SXM4" --pty bash

# Verify GPU
nvidia-smi

# Activate env after node allocation
conda activate vlm
cd /home/groups/kovscek/gmzhang/VLM-main
```

## Step 4 — Build the synthetic dataset

Run these scripts in order:

```bash
# 1. Generate synthetic ad frames (logos, QR codes, banners, product shots)
python scripts/generate_ad_frames.py --count 200 --width 640 --height 360

# 2. Generate base videos
python scripts/create_base_videos.py --duration 20

# 3. Inject ad bursts + write ground truth JSON
python scripts/inject_frames.py --injections_per_video 3

# 4. Split into train/val/test JSONL
python scripts/build_dataset.py
```

## Step 5 — Fine-tune on H100

```bash
python scripts/fine_tune_qwen.py --gpu_tier h100
```

Uses LoRA r=128, bf16, flash attention. Best checkpoint (by F1) saves to `models/qwen_ad_detector/`.

## Step 6 — Run detection

```bash
# Drop a test video into video_input/
cp data/synthetic/videos/<any_injected_video>.mp4 video_input/

# Run detection with fine-tuned model
python main.py --video video_input/<video>.mp4 --use_finetuned
```
