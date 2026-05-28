"""
Merge a trained LoRA adapter into the base Qwen2.5-VL model weights.

Run this ONCE after fine-tuning to produce a standalone merged model
that can be loaded with a plain `from_pretrained` call — no PEFT overhead
at inference time.

Usage:
  python scripts/merge_adapter.py \
    --adapter  models/qwen_ad_detector \
    --output   models/qwen_ad_detector_merged \
    [--push_to_hub your-hf-username/qwen-ad-detector]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def merge(adapter_path: str, output_path: str, push_to_hub: str | None = None):
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    print(f"Loading adapter config from: {adapter_path}")
    # Read base model name from adapter config
    import json
    adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
    with open(adapter_config_path) as f:
        adapter_cfg = json.load(f)
    base_model_name = adapter_cfg["base_model_name_or_path"]
    print(f"Base model: {base_model_name}")

    print("Loading base model in bf16...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("Merging weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)

    print("Saving processor...")
    processor = AutoProcessor.from_pretrained(adapter_path)
    processor.save_pretrained(output_path)

    print(f"Merged model saved. Load it with:\n"
          f"  model = Qwen2_5_VLForConditionalGeneration.from_pretrained('{output_path}')")

    if push_to_hub:
        print(f"Pushing to HuggingFace Hub: {push_to_hub}")
        model.push_to_hub(push_to_hub)
        processor.push_to_hub(push_to_hub)
        print("Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter",  required=True, help="Path to LoRA adapter (output of fine_tune_qwen.py)")
    parser.add_argument("--output",   required=True, help="Where to save the merged model")
    parser.add_argument("--push_to_hub", default=None, help="HuggingFace repo to push to (optional)")
    args = parser.parse_args()
    merge(args.adapter, args.output, args.push_to_hub)


if __name__ == "__main__":
    main()
