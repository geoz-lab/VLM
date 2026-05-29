"""
Fine-tune Qwen2.5-VL-7B-Instruct on the ad-insertion detection dataset.

Supports three GPU tiers:
  --gpu_tier a100   → bf16 + LoRA r=64, no quantization (A100/H100 80GB)
  --gpu_tier a100_4bit → QLoRA 4-bit + LoRA r=16  (A100 40GB or multi-GPU)
  --gpu_tier h100   → bf16 + LoRA r=128 + flash_attention_2 (H100 80GB)

Multi-GPU (DDP or DeepSpeed ZeRO-3):
  Single node, 2× A100:
    torchrun --nproc_per_node=2 scripts/fine_tune_qwen.py --gpu_tier a100

  With DeepSpeed ZeRO-3:
    deepspeed --num_gpus=2 scripts/fine_tune_qwen.py \
      --gpu_tier a100 --deepspeed configs/deepspeed_zero3.json

Run:
  python scripts/fine_tune_qwen.py --gpu_tier a100
  python scripts/fine_tune_qwen.py --gpu_tier a100 --resume_from_checkpoint models/qwen_ad_detector/checkpoint-100
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ── GPU tier presets ──────────────────────────────────────────────────────────

GPU_TIERS = {
    "a100": dict(
        use_4bit=False,
        lora_r=32,
        lora_alpha=128,
        per_device_batch=2,
        gradient_accumulation=4,   # effective batch = 8
        flash_attn=True,
        dtype=torch.bfloat16,
        desc="A100 80GB / H100 80GB — bf16, LoRA r=64",
    ),
    "a100_4bit": dict(
        use_4bit=True,
        lora_r=16,
        lora_alpha=32,
        per_device_batch=1,
        gradient_accumulation=8,   # effective batch = 8
        flash_attn=True,
        dtype=torch.bfloat16,
        desc="A100 40GB / multi-GPU — QLoRA 4-bit, LoRA r=16",
    ),
    "h100": dict(
        use_4bit=False,
        lora_r=128,
        lora_alpha=256,
        per_device_batch=4,
        gradient_accumulation=2,   # effective batch = 8
        flash_attn=True,
        dtype=torch.bfloat16,
        desc="H100 80GB — bf16, LoRA r=128",
    ),
}


# ── Dataset utilities ─────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def collate_fn(batch):
    """Collate batch for Qwen2.5-VL training."""
    import torch
    from torch.nn.utils.rnn import pad_sequence

    result = {}
    all_keys = set(k for b in batch for k in b.keys())

    for key in all_keys:
        vals = [b[key] for b in batch if b.get(key) is not None]
        if not vals:
            continue
        # Convert lists back to tensors (HF Dataset stores as lists)
        if isinstance(vals[0], list):
            try:
                vals = [torch.tensor(v) for v in vals]
            except Exception:
                continue
        if not isinstance(vals[0], torch.Tensor):
            continue
        if key in ("input_ids", "attention_mask"):
            pad_val = 0
            result[key] = pad_sequence(vals, batch_first=True, padding_value=pad_val)
        elif key == "labels":
            result[key] = pad_sequence(vals, batch_first=True, padding_value=-100)
        elif key == "pixel_values":
            # pixel_values: cat along batch dim (each sample may have diff num patches)
            result[key] = torch.cat([v.reshape(-1, v.shape[-1]) if v.dim() > 2 else v for v in vals], dim=0)
        elif key == "image_grid_thw":
            result[key] = torch.cat([v.reshape(-1, 3) if v.dim() > 1 else v.unsqueeze(0) for v in vals], dim=0)
        else:
            try:
                result[key] = torch.stack(vals)
            except Exception:
                result[key] = vals[0].unsqueeze(0)
    return result


def build_hf_dataset(jsonl_path: str, processor, max_seq_length: int):
    from datasets import Dataset
    from qwen_vl_utils import process_vision_info

    records = load_jsonl(jsonl_path)
    processed = []

    for rec in records:
        messages = rec["messages"]
        try:
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs if image_inputs else None,
                videos=video_inputs if video_inputs else None,
                padding=True,
                truncation=True,
                max_length=max_seq_length,
                return_tensors="pt",
            )

            input_ids = inputs["input_ids"].squeeze(0)
            labels = input_ids.clone()

            # Mask loss on system + user tokens; only train on assistant response
            assistant_tokens = processor.tokenizer.encode(
                "<|im_start|>assistant", add_special_tokens=False
            )
            seq = input_ids.tolist()
            for start_idx in range(len(seq) - len(assistant_tokens)):
                if seq[start_idx: start_idx + len(assistant_tokens)] == assistant_tokens:
                    labels[:start_idx + len(assistant_tokens)] = -100
                    break
            else:
                labels[:] = -100

            sample = {
                "input_ids": input_ids,
                "attention_mask": inputs["attention_mask"].squeeze(0),
                "labels": labels,
            }
            if inputs.get("pixel_values") is not None:
                sample["pixel_values"] = inputs["pixel_values"].squeeze(0)
            if inputs.get("image_grid_thw") is not None:
                sample["image_grid_thw"] = inputs["image_grid_thw"].squeeze(0)

            processed.append(sample)
        except Exception as e:
            print(f"  Skipping sample (error: {e})")

    return Dataset.from_list(processed)


# ── Evaluation metrics ────────────────────────────────────────────────────────

def compute_metrics(eval_pred, processor):
    """
    Compute precision/recall/F1 on structured JSON predictions.
    Called by Trainer after each eval step.
    """
    predictions, labels = eval_pred

    # Decode predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # Replace -100 in labels (masked tokens)
    labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)

    decoded_preds = processor.tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)

    tp = fp = fn = 0
    for pred_text, label_text in zip(decoded_preds, decoded_labels):
        pred_label = _extract_label(pred_text)
        true_label = _extract_label(label_text)
        if true_label == "ad_insertion":
            if pred_label == "ad_insertion":
                tp += 1
            else:
                fn += 1
        else:
            if pred_label == "ad_insertion":
                fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def _extract_label(text: str) -> str:
    import re, json as _json
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return _json.loads(text).get("label", "normal")
    except Exception:
        match = re.search(r'"label"\s*:\s*"(\w+)"', text)
        return match.group(1) if match else "normal"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--gpu_tier", choices=list(GPU_TIERS), default="a100")
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--deepspeed", default=None, help="Path to DeepSpeed config JSON")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    tier = GPU_TIERS[args.gpu_tier]
    print(f"GPU tier: {args.gpu_tier} — {tier['desc']}")

    tc = cfg["training"]
    dc = cfg["dataset"]

    # ── Quantization ─────────────────────────────────────────────────────────
    bnb_config = None
    if tier["use_4bit"]:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=tier["dtype"],
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    # ── Load model ───────────────────────────────────────────────────────────
    model_name = cfg["model"]["name"]
    print(f"Loading model: {model_name}")

    attn_impl = "eager"
    from transformers import Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=tier["dtype"] if not tier["use_4bit"] else None,
        attn_implementation=attn_impl,
        device_map="auto" if not args.deepspeed else None,
    )

    if tier["use_4bit"]:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)

    # ── LoRA ─────────────────────────────────────────────────────────────────
    from peft import LoraConfig, get_peft_model
    lc = cfg["lora"]
    lora_config = LoraConfig(
        r=tier["lora_r"],
        lora_alpha=tier["lora_alpha"],
        lora_dropout=lc["dropout"],
        target_modules=lc["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Processor ────────────────────────────────────────────────────────────
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(
        model_name,
        min_pixels=256 * 28 * 28,
        max_pixels=dc.get("image_max_pixels", 1003520),
    )

    # ── Datasets ─────────────────────────────────────────────────────────────
    print("Building datasets...")
    train_ds = build_hf_dataset(dc["train_file"], processor, dc["max_seq_length"])
    eval_ds = build_hf_dataset(dc["val_file"], processor, dc["max_seq_length"])
    print(f"  Train: {len(train_ds)}  |  Val: {len(eval_ds)}")

    # ── Training args ────────────────────────────────────────────────────────
    from transformers import TrainingArguments
    output_dir = tc["output_dir"]

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=tc["num_train_epochs"],
        per_device_train_batch_size=tier["per_device_batch"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=tier["gradient_accumulation"],
        learning_rate=tc["learning_rate"],
        lr_scheduler_type=tc["lr_scheduler_type"],
        warmup_ratio=tc["warmup_ratio"],
        weight_decay=tc["weight_decay"],
        max_grad_norm=tc["max_grad_norm"],
        logging_steps=tc["logging_steps"],
        save_steps=tc["save_steps"],
        eval_steps=tc["eval_steps"],
        eval_strategy="steps",
        save_total_limit=tc["save_total_limit"],
        bf16=True,
        fp16=False,
        dataloader_num_workers=tc["dataloader_num_workers"],
        remove_unused_columns=False,
        report_to=tc.get("report_to", "none"),
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        deepspeed=args.deepspeed,
        # Gradient checkpointing saves memory at slight speed cost
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # ── Trainer ──────────────────────────────────────────────────────────────
    from transformers import Trainer
    import functools

    # Fix for _signature_columns bug in transformers
    class PatchedTrainer(Trainer):
        def _prepare_inputs(self, inputs):
            if not inputs:
                # Print dataset sample keys for debugging
                sample = self.train_dataset[0]
                print(f"DEBUG sample keys: {list(sample.keys())}")
                print(f"DEBUG sample types: {[(k, type(v), v.shape if hasattr(v,'shape') else '') for k,v in sample.items()]}")
            if self._signature_columns is None:
                self._signature_columns = ["input_ids", "attention_mask", "labels",
                                           "pixel_values", "image_grid_thw"]
            return super()._prepare_inputs(inputs)

    trainer = PatchedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate_fn,
        compute_metrics=functools.partial(compute_metrics, processor=processor),
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"Starting training (tier={args.gpu_tier})...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # ── Save adapter ──────────────────────────────────────────────────────────
    print(f"Saving LoRA adapter to: {output_dir}")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    # Save tier info for loading
    with open(os.path.join(output_dir, "training_info.json"), "w") as f:
        json.dump({
            "gpu_tier": args.gpu_tier,
            "lora_r": tier["lora_r"],
            "base_model": model_name,
            "use_4bit": tier["use_4bit"],
        }, f, indent=2)

    print(f"\nFine-tuning complete. Adapter saved to: {output_dir}")
    print(f"Next step — merge weights for clean inference:")
    print(f"  python scripts/merge_adapter.py --adapter {output_dir} --output {output_dir}_merged")


if __name__ == "__main__":
    main()
