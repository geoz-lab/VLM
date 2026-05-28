"""
Fine-tune Qwen2.5-VL-7B-Instruct on the ad-insertion detection dataset.

Uses QLoRA (4-bit) + PEFT LoRA via the TRL SFTTrainer.

Run:
  python scripts/fine_tune_qwen.py \
    --config configs/training_config.yaml \
    [--resume_from_checkpoint models/qwen_ad_detector/checkpoint-200]
"""

import argparse
import json
import os

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def preprocess_sample(sample: dict, processor, max_seq_length: int):
    """
    Convert a JSONL record into model inputs.
    The record's messages follow OpenAI-style chat format with image content blocks.
    """
    from qwen_vl_utils import process_vision_info

    messages = sample["messages"]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
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
    # Mask loss on everything except the assistant response
    input_ids = inputs["input_ids"].squeeze(0)
    labels = input_ids.clone()

    # Find where the assistant turn starts
    # Apply loss masking: set labels=-100 for system+user tokens
    assistant_token = processor.tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
    seq = input_ids.tolist()
    for start_idx in range(len(seq) - len(assistant_token)):
        if seq[start_idx: start_idx + len(assistant_token)] == assistant_token:
            labels[:start_idx + len(assistant_token)] = -100
            break
    else:
        labels[:] = -100  # fallback: mask all

    return {
        "input_ids": input_ids,
        "attention_mask": inputs["attention_mask"].squeeze(0),
        "pixel_values": inputs.get("pixel_values"),
        "image_grid_thw": inputs.get("image_grid_thw"),
        "labels": labels,
    }


def build_hf_dataset(jsonl_path: str, processor, max_seq_length: int) -> Dataset:
    records = load_jsonl(jsonl_path)
    processed = []
    for rec in records:
        try:
            processed.append(preprocess_sample(rec, processor, max_seq_length))
        except Exception as e:
            print(f"  Skipping sample: {e}")
    return Dataset.from_list(processed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--resume_from_checkpoint", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    mc = cfg["model"]
    lc = cfg["lora"]
    tc = cfg["training"]
    dc = cfg["dataset"]

    # ── Quantization config ────────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=mc["use_4bit"],
        bnb_4bit_compute_dtype=getattr(torch, mc["bnb_4bit_compute_dtype"]),
        bnb_4bit_quant_type=mc["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=mc["use_nested_quant"],
    ) if mc["use_4bit"] else None

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"Loading model: {mc['name']}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        mc["name"],
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    # ── LoRA ───────────────────────────────────────────────────────────────
    target_modules = lc["target_modules"] + lc.get("vision_modules", [])
    lora_config = LoraConfig(
        r=lc["r"],
        lora_alpha=lc["alpha"],
        lora_dropout=lc["dropout"],
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Processor ─────────────────────────────────────────────────────────
    processor = AutoProcessor.from_pretrained(
        mc["name"],
        min_pixels=256 * 28 * 28,
        max_pixels=dc.get("image_max_pixels", 1003520),
    )

    # ── Dataset ───────────────────────────────────────────────────────────
    print("Loading datasets...")
    train_dataset = build_hf_dataset(dc["train_file"], processor, dc["max_seq_length"])
    eval_dataset = build_hf_dataset(dc["val_file"], processor, dc["max_seq_length"])
    print(f"  Train: {len(train_dataset)}  |  Val: {len(eval_dataset)}")

    # ── Trainer ───────────────────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=tc["output_dir"],
        num_train_epochs=tc["num_train_epochs"],
        per_device_train_batch_size=tc["per_device_train_batch_size"],
        gradient_accumulation_steps=tc["gradient_accumulation_steps"],
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
        bf16=tc["bf16"],
        dataloader_num_workers=tc["dataloader_num_workers"],
        remove_unused_columns=tc["remove_unused_columns"],
        report_to=tc["report_to"],
        dataset_text_field="",  # we handle tokenization ourselves
        max_seq_length=dc["max_seq_length"],
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor.tokenizer,
    )

    # ── Train ──────────────────────────────────────────────────────────────
    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # ── Save ──────────────────────────────────────────────────────────────
    print(f"Saving model to {tc['output_dir']}")
    trainer.save_model(tc["output_dir"])
    processor.save_pretrained(tc["output_dir"])
    print("Fine-tuning complete.")


if __name__ == "__main__":
    main()
