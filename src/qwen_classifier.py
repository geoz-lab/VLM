"""
Steps 7–8: Qwen2.5-VL classifier.

Accepts a contact sheet image + subtitle context and returns a structured
prediction: label, confidence, detected_elements, reason.

Supports both base (zero-shot) and fine-tuned model checkpoints.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


@dataclass
class ClassifierOutput:
    label: str                          # "ad_insertion" | "normal"
    confidence: float                   # 0.0 – 1.0
    detected_elements: list[str]        # e.g. ["logo", "QR code", "promo text"]
    reason: str
    raw_response: str                   # Full model output string

    @property
    def is_ad(self) -> bool:
        return self.label == "ad_insertion"


SYSTEM_PROMPT = """You are a video content moderation assistant. Your task is to detect
unauthorized inserted frames in videos — short sequences injected by creators to show
promotional content, advertisements, or sponsored material that violates platform policies.

You are shown a contact sheet with three rows of frames:
- TOP ROW (green border):    frames BEFORE the suspicious segment — normal content context
- MIDDLE ROW (red border):   the SUSPICIOUS frames to analyze
- BOTTOM ROW (green border): frames AFTER the suspicious segment — normal content context

Subtitle text from each section is shown at the bottom of the image.

Respond ONLY with valid JSON, no markdown, no explanation outside JSON."""


def _build_user_prompt(
    subtitle_before: str,
    subtitle_during: str,
    subtitle_after: str,
    temporal_score: float,
    duration_ms: float,
    num_frames: int,
) -> str:
    return f"""Analyze the contact sheet for inserted promotional frames.

Subtitle context:
- Before: {subtitle_before or '(none)'}
- During: {subtitle_during or '(none)'}
- After:  {subtitle_after or '(none)'}

Temporal anomaly score: {temporal_score:.3f}
Segment duration: {duration_ms:.0f}ms ({num_frames} frames)

Respond with JSON:
{{
  "label": "ad_insertion" or "normal",
  "confidence": <float 0.0-1.0>,
  "detected_elements": [<list of strings, e.g. "logo", "QR code", "promo text", "product image">],
  "reason": "<one concise sentence>"
}}"""


def _parse_response(raw: str) -> ClassifierOutput:
    """Parse JSON from model output, with fallback for partial/malformed output."""
    # Strip markdown code fences if present
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
        return ClassifierOutput(
            label=data.get("label", "normal"),
            confidence=float(data.get("confidence", 0.5)),
            detected_elements=data.get("detected_elements", []),
            reason=data.get("reason", ""),
            raw_response=raw,
        )
    except json.JSONDecodeError:
        # Try to extract JSON object with regex
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                return ClassifierOutput(
                    label=data.get("label", "normal"),
                    confidence=float(data.get("confidence", 0.5)),
                    detected_elements=data.get("detected_elements", []),
                    reason=data.get("reason", "parse_fallback"),
                    raw_response=raw,
                )
            except Exception:
                pass

        # Complete fallback: treat as uncertain
        return ClassifierOutput(
            label="normal",
            confidence=0.5,
            detected_elements=[],
            reason="Failed to parse model response",
            raw_response=raw,
        )


class QwenVLClassifier:
    """
    Wrapper around Qwen2.5-VL-7B-Instruct for ad insertion classification.

    Lazy-loads model on first call to avoid slow import at startup.
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        use_4bit: bool = True,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
    ):
        self.model_path = model_path
        self.use_4bit = use_4bit
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return

        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
        from qwen_vl_utils import process_vision_info

        print(f"[QwenVLClassifier] Loading model from: {self.model_path}")

        quantization_config = None
        if self.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16 if not self.use_4bit else None,
            quantization_config=quantization_config,
            device_map="auto",
        )
        self._model.eval()

        self._processor = AutoProcessor.from_pretrained(
            self.model_path,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
        )
        self._process_vision_info = process_vision_info
        print("[QwenVLClassifier] Model loaded.")

    def classify(
        self,
        sheet_path: str,
        subtitle_before: str = "",
        subtitle_during: str = "",
        subtitle_after: str = "",
        temporal_score: float = 0.0,
        duration_ms: float = 0.0,
        num_frames: int = 0,
    ) -> ClassifierOutput:
        """Run inference on one contact sheet."""
        self._load()

        user_prompt = _build_user_prompt(
            subtitle_before, subtitle_during, subtitle_after,
            temporal_score, duration_ms, num_frames,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sheet_path},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
            )

        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        raw = self._processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return _parse_response(raw)

    def classify_batch(self, items: list[dict]) -> list[ClassifierOutput]:
        """
        Classify multiple contact sheets.
        Each item: {sheet_path, subtitle_before, subtitle_during, subtitle_after,
                    temporal_score, duration_ms, num_frames}
        """
        return [self.classify(**item) for item in items]
