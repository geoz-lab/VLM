"""
Step 9: Combine temporal anomaly score + VLM confidence into a final decision.

Final score = temporal_weight * temporal_score + vlm_weight * vlm_confidence
Decision:    final_score >= threshold → "ad_insertion"
"""

from dataclasses import dataclass

from src.anomaly_detector import AnomalySegment
from src.qwen_classifier import ClassifierOutput


@dataclass
class DetectionResult:
    segment: AnomalySegment
    classifier_output: ClassifierOutput
    sheet_path: str

    # Scores
    temporal_score: float
    vlm_confidence: float
    final_score: float

    # Decision
    is_flagged: bool
    label: str                      # "ad_insertion" | "normal"
    detected_elements: list[str]
    reason: str

    # Context
    subtitle_before: str
    subtitle_during: str
    subtitle_after: str

    @property
    def start_ms(self) -> float:
        return self.segment.start_ms

    @property
    def end_ms(self) -> float:
        return self.segment.end_ms

    @property
    def duration_ms(self) -> float:
        return self.segment.duration_ms

    @property
    def start_frame(self) -> int:
        return self.segment.start_frame

    @property
    def end_frame(self) -> int:
        return self.segment.end_frame


def combine_scores(
    segment: AnomalySegment,
    classifier_output: ClassifierOutput,
    sheet_path: str,
    subtitle_before: str = "",
    subtitle_during: str = "",
    subtitle_after: str = "",
    temporal_weight: float = 0.35,
    vlm_weight: float = 0.65,
    decision_threshold: float = 0.55,
) -> DetectionResult:
    """Fuse temporal and VLM scores into a single detection result."""
    # If VLM says normal, flip confidence (use "not-ad" probability as VLM score)
    vlm_score = (
        classifier_output.confidence
        if classifier_output.is_ad
        else (1.0 - classifier_output.confidence)
    )

    final_score = temporal_weight * segment.temporal_score + vlm_weight * vlm_score
    is_flagged = final_score >= decision_threshold

    return DetectionResult(
        segment=segment,
        classifier_output=classifier_output,
        sheet_path=sheet_path,
        temporal_score=segment.temporal_score,
        vlm_confidence=vlm_score,
        final_score=final_score,
        is_flagged=is_flagged,
        label="ad_insertion" if is_flagged else "normal",
        detected_elements=classifier_output.detected_elements,
        reason=classifier_output.reason,
        subtitle_before=subtitle_before,
        subtitle_during=subtitle_during,
        subtitle_after=subtitle_after,
    )


def combine_all(
    sheet_items: list[dict],
    classifier_outputs: list[ClassifierOutput],
    temporal_weight: float = 0.35,
    vlm_weight: float = 0.65,
    decision_threshold: float = 0.55,
) -> list[DetectionResult]:
    """
    Combine scores for all segments.

    sheet_items: output from contact_sheet.build_all_contact_sheets()
    classifier_outputs: output from QwenVLClassifier.classify_batch()
    """
    results = []
    for item, output in zip(sheet_items, classifier_outputs):
        result = combine_scores(
            segment=item["segment"],
            classifier_output=output,
            sheet_path=item["sheet_path"],
            subtitle_before=item.get("subtitle_before", ""),
            subtitle_during=item.get("subtitle_during", ""),
            subtitle_after=item.get("subtitle_after", ""),
            temporal_weight=temporal_weight,
            vlm_weight=vlm_weight,
            decision_threshold=decision_threshold,
        )
        results.append(result)
    return results
