"""
Risk-based triage: maps a prediction + its uncertainty to one of three
categories, so the system doesn't just output a label — it also tells
the viewer how much to trust that label.

Thresholds are a tunable policy, not a model property, so they live
here as named constants rather than being scattered through the API.
"""

from dataclasses import dataclass
from core.architecture import CLASS_NAMES

# Classes that always warrant at least a "Suspicious" classification
# regardless of uncertainty — malignant/precancerous categories.
HIGH_STAKES_CLASSES = {"mel", "bcc", "akiec"}

LOW_RISK_MAX_UNCERTAINTY = 0.25
SUSPICIOUS_MAX_UNCERTAINTY = 0.45


@dataclass
class TriageResult:
    level: str          # "low_risk" | "suspicious" | "expert_review"
    label: str           # human-readable
    reason: str           # why this level was assigned


def triage(predicted_class: str, uncertainty: float, quality_passed: bool = True) -> TriageResult:
    if predicted_class not in CLASS_NAMES:
        raise ValueError(f"Unknown class '{predicted_class}'")

    if not quality_passed:
        return TriageResult(
            level="expert_review",
            label="Expert Review Recommended",
            reason="Image quality check failed — prediction may be unreliable.",
        )

    if uncertainty > SUSPICIOUS_MAX_UNCERTAINTY:
        return TriageResult(
            level="expert_review",
            label="Expert Review Recommended",
            reason=f"Uncertainty ({uncertainty:.2f}) exceeds the safety threshold.",
        )

    if predicted_class in HIGH_STAKES_CLASSES or uncertainty > LOW_RISK_MAX_UNCERTAINTY:
        return TriageResult(
            level="suspicious",
            label="Suspicious",
            reason=(
                "Predicted class requires clinical confirmation."
                if predicted_class in HIGH_STAKES_CLASSES
                else f"Moderate uncertainty ({uncertainty:.2f})."
            ),
        )

    return TriageResult(
        level="low_risk",
        label="Low Risk",
        reason="Benign prediction with high evidence strength.",
    )
