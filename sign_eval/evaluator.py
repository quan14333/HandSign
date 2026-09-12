"""High-level semantic, quality, form, and feedback evaluation."""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .landmarks import (
    REGIONS,
    LandmarkSequence,
    calculate_dtw,
    largest_error_segment,
    load_landmark_sequence,
    region_distances_on_path,
    tracking_report,
)


DEFAULT_CALIBRATION_PATH = Path("artifacts/label_calibration.json")
DEFAULT_VALIDATION_PATH = Path("artifacts/user_validation_calibration.json")


def canonical_label(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


@dataclass(frozen=True)
class EvaluationResult:
    """Serializable result returned by :class:`SignEvaluator`."""

    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return self.payload

    @property
    def score(self) -> float | None:
        return self.payload.get("form", {}).get("score")


class SignEvaluator:
    """Evaluate a learner recording against one requested sign label."""

    def __init__(
        self,
        calibration_path: str | Path = DEFAULT_CALIBRATION_PATH,
        reference_root: str | Path | None = None,
        validation_path: str | Path = DEFAULT_VALIDATION_PATH,
    ) -> None:
        calibration_path = Path(calibration_path)
        if not calibration_path.is_file():
            raise FileNotFoundError(
                f"Calibration file not found: {calibration_path}. "
                "Run build_calibration.py first."
            )
        with calibration_path.open("r", encoding="utf-8") as calibration_file:
            self.calibration = json.load(calibration_file)
        if self.calibration.get("format_version") != 2:
            raise ValueError("Unsupported calibration format. Rebuild calibration data.")
        configured_root = Path(self.calibration["reference_root"])
        self.reference_root = Path(reference_root) if reference_root else configured_root
        self._label_lookup = {
            canonical_label(label): label for label in self.calibration["labels"]
        }
        validation_path = Path(validation_path)
        if validation_path.is_file():
            with validation_path.open("r", encoding="utf-8") as validation_file:
                validation = json.load(validation_file)
            self.validation_labels = validation.get("labels", {})
        else:
            self.validation_labels = {}
        self._validation_lookup = {
            canonical_label(label): details for label, details in self.validation_labels.items()
        }
        self._region_threshold_cache: dict[str, dict[str, Any]] = {}

    def available_labels(self) -> list[str]:
        return sorted(self.calibration["labels"])

    def _resolve_label(self, label: str) -> str:
        resolved = self._label_lookup.get(canonical_label(label))
        if resolved is None:
            raise ValueError(
                f"Unknown label '{label}'. Use one of the calibrated reference labels."
            )
        return resolved

    def _load_references(self, label: str) -> list[tuple[str, LandmarkSequence]]:
        details = self.calibration["labels"][label]
        label_dir = self.reference_root / label
        references: list[tuple[str, LandmarkSequence]] = []
        for filename in details["reference_files"]:
            path = label_dir / filename
            if not path.is_file():
                raise FileNotFoundError(
                    f"Reference '{path}' is missing. Rebuild the calibration file."
                )
            references.append((filename, load_landmark_sequence(path)))
        if not references:
            raise ValueError(f"No valid references remain for label '{label}'.")
        return references

    def _region_thresholds(self, label: str) -> dict[str, Any]:
        """Compute label-specific regional limits lazily and cache them."""

        if label in self._region_threshold_cache:
            return self._region_threshold_cache[label]

        details = self.calibration["labels"][label]
        samples = details.get("region_pair_samples", [])
        if len(samples) < 3:
            result = {"thresholds": None, "sample_count": len(samples)}
            self._region_threshold_cache[label] = result
            return result

        region_values = {name: [] for name in REGIONS}
        label_dir = self.reference_root / label
        for file_a, file_b in samples:
            sequence_a = load_landmark_sequence(label_dir / file_a).landmarks
            sequence_b = load_landmark_sequence(label_dir / file_b).landmarks
            _, path = calculate_dtw(sequence_a, sequence_b)
            means, _ = region_distances_on_path(sequence_a, sequence_b, path)
            for region, value in means.items():
                region_values[region].append(value)

        thresholds = {
            region: round(float(np.quantile(values, 0.90)), 6)
            for region, values in region_values.items()
            if values
        }
        result = {"thresholds": thresholds, "sample_count": len(samples)}
        self._region_threshold_cache[label] = result
        return result

    @staticmethod
    def _validation_score(distance: float, validation: dict[str, Any]) -> float:
        """Map DTW distance to a practice score, not a probability of correctness."""

        a = float(validation["correct_median_distance"])
        b = float(validation["threshold_distance"])
        c = float(validation["incorrect_median_distance"])
        if not np.isfinite([a, b, c]).all() or not 0 < a < b < c:
            raise ValueError(
                "Practice scoring requires 0 < correct_median_distance < "
                "threshold_distance < incorrect_median_distance. "
                "Rebuild validation with representative correct/incorrect clips."
            )
        if not np.isfinite(distance) or distance < 0:
            raise ValueError("Comparison distance must be finite and non-negative.")
        if distance <= a:
            score = 100 - 20 * distance / a
        elif distance <= b:
            score = 80 - 20 * (distance - a) / (b - a)
        elif distance <= c:
            score = 60 - 30 * (distance - b) / (c - b)
        else:
            score = 30 * 2 ** (-(distance - c) / (c - b))
        return round(float(score), 2)

    @staticmethod
    def _form_status(
        score: float,
        low_sample: bool,
        validation: dict[str, Any] | None,
    ) -> str:
        if validation is not None:
            if score >= 80:
                return "excellent"
            return "good" if score >= 60 else "needs_practice"
        if low_sample:
            return "estimated_low_sample"
        if score >= 75:
            return "excellent"
        if score >= 45:
            return "good"
        if score >= 20:
            return "needs_practice"
        return "far_from_reference"

    def _validation_for_label(self, label: str) -> dict[str, Any] | None:
        return self._validation_lookup.get(canonical_label(label))

    def _recognition_status(
        self, target_label: str, recognition: dict[str, Any] | None
    ) -> dict[str, Any]:
        if recognition is None:
            return {"status": "not_run"}

        predicted_label = recognition.get("predicted_label")
        confidence = recognition.get("confidence")
        margin = recognition.get("margin")
        result: dict[str, Any] = {
            "status": "uncertain",
            "predicted_label": predicted_label,
            "confidence": round(float(confidence), 4) if confidence is not None else None,
            "margin": round(float(margin), 4) if margin is not None else None,
        }
        if not predicted_label:
            result["reason"] = "Recognizer returned no predicted label."
            return result
        if confidence is not None and float(confidence) < 0.50:
            result["reason"] = "Recognizer confidence is below 0.50."
            return result
        if margin is not None and float(margin) < 0.12:
            result["reason"] = "The two leading label probabilities are too close."
            return result
        result["status"] = (
            "match"
            if canonical_label(predicted_label) == canonical_label(target_label)
            else "mismatch"
        )
        return result

    @staticmethod
    def _feedback_message(region: str, start: int, end: int, frame_count: int) -> str:
        labels = {
            "left_hand": "Tay trái",
            "right_hand": "Tay phải",
            "face": "Khuôn mặt",
        }
        start_percent = round(start / max(frame_count - 1, 1) * 100)
        end_percent = round(end / max(frame_count - 1, 1) * 100)
        return (
            f"{labels[region]} lệch rõ nhất ở khoảng "
            f"{start_percent}-{end_percent}% của động tác."
        )

    def _build_feedback(
        self,
        label: str,
        user: np.ndarray,
        matches: list[dict[str, Any]],
        low_sample: bool,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        threshold_info = self._region_thresholds(label)
        thresholds = threshold_info["thresholds"]
        if thresholds is None:
            return (
                [
                    {
                        "severity": "unknown",
                        "message": "Chưa đủ mẫu tham chiếu để tạo feedback theo vùng đáng tin cậy.",
                    }
                ],
                threshold_info,
            )

        top_matches = matches[: min(3, len(matches))]
        mean_values: dict[str, list[float]] = {region: [] for region in REGIONS}
        best_traces: dict[str, np.ndarray] | None = None
        for match_index, match in enumerate(top_matches):
            means, traces = region_distances_on_path(
                user, match["reference"], match["path"]
            )
            for region, value in means.items():
                mean_values[region].append(value)
            if match_index == 0:
                best_traces = traces

        feedback: list[dict[str, Any]] = []
        for region in REGIONS:
            distance = float(np.median(mean_values[region]))
            threshold = float(thresholds[region])
            ratio = distance / max(threshold, 1e-8)
            item: dict[str, Any] = {
                "region": region,
                "distance": round(distance, 4),
                "threshold": round(threshold, 4),
                "severity": "ok" if ratio <= 1 else "needs_attention",
            }
            if ratio > 1 and best_traces is not None:
                segment = largest_error_segment(best_traces[region], threshold)
                if segment is not None:
                    start, end = segment
                    item["segment"] = {
                        "start_frame": start,
                        "end_frame": end,
                        "start_percent": round(start / max(len(user) - 1, 1) * 100),
                        "end_percent": round(end / max(len(user) - 1, 1) * 100),
                    }
                    item["message"] = self._feedback_message(
                        region, start, end, len(user)
                    )
                else:
                    item["message"] = "Vùng này chưa khớp ổn định với các mẫu gần nhất."
            else:
                item["message"] = "Vùng này đang khớp tốt với các mẫu gần nhất."
            feedback.append(item)

        if low_sample:
            feedback.insert(
                0,
                {
                    "severity": "low_confidence",
                    "message": "Label này có ít hơn 10 mẫu tham chiếu; feedback chỉ nên dùng để tham khảo.",
                },
            )
        return feedback, threshold_info

    def evaluate(
        self,
        sequence_path: str | Path,
        target_label: str,
        recognition: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        label = self._resolve_label(target_label)
        user_sequence = load_landmark_sequence(sequence_path)
        tracking = tracking_report(user_sequence)
        recognition_result = self._recognition_status(label, recognition)
        base_payload: dict[str, Any] = {
            "target_label": label,
            "sequence_path": str(sequence_path),
            "tracking": tracking,
            "recognition": recognition_result,
        }
        if tracking["status"] == "invalid":
            base_payload.update(
                {
                    "status": "not_scored",
                    "form": {"status": "not_scored", "score": None},
                    "feedback": [
                        {
                            "severity": "blocking",
                            "message": "Không thể chấm vì tracker không tạo được landmark hợp lệ.",
                        }
                    ],
                }
            )
            return EvaluationResult(base_payload)
        if tracking["status"] == "low_quality":
            base_payload.update(
                {
                    "status": "not_scored",
                    "form": {"status": "not_scored", "score": None},
                    "feedback": [
                        {
                            "severity": "blocking",
                            "message": "Video thiếu landmark tay ở quá nhiều frame để chấm tin cậy.",
                        }
                    ],
                }
            )
            return EvaluationResult(base_payload)

        details = self.calibration["labels"][label]
        references = self._load_references(label)
        matches: list[dict[str, Any]] = []
        for filename, reference in references:
            distance, path = calculate_dtw(user_sequence.landmarks, reference.landmarks)
            matches.append(
                {
                    "file": filename,
                    "distance": distance,
                    "path": path,
                    "reference": reference.landmarks,
                }
            )
        matches.sort(key=lambda item: item["distance"])
        top_k = min(int(details["top_k"]), len(matches))
        comparison_distance = float(
            np.mean([match["distance"] for match in matches[:top_k]])
        )
        reference_distribution = np.asarray(details["loo_top_k_distances"], dtype=float)
        if len(reference_distribution) == 0:
            raise ValueError(f"No leave-one-out calibration values for '{label}'.")
        percentile_score = float(
            100 * np.mean(reference_distribution >= comparison_distance)
        )
        low_sample = details["quality"] == "low_sample"
        validation = self._validation_for_label(label)
        score = (
            self._validation_score(comparison_distance, validation)
            if validation is not None else round(percentile_score, 2)
        )
        form = {
            "status": self._form_status(score, low_sample, validation),
            "score": score,
            "reference_percentile": round(percentile_score, 2),
            "comparison_distance": round(comparison_distance, 6),
            "reference_count": details["reference_count"],
            "calibration_count": len(reference_distribution),
            "calibration_quality": details["quality"],
            "decision_source": (
                "labeled_user_validation" if validation is not None else "reference_distribution"
            ),
        }
        if validation is not None:
            form["validation"] = validation
        feedback, threshold_info = self._build_feedback(
            label, user_sequence.landmarks, matches, low_sample
        )
        if recognition_result["status"] == "mismatch":
            status = "incorrect_label"
        elif recognition_result["status"] == "match":
            status = (
                "correct"
                if form["status"] in {"excellent", "good"}
                else "right_label_needs_practice"
            )
        else:
            status = "needs_label_confirmation"

        base_payload.update(
            {
                "status": status,
                "form": form,
                "feedback": feedback,
                "feedback_calibration": threshold_info,
                "nearest_references": [
                    {
                        "file": match["file"],
                        "distance": round(float(match["distance"]), 6),
                    }
                    for match in matches[:top_k]
                ],
            }
        )
        return EvaluationResult(base_payload)
