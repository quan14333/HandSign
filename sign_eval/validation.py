"""Fit optional per-label practice thresholds from manually labelled clips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .evaluator import DEFAULT_CALIBRATION_PATH, SignEvaluator


DEFAULT_OUTPUT = Path("artifacts/user_validation_calibration.json")


def _candidate_thresholds(correct: list[float], incorrect: list[float]) -> list[float]:
    values = sorted(set(correct + incorrect))
    if len(values) == 1:
        return values
    candidates = [values[0] - 1e-6, values[-1] + 1e-6]
    candidates.extend((left + right) / 2 for left, right in zip(values, values[1:]))
    return candidates


def _find_best_threshold(correct: list[float], incorrect: list[float]) -> tuple[float, float]:
    """Maximize balanced accuracy where a smaller distance is a better form."""

    best_threshold = 0.0
    best_score = -1.0
    for threshold in _candidate_thresholds(correct, incorrect):
        true_positive_rate = float(np.mean(np.asarray(correct) <= threshold))
        true_negative_rate = float(np.mean(np.asarray(incorrect) > threshold))
        score = (true_positive_rate + true_negative_rate) / 2
        if score > best_score:
            best_threshold, best_score = threshold, score
    return best_threshold, best_score


def build_validation_calibration(
    label: str,
    correct_dir: str | Path,
    incorrect_dir: str | Path,
    output_path: str | Path = DEFAULT_OUTPUT,
    calibration_path: str | Path = DEFAULT_CALIBRATION_PATH,
) -> dict[str, Any]:
    """Create a threshold from clips manually labelled as correct/incorrect."""

    correct_paths = sorted(Path(correct_dir).glob("*.npy"))
    incorrect_paths = sorted(Path(incorrect_dir).glob("*.npy"))
    if not correct_paths or not incorrect_paths:
        raise ValueError("Both correct and incorrect directories need at least one .npy file.")
    evaluator = SignEvaluator(calibration_path)
    correct_distances = [
        evaluator.evaluate(path, label).to_dict()["form"]["comparison_distance"]
        for path in correct_paths
    ]
    incorrect_distances = [
        evaluator.evaluate(path, label).to_dict()["form"]["comparison_distance"]
        for path in incorrect_paths
    ]
    threshold, balanced_accuracy = _find_best_threshold(correct_distances, incorrect_distances)
    output_path = Path(output_path)
    if output_path.is_file():
        with output_path.open("r", encoding="utf-8") as output_file:
            payload = json.load(output_file)
    else:
        payload = {"format_version": 1, "labels": {}}
    payload["labels"][label] = {
        "threshold_distance": round(threshold, 6),
        "correct_count": len(correct_distances),
        "incorrect_count": len(incorrect_distances),
        "correct_median_distance": round(float(np.median(correct_distances)), 6),
        "incorrect_median_distance": round(float(np.median(incorrect_distances)), 6),
        "balanced_accuracy": round(float(balanced_accuracy), 4),
        "quality": "ok" if min(len(correct_paths), len(incorrect_paths)) >= 20 else "low_sample",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a labelled user-practice calibration.")
    parser.add_argument("--label", required=True)
    parser.add_argument("--correct-dir", required=True)
    parser.add_argument("--incorrect-dir", required=True)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION_PATH))
    args = parser.parse_args()
    payload = build_validation_calibration(
        args.label,
        args.correct_dir,
        args.incorrect_dir,
        args.output,
        args.calibration,
    )
    print(json.dumps(payload["labels"][args.label], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
