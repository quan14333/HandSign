"""Compatibility entry point for the structured sign evaluator.

Use ``python evaluate.py --target-label <label>`` for new integrations.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from sign_eval.evaluator import DEFAULT_CALIBRATION_PATH, SignEvaluator


def score_user(
    user_sequence_path: str | Path,
    label_dir: str | Path,
    calibration_path: str | Path = DEFAULT_CALIBRATION_PATH,
) -> tuple[float | None, list[dict[str, Any]], list[dict[str, Any]]]:
    """Compatibility wrapper returning score, nearest references, feedback."""

    label_dir = Path(label_dir)
    evaluator = SignEvaluator(calibration_path, reference_root=label_dir.parent)
    result = evaluator.evaluate(user_sequence_path, label_dir.name).to_dict()
    return result["form"]["score"], result.get("nearest_references", []), result["feedback"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a learner sequence against one label.")
    parser.add_argument("sequence")
    parser.add_argument("label_dir")
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION_PATH))
    args = parser.parse_args()
    score, nearest, feedback = score_user(args.sequence, args.label_dir, args.calibration)
    print(f"Form score: {score}")
    print("Nearest references:")
    for item in nearest:
        print(f"- {item['file']}: {item['distance']}")
    print("Feedback:")
    for item in feedback:
        print(f"- {item['message']}")


if __name__ == "__main__":
    main()
