"""Evaluate one learner recording and write a structured JSON result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from sign_eval.calibration import build_calibration
from sign_eval.evaluator import DEFAULT_CALIBRATION_PATH, SignEvaluator
from sign_eval.paths import USER_ROOT


def evaluate_recording(
    target_label: str,
    sequence_path: str | Path,
    video_path: str | Path | None = None,
    output_path: str | Path = USER_ROOT / "evaluation.json",
    *,
    rebuild_calibration: bool = False,
) -> dict[str, Any]:
    """Shared CLI/desktop pipeline; return this run's result, never a cached JSON."""

    calibration_path = DEFAULT_CALIBRATION_PATH
    if rebuild_calibration or not calibration_path.is_file():
        build_calibration(output_path=calibration_path)
    recognition = None
    if video_path is not None:
        from recognizer import predict_video

        recognition = predict_video(video_path)
    evaluator = SignEvaluator(calibration_path)
    result = evaluator.evaluate(sequence_path, target_label, recognition).to_dict()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, ensure_ascii=False, indent=2, allow_nan=False)
    return result


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Evaluate a sign-language practice recording.")
    parser.add_argument("--target-label", required=True, help="The sign the learner was asked to perform.")
    parser.add_argument("--sequence", help=".npy or .npz landmark sequence to evaluate.")
    parser.add_argument(
        "--video",
        help="Video from the same recording as --sequence. Required to receive a score; "
             "without it, the result is unscored with a recognition reminder.",
    )
    parser.add_argument("--output", default=str(USER_ROOT / "evaluation.json"))
    parser.add_argument("--rebuild-calibration", action="store_true")
    args = parser.parse_args()

    default_npz = USER_ROOT / "sequence_user/record.npz"
    default_npy = USER_ROOT / "sequence_user/sequence.npy"
    sequence_path = Path(args.sequence) if args.sequence else (
        default_npz if default_npz.is_file() else default_npy
    )
    result = evaluate_recording(
        args.target_label, sequence_path, args.video, args.output,
        rebuild_calibration=args.rebuild_calibration,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved evaluation to {args.output}.")


if __name__ == "__main__":
    main()
