"""Evaluate one learner recording and write a structured JSON result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sign_eval.calibration import build_calibration
from sign_eval.evaluator import DEFAULT_CALIBRATION_PATH, SignEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a sign-language practice recording.")
    parser.add_argument("--target-label", required=True, help="The sign the learner was asked to perform.")
    parser.add_argument("--sequence", help=".npy or .npz landmark sequence to evaluate.")
    parser.add_argument("--video", help="Optional video for semantic label recognition.")
    parser.add_argument("--output", default="data/user/evaluation.json")
    parser.add_argument("--rebuild-calibration", action="store_true")
    args = parser.parse_args()

    calibration_path = DEFAULT_CALIBRATION_PATH
    if args.rebuild_calibration or not calibration_path.is_file():
        payload = build_calibration(output_path=calibration_path)
        print(f"Built calibration for {payload['summary']['label_count']} labels.")

    default_npz = Path("data/user/sequence_user/record.npz")
    default_npy = Path("data/user/sequence_user/sequence.npy")
    sequence_path = Path(args.sequence) if args.sequence else (
        default_npz if default_npz.is_file() else default_npy
    )
    recognition = None
    if args.video:
        from recognizer import predict_video

        recognition = predict_video(args.video)
    evaluator = SignEvaluator(calibration_path)
    result = evaluator.evaluate(sequence_path, args.target_label, recognition).to_dict()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved evaluation to {output_path}.")


if __name__ == "__main__":
    main()
