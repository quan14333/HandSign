"""Predict labels for every test video and save them as a CSV file."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any, Callable

from recognizer import predict_video


DEFAULT_TEST_DIR = Path("data/video/dataset/test")
DEFAULT_OUTPUT = Path("data/video/dataset/test_labels.csv")


def label_test_videos(
    test_dir: str | Path = DEFAULT_TEST_DIR,
    output_path: str | Path = DEFAULT_OUTPUT,
    predictor: Callable[[str | Path], dict[str, Any]] = predict_video,
) -> int:
    """Label all MP4 files in ``test_dir`` and return the number processed."""

    test_dir = Path(test_dir)
    output_path = Path(output_path)
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory does not exist: {test_dir}")

    video_paths = sorted(test_dir.glob("*.mp4"), key=lambda path: path.name)
    if not video_paths:
        raise FileNotFoundError(f"No MP4 videos found in: {test_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["video_name", "label"])
        writer.writeheader()

        total = len(video_paths)
        for number, video_path in enumerate(video_paths, start=1):
            result = predictor(video_path)
            label = str(result["predicted_label"])
            writer.writerow({"video_name": video_path.name, "label": label})
            output_file.flush()
            print(f"[{number}/{total}] {video_path.name} -> {label}")

    return len(video_paths)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Predict labels for test videos and write video_name,label CSV rows."
    )
    parser.add_argument("--test-dir", default=str(DEFAULT_TEST_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    count = label_test_videos(args.test_dir, args.output)
    print(f"Saved {count} predictions to {args.output}")


if __name__ == "__main__":
    main()
