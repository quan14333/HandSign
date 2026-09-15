import csv
from pathlib import Path
import tempfile
import unittest

from label_test import label_test_videos


class LabelTestVideosTests(unittest.TestCase):
    def test_writes_video_name_and_predicted_label(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            test_dir = root / "test"
            test_dir.mkdir()
            (test_dir / "b.mp4").touch()
            (test_dir / "a.mp4").touch()
            (test_dir / "ignored.txt").touch()
            output_path = root / "result" / "labels.csv"

            labels = {"a.mp4": "Cảm ơn", "b.mp4": "Cách ly"}

            def predictor(video_path: str | Path) -> dict[str, str]:
                return {"predicted_label": labels[Path(video_path).name]}

            count = label_test_videos(test_dir, output_path, predictor)

            with output_path.open(encoding="utf-8-sig", newline="") as output_file:
                rows = list(csv.DictReader(output_file))

            self.assertEqual(count, 2)
            self.assertEqual(rows, [
                {"video_name": "a.mp4", "label": "Cảm ơn"},
                {"video_name": "b.mp4", "label": "Cách ly"},
            ])

    def test_rejects_a_directory_without_mp4_videos(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaises(FileNotFoundError):
                label_test_videos(temporary_directory, Path(temporary_directory) / "labels.csv")


if __name__ == "__main__":
    unittest.main()
