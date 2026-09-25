import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from sign_eval.paths import PROJECT_ROOT, REFERENCE_ARCHIVE, SAMPLE_VIDEO_ROOT


class FreshCheckoutTests(unittest.TestCase):
    def test_relocated_checkout_finds_bundled_sample_by_relative_path(self):
        source_video = next(SAMPLE_VIDEO_ROOT.glob("*.mp4"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "HandSign copy"
            shutil.copytree(PROJECT_ROOT / "sign_eval", root / "sign_eval",
                            ignore=shutil.ignore_patterns("__pycache__"))
            sample_root = root / "sample_videos"
            sample_root.mkdir(parents=True)
            shutil.copy2(source_video, sample_root / source_video.name)
            env = os.environ.copy()
            env["PYTHONPATH"] = str(root)
            result = subprocess.run([
                sys.executable, "-X", "utf8", "-B", "-c",
                "from sign_eval.samples import open_sample_video; "
                f"video = open_sample_video({source_video.stem!r}); "
                "assert video is not None; video.close(); print('sample OK')",
            ], cwd=directory, env=env, capture_output=True, text=True,
                encoding="utf-8", timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("sample OK", result.stdout)

    def test_relocated_checkout_loads_bundled_references_without_dataset_or_videos(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "HandSign copy"
            root.mkdir()
            shutil.copytree(PROJECT_ROOT / "sign_eval", root / "sign_eval",
                            ignore=shutil.ignore_patterns("__pycache__"))
            (root / "artifacts").mkdir()
            shutil.copy2(PROJECT_ROOT / "artifacts/label_calibration.json", root / "artifacts")
            shutil.copy2(REFERENCE_ARCHIVE, root / "artifacts")
            env = os.environ.copy()
            env["PYTHONPATH"] = str(root)
            result = subprocess.run([
                sys.executable, "-X", "utf8", "-B", "-c",
                "import numpy as np; from sign_eval.evaluator import SignEvaluator; "
                "from sign_eval.samples import open_sample_video; "
                "e = SignEvaluator(); label = e.available_labels()[0]; "
                "refs = e._load_references(label); "
                "assert len(refs) == e.calibration['labels'][label]['reference_count']; "
                "assert refs[0][1].landmarks.shape[1:] == (48, 2); "
                "assert open_sample_video(label) is None; "
                "np.save('attempt.npy', refs[0][1].landmarks); "
                "result = e.evaluate('attempt.npy', label, {'predicted_label': label}); "
                "assert result.score is not None; print('relocated OK')",
            ], cwd=directory, env=env, capture_output=True, text=True, encoding="utf-8", timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("relocated OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
