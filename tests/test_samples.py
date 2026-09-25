from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from recognizer import _load_video
from sign_eval.desktop import suggest_labels
from sign_eval.samples import open_sample_video, sample_candidates
from sign_eval.paths import SAMPLE_VIDEO_ROOT


class SuggestionsTests(unittest.TestCase):
    def test_prefix_search_with_case_and_vietnamese_accents(self):
        labels = ["An ủi", "Ban ngày", "Bế mạc", "Biết", "Cá biển", "Đúng"]
        self.assertEqual(suggest_labels(labels, "B"), ["Ban ngày", "Bế mạc", "Biết"])
        self.assertEqual(suggest_labels(labels, "Bế"), ["Bế mạc"])
        self.assertEqual(suggest_labels(labels, "be"), ["Bế mạc"])
        self.assertEqual(suggest_labels(labels, "du"), ["Đúng"])
        self.assertEqual(suggest_labels(labels, "xyz"), [])
        self.assertEqual(suggest_labels(labels, ""), labels)


class SampleVideoTests(unittest.TestCase):
    def test_bundled_library_has_one_video_for_each_label(self):
        videos = sorted(SAMPLE_VIDEO_ROOT.glob("*.mp4"))
        self.assertEqual(len(videos), 100)
        self.assertEqual(len({path.stem for path in videos}), 100)

    def test_missing_or_unreadable_video_directory_is_optional(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertIsNone(open_sample_video("Bế mạc", Path(folder) / "missing"))
        with patch.object(Path, "iterdir", side_effect=PermissionError):
            self.assertEqual(sample_candidates("Bế mạc", Path("blocked")), [])

    def test_only_label_matched_videos_are_candidates(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "Bế mạc").mkdir()
            good = root / "Bế mạc" / "123.mp4"
            good.touch()
            (root / "Bế mạc" / "notes.txt").touch()
            (root / "Other.mp4").touch()
            self.assertEqual(sample_candidates("bế mạc", root), [good])

    def test_bad_first_video_falls_back_to_playable_candidate(self):
        broken, valid = Mock(), Mock()
        broken.isOpened.return_value = True
        broken.read.return_value = (False, None)
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        valid.isOpened.return_value = True
        valid.read.return_value = (True, frame)
        valid.get.return_value = float("nan")
        with patch("sign_eval.samples.sample_candidates", return_value=[Path("bad.mp4"), Path("good.mp4")]), \
                patch("sign_eval.samples.cv2.VideoCapture", side_effect=[broken, valid]):
            video = open_sample_video("Bế mạc")
        broken.release.assert_called_once()
        self.assertIs(video.first_frame, frame)
        self.assertEqual(video.fps, 25)
        valid.release.assert_not_called()
        video.close()
        valid.release.assert_called_once()

    def test_decoder_exception_becomes_no_video(self):
        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.side_effect = cv2.error("corrupt")
        with patch("sign_eval.samples.sample_candidates", return_value=[Path("bad.mp4")]), \
                patch("sign_eval.samples.cv2.VideoCapture", return_value=capture):
            self.assertIsNone(open_sample_video("Bế mạc"))
        capture.release.assert_called_once()


class PortableRecognizerVideoTests(unittest.TestCase):
    def test_rgb_sampling_repeats_frames_for_short_videos(self):
        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.return_value = 3
        frames = [np.full((2, 2, 3), (index, 10, 200), dtype=np.uint8) for index in range(3)]
        capture.read.side_effect = [(True, frame) for frame in frames]
        with patch("cv2.VideoCapture", return_value=capture):
            result = _load_video("test.mp4", 4)
        self.assertEqual(len(result), 4)
        self.assertEqual([frame[0, 0].tolist() for frame in result],
                         [[200, 10, 0], [200, 10, 0], [200, 10, 1], [200, 10, 2]])
        capture.release.assert_called_once()

    def test_failed_decode_is_an_error_and_releases_capture(self):
        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.return_value = 10
        capture.read.return_value = (False, None)
        with patch("cv2.VideoCapture", return_value=capture), self.assertRaises(ValueError):
            _load_video("bad.mp4", 16)
        capture.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
