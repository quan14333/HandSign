import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

from sign_eval.evaluator import SignEvaluator


class ActiveRegionEvaluatorTests(unittest.TestCase):
    def test_one_hand_label_does_not_score_or_require_the_unused_hand(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            label = "One hand"
            label_dir = root / "references" / label
            label_dir.mkdir(parents=True)

            reference = np.zeros((5, 48, 2), dtype=np.float32)
            reference[:, 21:42, 0] = np.arange(21)
            reference[:, 42:48, 0] = np.arange(6)
            filenames = ["a.npy", "b.npy", "c.npy"]
            for filename in filenames:
                np.save(label_dir / filename, reference)

            calibration = {
                "format_version": 2,
                "reference_root": str(root / "references"),
                "labels": {
                    label: {
                        "reference_files": filenames,
                        "reference_count": 3,
                        "top_k": 1,
                        "loo_top_k_distances": [0.0, 0.0, 0.0],
                        "region_pair_samples": [
                            ["a.npy", "b.npy"],
                            ["a.npy", "c.npy"],
                            ["b.npy", "c.npy"],
                        ],
                        "quality": "ok",
                    }
                },
            }
            calibration_path = root / "calibration.json"
            calibration_path.write_text(json.dumps(calibration), encoding="utf-8")

            user = reference.copy()
            user[:, :21, 0] = np.arange(21) + 1000
            validity = np.ones((5, 3), dtype=bool)
            validity[:, 0] = False
            user_path = root / "user.npz"
            np.savez_compressed(user_path, landmarks=user, validity=validity)

            evaluator = SignEvaluator(
                calibration_path,
                validation_path=root / "missing-validation.json",
            )
            result = evaluator.evaluate(
                user_path,
                label,
                {"predicted_label": label, "confidence": 0.9, "margin": 0.5},
            ).to_dict()

            self.assertEqual(result["tracking"]["status"], "ok")
            self.assertEqual(result["form"]["required_regions"], ["right_hand", "face"])
            self.assertEqual(result["form"]["ignored_regions"], ["left_hand"])
            self.assertEqual(result["form"]["comparison_distance"], 0.0)
            self.assertEqual(result["status"], "correct")
            self.assertNotIn("left_hand", [item.get("region") for item in result["feedback"]])
            self.assertEqual(result["form"]["score"], 100)
            self.assertEqual(result["form"]["region_weights"], {"right_hand": 0.8, "face": 0.2})

            # A perfect landmark match cannot override a confident wrong label.
            with patch("sign_eval.evaluator.calculate_dtw") as dtw:
                result = evaluator.evaluate(user_path, label, {
                    "predicted_label": "Different", "confidence": 0.7,
                }).to_dict()
                dtw.assert_not_called()
            self.assertEqual(result["status"], "incorrect_label")
            self.assertIsNone(result["form"]["score"])
            self.assertEqual([item["region"] for item in result["feedback"]], ["recognition"])
            result = evaluator.evaluate(user_path, label, {
                "predicted_label": "Different", "confidence": 0.69999,
            }).to_dict()
            self.assertEqual(result["status"], "needs_label_confirmation")
            self.assertEqual(result["form"]["score"], 100)
            self.assertTrue(any(item.get("region") == "recognition"
                                and item["severity"] == "warning"
                                for item in result["feedback"]))

            # The user's example must be scored from regional limits even when
            # the independent DTW percentile or an old validation disagrees.
            evaluator._validation_lookup = {"one hand": {"threshold_distance": 0}}
            with patch.object(evaluator, "_region_thresholds", return_value={
                "thresholds": {"right_hand": 1.302055, "face": 1.763011},
                "sample_count": 24,
            }), patch("sign_eval.evaluator.region_distances_on_path", return_value=(
                {"right_hand": 0.9662, "face": 0.2686}, {}
            )), patch.object(evaluator, "_reference_distribution", return_value=np.array([-1.0])):
                result = evaluator.evaluate(user_path, label, {
                    "predicted_label": label, "confidence": 0.9, "margin": 0.5
                }).to_dict()
            self.assertEqual(result["form"]["score"], 84.84)
            self.assertEqual(result["form"]["status"], "good")
            self.assertEqual(result["form"]["reference_percentile"], 0)
            self.assertEqual(result["form"]["decision_source"], "weighted_region_thresholds")
            self.assertEqual(result["status"], "correct")
            self.assertEqual(result["form"]["score"], round(sum(
                item["score"] * item["weight"] for item in result["feedback"]
            ), 2))

            with patch.object(evaluator, "_region_thresholds", return_value={
                "thresholds": None, "sample_count": 2,
            }):
                result = evaluator.evaluate(user_path, label).to_dict()
            self.assertIsNone(result["form"]["score"])
            self.assertEqual(result["status"], "not_scored")

            # Even zero simultaneous hand coverage must not suppress form or
            # either hand's feedback. Invalid all-zero arrays still cannot score.
            evaluator.calibration["labels"][label]["required_regions"] = [
                "left_hand", "right_hand", "face"
            ]
            for detected_frames in (0, 1, 2):
                with self.subTest(detected_frames=detected_frames):
                    sparse_validity = np.zeros((5, 3), dtype=bool)
                    sparse_validity[:, 2] = True
                    sparse_validity[:detected_frames, :2] = True
                    sparse_user = user.copy()
                    sparse_user[detected_frames:, :42] = 0
                    np.savez_compressed(user_path, landmarks=sparse_user, validity=sparse_validity)
                    result = evaluator.evaluate(user_path, label).to_dict()
                    self.assertEqual(result["tracking"]["status"], "low_quality")
                    self.assertIsNotNone(result["form"]["score"])
                    self.assertEqual(set(result["form"]["region_scores"]),
                                     {"left_hand", "right_hand", "face"})
                    self.assertTrue(any(item.get("region") == "tracking"
                                        and item["severity"] == "warning"
                                        for item in result["feedback"]))

            np.savez_compressed(user_path, landmarks=np.zeros_like(user), validity=validity)
            result = evaluator.evaluate(user_path, label).to_dict()
            self.assertEqual(result["tracking"]["status"], "invalid")
            self.assertEqual(result["status"], "not_scored")


if __name__ == "__main__":
    unittest.main()
