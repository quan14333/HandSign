import copy
import json
from itertools import combinations
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from sign_eval.calibration import build_calibration
from sign_eval.evaluator import SignEvaluator
from sign_eval.landmarks import calculate_dtw, load_landmark_sequence


class PrecomputedThresholdTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.references = self.root / "references"
        self.calibration_path = self.root / "calibration.json"
        self.pair_info_path = self.root / "info.npy"
        rows = []
        self.sequences = {}
        for label, count, both_hands in [
            ("Both", 4, True), ("Right", 4, False), ("Sparse", 2, True), ("Empty", 0, True)
        ]:
            label_dir = self.references / label
            label_dir.mkdir(parents=True)
            sequences = {}
            for index in range(count):
                sequence = np.zeros((5, 48, 2), dtype=np.float32)
                if both_hands:
                    sequence[:, :21, 0] = np.arange(21) + index
                sequence[:, 21:42, 0] = np.arange(21) + 10 + 2 * index
                sequence[:, 42:48, 0] = np.arange(6) + 5 + 0.5 * index
                filename = f"{index}.npy"
                np.save(label_dir / filename, sequence)
                sequences[filename] = sequence
            for file_a, file_b in combinations(sequences, 2):
                distance, _ = calculate_dtw(sequences[file_a], sequences[file_b])
                rows.append([label, file_a, file_b, str(distance)])
            self.sequences[label] = sequences
        # Invalid and duplicate references must stay excluded from threshold building.
        np.save(self.references / "Both" / "zero.npy", np.zeros((5, 48, 2)))
        np.save(self.references / "Both" / "duplicate.npy", self.sequences["Both"]["0.npy"])
        np.save(self.pair_info_path, np.asarray(rows))
        self.payload = build_calibration(
            self.references, self.pair_info_path, self.calibration_path
        )

    def evaluator(self) -> SignEvaluator:
        return SignEvaluator(
            self.calibration_path, validation_path=self.root / "no-validation.json"
        )

    def write_calibration(self, payload: dict) -> None:
        self.calibration_path.write_text(json.dumps(payload), encoding="utf-8")

    def test_builder_persists_p90_for_each_labels_active_regions(self) -> None:
        saved = json.loads(self.calibration_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["format_version"], 3)
        both = saved["labels"]["Both"]
        self.assertEqual(both["reference_count"], 4)
        self.assertEqual(both["required_regions"], ["left_hand", "right_hand", "face"])
        self.assertEqual(both["region_threshold_sample_count"], 5)
        # The selected pairs differ by 1, 1, 1, 2, 2 offsets: P90 = 2 offsets.
        self.assertEqual(both["region_thresholds"], {
            "left_hand": 2.0, "right_hand": 4.0, "face": 1.0
        })
        right = saved["labels"]["Right"]
        self.assertEqual(right["required_regions"], ["right_hand", "face"])
        self.assertEqual(right["region_thresholds"], {"right_hand": 4.0, "face": 1.0})
        self.assertEqual(saved["summary"]["excluded_count"], 2)

    def test_fresh_evaluators_use_saved_limits_without_recomputing_pairs(self) -> None:
        user = self.sequences["Both"]["3.npy"].copy()
        user[:, :21, 0] += 3
        user[:, 21:42, 0] += 6
        user[:, 42:48, 0] += 0.5
        user_path = self.root / "user.npy"
        np.save(user_path, user)
        for _ in range(2):
            evaluator = self.evaluator()
            with (
                patch("sign_eval.evaluator.calculate_dtw", wraps=calculate_dtw) as dtw,
                patch("sign_eval.evaluator.load_landmark_sequence", wraps=load_landmark_sequence) as load,
            ):
                result = evaluator.evaluate(user_path, "Both", {
                    "predicted_label": "Both", "confidence": 0.9, "margin": 0.5
                }).to_dict()
            # Only user/reference comparisons and one load per input; no regional pair DTW.
            self.assertEqual(dtw.call_count, 4)
            self.assertEqual(load.call_count, 5)
            self.assertEqual(result["form"]["region_scores"], {
                "left_hand": 35.0, "right_hand": 35.0, "face": 70.0
            })
            self.assertEqual(result["form"]["score"], 42.0)
            self.assertEqual(result["status"], "right_label_needs_practice")
            feedback = {item["region"]: item for item in result["feedback"] if "region" in item}
            self.assertEqual(feedback["right_hand"]["severity"], "needs_attention")
            self.assertEqual(feedback["right_hand"]["segment"]["start_frame"], 0)
            self.assertEqual(feedback["right_hand"]["segment"]["end_frame"], 4)
            self.assertEqual(feedback["face"]["severity"], "ok")

    def test_insufficient_pairs_are_saved_as_null_and_remain_unscored(self) -> None:
        for label, expected_count in [("Sparse", 1), ("Empty", 0)]:
            self.assertIsNone(self.payload["labels"][label]["region_thresholds"])
            self.assertEqual(
                self.payload["labels"][label]["region_threshold_sample_count"], expected_count
            )
        self.assertEqual(self.payload["summary"]["uncalibrated_region_labels"], 2)
        user_path = self.root / "user.npy"
        np.save(user_path, self.sequences["Sparse"]["0.npy"])
        result = self.evaluator().evaluate(user_path, "Sparse").to_dict()
        self.assertIsNone(result["form"]["score"])
        self.assertEqual(result["status"], "not_scored")
        self.assertEqual(result["feedback_calibration"], {"thresholds": None, "sample_count": 1})
        self.assertEqual(result["feedback"][0]["severity"], "unknown")

    def test_legacy_calibration_requires_an_explicit_rebuild(self) -> None:
        self.payload["format_version"] = 2
        self.write_calibration(self.payload)
        with self.assertRaisesRegex(ValueError, "python build_calibration.py"):
            self.evaluator()

    def test_missing_invalid_or_mismatched_limits_are_not_silently_recomputed(self) -> None:
        variants = []
        for field in ("required_regions", "region_thresholds", "region_threshold_sample_count"):
            details = copy.deepcopy(self.payload["labels"]["Both"])
            del details[field]
            variants.append(details)
        for value in (None, {}, {"right_hand": 4.0},
                      {"left_hand": -1.0, "right_hand": 4.0, "face": 1.0},
                      {"left_hand": float("nan"), "right_hand": 4.0, "face": 1.0}):
            details = copy.deepcopy(self.payload["labels"]["Both"])
            details["region_thresholds"] = value
            variants.append(details)
        details = copy.deepcopy(self.payload["labels"]["Both"])
        details["required_regions"] = ["right_hand", "face"]
        variants.append(details)
        for details in variants:
            with self.subTest(details=details):
                payload = copy.deepcopy(self.payload)
                payload["labels"]["Both"] = details
                self.write_calibration(payload)
                evaluator = self.evaluator()
                regions = tuple(details.get("required_regions", ["left_hand", "right_hand", "face"]))
                with patch("sign_eval.evaluator.calculate_dtw") as dtw:
                    with self.assertRaisesRegex(ValueError, "python build_calibration.py"):
                        evaluator._region_thresholds("Both", regions)
                    dtw.assert_not_called()


if __name__ == "__main__":
    unittest.main()
