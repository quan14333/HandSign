import unittest

import numpy as np

from sign_eval.landmarks import (
    LandmarkSequence,
    calculate_dtw,
    infer_required_regions,
    tracking_report,
    trim_to_active_hands,
    zscore_normalize_sequence,
)
from sign_eval.validation import _find_best_threshold


class LandmarkUtilitiesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sequence = np.arange(5 * 48 * 2, dtype=np.float32).reshape(5, 48, 2)

    def test_normalization_matches_reference_contract(self) -> None:
        normalized = zscore_normalize_sequence(self.sequence)
        self.assertTrue(np.allclose(normalized.mean(axis=(0, 1)), 0, atol=1e-6))
        self.assertTrue(np.allclose(normalized.std(axis=(0, 1)), 1, atol=1e-6))

    def test_trim_uses_the_same_active_range_for_masks_and_landmarks(self) -> None:
        validity = np.array(
            [
                [False, False, True],
                [True, False, True],
                [False, True, True],
                [False, False, True],
                [False, False, False],
            ]
        )
        trimmed, trimmed_validity = trim_to_active_hands(self.sequence, validity)
        self.assertEqual(trimmed.shape[0], 2)
        self.assertTrue(np.array_equal(trimmed, self.sequence[1:3]))
        self.assertTrue(np.array_equal(trimmed_validity, validity[1:3]))

    def test_dtw_distance_is_zero_for_identical_sequences(self) -> None:
        distance, path = calculate_dtw(self.sequence, self.sequence.copy())
        self.assertEqual(distance, 0.0)
        self.assertGreater(len(path), 0)

    def test_tracking_rejects_missing_hands_in_mask_aware_data(self) -> None:
        validity = np.zeros((5, 3), dtype=bool)
        report = tracking_report(LandmarkSequence(self.sequence, validity))
        self.assertEqual(report["status"], "low_quality")

    def test_tracking_only_requires_the_hands_used_by_the_label(self) -> None:
        validity = np.zeros((5, 3), dtype=bool)
        validity[:3, 1] = True
        validity[:, 2] = True
        report = tracking_report(
            LandmarkSequence(self.sequence, validity),
            ("right_hand", "face"),
        )
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["required_hand_detected_fraction"], 0.6)

    def test_required_regions_ignore_sporadic_unused_hand_detections(self) -> None:
        references = []
        for offset in range(3):
            sequence = np.zeros((5, 48, 2), dtype=np.float32)
            sequence[:, 21:42, 0] = np.arange(21) + offset
            sequence[:, 42:48, 0] = np.arange(6)
            if offset == 0:
                sequence[0, :21, 0] = np.arange(21)
            references.append(sequence)
        regions, presence = infer_required_regions(references)
        self.assertEqual(regions, ("right_hand", "face"))
        self.assertLess(presence["left_hand"], 0.10)
        self.assertEqual(presence["right_hand"], 1.0)

    def test_dtw_can_ignore_an_inactive_region(self) -> None:
        changed_left = self.sequence.copy()
        changed_left[:, :21] += 1000
        full_distance, _ = calculate_dtw(changed_left, self.sequence)
        active_distance, _ = calculate_dtw(
            changed_left, self.sequence, ("right_hand", "face")
        )
        self.assertGreater(full_distance, 0)
        self.assertEqual(active_distance, 0.0)

    def test_validation_threshold_uses_smaller_distances_as_better_form(self) -> None:
        threshold, balanced_accuracy = _find_best_threshold([1.0, 2.0], [4.0, 5.0])
        self.assertGreater(threshold, 2.0)
        self.assertLess(threshold, 4.0)
        self.assertEqual(balanced_accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
