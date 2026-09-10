import unittest

import numpy as np

from sign_eval.landmarks import (
    LandmarkSequence,
    calculate_dtw,
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

    def test_validation_threshold_uses_smaller_distances_as_better_form(self) -> None:
        threshold, balanced_accuracy = _find_best_threshold([1.0, 2.0], [4.0, 5.0])
        self.assertGreater(threshold, 2.0)
        self.assertLess(threshold, 4.0)
        self.assertEqual(balanced_accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
