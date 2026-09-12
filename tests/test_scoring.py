import unittest

import numpy as np

from sign_eval.evaluator import SignEvaluator


class PracticeScoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.validation = {
            "correct_median_distance": 4.627085,
            "threshold_distance": 4.94003,
            "incorrect_median_distance": 6.332769,
        }

    def test_agreed_example_and_score_anchors(self) -> None:
        a, b, c = self.validation.values()
        for distance, expected in [(0, 100), (a, 80), (b, 60), (c, 30),
                                   (c + (c - b), 15), (4.172959, 81.96)]:
            with self.subTest(distance=distance):
                self.assertEqual(
                    SignEvaluator._validation_score(distance, self.validation), expected
                )

    def test_score_is_bounded_monotone_and_continuous_at_anchors(self) -> None:
        scores = [SignEvaluator._validation_score(float(d), self.validation)
                  for d in np.linspace(0, 100, 10001)]
        self.assertTrue(all(0 <= s <= 100 for s in scores))
        self.assertTrue(all(left >= right for left, right in zip(scores, scores[1:])))
        for anchor in self.validation.values():
            left = SignEvaluator._validation_score(anchor - 1e-8, self.validation)
            right = SignEvaluator._validation_score(anchor + 1e-8, self.validation)
            self.assertAlmostEqual(left, right, places=2)

    def test_status_always_agrees_with_displayed_score_even_near_boundaries(self) -> None:
        a, b, _ = self.validation.values()
        distances = [0, 4.172959, a - 1e-6, a, a + 1e-6,
                     b - 1e-6, b, b + 1e-6, b + 0.01, 100]
        for distance in distances:
            score = SignEvaluator._validation_score(distance, self.validation)
            expected = "excellent" if score >= 80 else "good" if score >= 60 else "needs_practice"
            self.assertEqual(SignEvaluator._form_status(score, True, self.validation), expected)

    def test_invalid_anchors_and_distances_are_not_silently_scored(self) -> None:
        for a, b, c in [(0, 2, 3), (2, 2, 3), (3, 2, 4), (1, 2, 2), (1, 2, float("nan"))]:
            validation = dict(zip(self.validation, (a, b, c)))
            with self.assertRaises(ValueError):
                SignEvaluator._validation_score(1, validation)
        for distance in [-1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                SignEvaluator._validation_score(distance, self.validation)

    def test_labels_without_validation_keep_existing_percentile_statuses(self) -> None:
        for score, expected in [(75, "excellent"), (45, "good"),
                                (20, "needs_practice"), (0, "far_from_reference")]:
            self.assertEqual(SignEvaluator._form_status(score, False, None), expected)
        self.assertEqual(SignEvaluator._form_status(90, True, None), "estimated_low_sample")


if __name__ == "__main__":
    unittest.main()
