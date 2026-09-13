import unittest

import numpy as np

from sign_eval.evaluator import SignEvaluator


class PracticeScoreTests(unittest.TestCase):
    def test_different_label_requires_70_percent_confidence(self) -> None:
        evaluator = SignEvaluator.__new__(SignEvaluator)
        for confidence, expected in [(0.69999, "uncertain"), (0.7, "mismatch"),
                                     (0.95, "mismatch"), (None, "uncertain"),
                                     (float("nan"), "uncertain")]:
            with self.subTest(confidence=confidence):
                result = evaluator._recognition_status("Target", {
                    "predicted_label": "Different", "confidence": confidence,
                })
                self.assertEqual(result["status"], expected)
        self.assertEqual(evaluator._recognition_status("Target", {
            "predicted_label": " target ", "confidence": 0.6, "margin": 0.2,
        })["status"], "match")

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

    def test_status_agrees_with_new_displayed_score_boundaries(self) -> None:
        for score, expected in [(100, "excellent"), (85, "excellent"),
                                (84.99, "good"), (70, "good"),
                                (69.99, "needs_practice"), (30, "needs_practice"),
                                (29.99, "far_from_reference"), (0, "far_from_reference"),
                                (None, "not_scored")]:
            self.assertEqual(SignEvaluator._form_status(score), expected)

    def test_invalid_anchors_and_distances_are_not_silently_scored(self) -> None:
        for a, b, c in [(0, 2, 3), (2, 2, 3), (3, 2, 4), (1, 2, 2), (1, 2, float("nan"))]:
            validation = dict(zip(self.validation, (a, b, c)))
            with self.assertRaises(ValueError):
                SignEvaluator._validation_score(1, validation)
        for distance in [-1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                SignEvaluator._validation_score(distance, self.validation)

    def test_regional_score_anchors_and_tolerance(self) -> None:
        for ratio, expected in [(0, 100), (0.3, 100), (0.65, 85),
                                (1, 70), (2, 35), (3, 17.5)]:
            self.assertEqual(SignEvaluator._region_score(ratio * 2, 2), expected)
        self.assertEqual(SignEvaluator._region_score(0, 0), 100)
        self.assertEqual(SignEvaluator._region_score(0.1, 0), 0)

    def test_regional_score_is_continuous_bounded_and_monotone(self) -> None:
        scores = [SignEvaluator._region_score(float(r), 1) for r in np.linspace(0, 100, 1001)]
        self.assertTrue(all(0 <= score <= 100 for score in scores))
        self.assertTrue(all(a >= b for a, b in zip(scores, scores[1:])))
        for anchor in (0.3, 1):
            self.assertEqual(SignEvaluator._region_score(anchor - 1e-8, 1),
                             SignEvaluator._region_score(anchor + 1e-8, 1))
        for distance, threshold in [(-1, 1), (1, -1), (float("nan"), 1),
                                    (1, float("inf"))]:
            with self.assertRaises(ValueError):
                SignEvaluator._region_score(distance, threshold)

    def test_weights_follow_active_hands(self) -> None:
        for regions, expected in [
            (("right_hand", "face"), {"right_hand": 0.8, "face": 0.2}),
            (("left_hand", "face"), {"left_hand": 0.8, "face": 0.2}),
            (("left_hand", "right_hand", "face"),
             {"left_hand": 0.4, "right_hand": 0.4, "face": 0.2}),
            (("right_hand",), {"right_hand": 1.0}),
        ]:
            self.assertEqual(SignEvaluator._region_weights(regions), expected)


if __name__ == "__main__":
    unittest.main()
