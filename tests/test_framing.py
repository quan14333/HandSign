import unittest
from types import SimpleNamespace

from sign_eval.framing import CaptureGate, assess_framing


def centered_pose():
    # Everything except the shoulders is deliberately missing, including hips.
    points = [SimpleNamespace(x=float("nan"), y=float("nan"), visibility=0, presence=0)
              for _ in range(33)]
    for index, x in [(11, 0.15), (12, 0.85)]:
        points[index] = SimpleNamespace(x=x, y=0.8, visibility=0.99, presence=0.99)
    return points


class FramingTests(unittest.TestCase):
    def test_face_and_two_shoulders_are_enough_without_hips_or_hands(self):
        self.assertTrue(assess_framing(centered_pose(), face_detected=True).ready)

    def test_face_must_be_detected_even_when_shoulders_are_visible(self):
        self.assertEqual(assess_framing(centered_pose(), False).status, "no_face")

    def test_missing_low_confidence_and_nonfinite_shoulders_block_start(self):
        self.assertEqual(assess_framing([], True).status, "no_shoulders")
        for index in (11, 12):
            for field, value in [("visibility", 0.3), ("presence", 0.3),
                                 ("presence", None), ("x", float("nan")),
                                 ("y", float("inf"))]:
                with self.subTest(index=index, field=field, value=value):
                    points = centered_pose()
                    setattr(points[index], field, value)
                    self.assertEqual(assess_framing(points, True).status, "not_visible")

    def test_shoulders_must_be_inside_image(self):
        for index, field, value in [(11, "x", -0.01), (12, "x", 1.01),
                                    (11, "y", -0.01), (12, "y", 1.01)]:
            points = centered_pose()
            setattr(points[index], field, value)
            self.assertEqual(assess_framing(points, True).status, "out_of_frame")

    def test_no_distance_centering_or_torso_size_requirement(self):
        for left_x, right_x, y in [(0.02, 0.98, 0.98), (0.2, 0.3, 0.4), (0.7, 0.9, 0.3)]:
            points = centered_pose()
            points[11].x, points[12].x = left_x, right_x
            points[11].y = points[12].y = y
            self.assertTrue(assess_framing(points, True).ready)


class CaptureGateTests(unittest.TestCase):
    def test_green_preview_requires_s_and_three_second_countdown(self):
        gate = CaptureGate()
        gate.update(True, 0)
        self.assertTrue(gate.ready)
        self.assertFalse(gate.recording)
        self.assertTrue(gate.request_start(0))
        gate.update(True, 2.99)
        self.assertFalse(gate.recording)
        self.assertFalse(gate.request_start(2.99))
        gate.update(True, 3)
        self.assertTrue(gate.recording)
        self.assertFalse(gate.request_start(3))

    def test_s_before_ready_is_not_queued(self):
        gate = CaptureGate()
        self.assertFalse(gate.request_start(0))
        gate.update(True, 0)
        self.assertFalse(gate.recording)
        self.assertTrue(gate.request_start(0))

    def test_losing_face_or_shoulder_before_s_revokes_readiness(self):
        gate = CaptureGate()
        gate.update(True, 0)
        gate.update(False, 1)
        self.assertFalse(gate.ready)
        self.assertFalse(gate.request_start(1))
        self.assertFalse(gate.recording)

    def test_recording_continues_if_framing_changes_during_signing(self):
        gate = CaptureGate()
        gate.update(True, 0)
        gate.request_start(0)
        gate.update(True, 3)
        gate.update(False, 4)
        self.assertTrue(gate.recording)
        self.assertFalse(gate.request_start(4))

    def test_framing_loss_at_countdown_deadline_cancels_start(self):
        gate = CaptureGate()
        gate.update(True, 0)
        gate.request_start(0)
        gate.update(False, 3)
        self.assertIsNone(gate.countdown_until)
        gate.update(True, 4)
        self.assertFalse(gate.recording)
        self.assertTrue(gate.request_start(4))
        gate.update(True, 7)
        self.assertTrue(gate.recording)


if __name__ == "__main__":
    unittest.main()
