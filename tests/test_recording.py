from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import record
from test_framing import centered_pose


class RecordingTests(unittest.TestCase):
    def run_capture(self, times, keys, poses, faces=None):
        camera = Mock()
        camera.isOpened.return_value = True
        writer = Mock()
        writer.isOpened.return_value = True
        saved_frames = []
        writer.write.side_effect = lambda frame: saved_frames.append(frame.copy())
        current_time = [0.0]
        frame_index = [0]

        def read_frame():
            index = frame_index[0]
            current_time[0] = times[index]
            frame_index[0] += 1
            return True, np.full((240, 320, 3), index, dtype=np.uint8)

        camera.read.side_effect = read_frame
        hand, face, pose = Mock(), Mock(), Mock()
        pose.detect_for_video.side_effect = [
            SimpleNamespace(pose_landmarks=[points] if points else []) for points in poses
        ]
        with ExitStack() as stack:
            stack.enter_context(patch.object(record.cv2, "VideoCapture", return_value=camera))
            make_writer = stack.enter_context(patch.object(record.cv2, "VideoWriter", return_value=writer))
            stack.enter_context(patch.object(record.cv2, "imshow"))
            stack.enter_context(patch.object(record.cv2, "waitKey", side_effect=keys))
            stack.enter_context(patch.object(record.cv2, "destroyAllWindows"))
            stack.enter_context(patch.object(record.cv2, "getWindowProperty", return_value=1))
            stack.enter_context(patch.object(record.Path, "mkdir"))
            stack.enter_context(patch.object(record, "create_landmarkers", return_value=(hand, face, pose)))
            stack.enter_context(patch.object(record.time, "perf_counter", side_effect=lambda: current_time[0]))
            stack.enter_context(patch.object(record, "extract_hand_landmarks", return_value=(
                np.ones((2, 21, 2), dtype=np.float32), np.array([True, False]),
            )))
            stack.enter_context(patch.object(record, "extract_face_landmarks", side_effect=[
                (np.full((6, 2), 0.5, dtype=np.float32), visible)
                for visible in (faces if faces is not None else [True] * len(times))
            ]))
            # Obvious pixel changes reveal any preview overlay leaking into saved video.
            stack.enter_context(patch.object(record, "draw_landmarks", side_effect=lambda frame, *args: frame.fill(200)))
            stack.enter_context(patch.object(record, "draw_capture_guide", side_effect=lambda frame, *args: np.full_like(frame, 255)))
            result = record.record_sequence()
        camera.release.assert_called_once()
        return result, saved_frames, make_writer, writer, pose

    def test_start_gate_saves_only_recorded_frames_without_overlays(self):
        result, frames, make_writer, writer, pose = self.run_capture(
            [0, 0.1, 0.2, 3.2, 3.3],
            [ord("s"), -1, ord("S"), -1, ord("q")],
            [[], *[centered_pose() for _ in range(4)]],
        )
        make_writer.assert_called_once()
        writer.release.assert_called_once()
        self.assertEqual(len(frames), 2)
        self.assertTrue(np.all(frames[0] == 3))
        self.assertTrue(np.all(frames[1] == 4))
        sequence, validity, start, end = result
        self.assertEqual(sequence.shape, (2, 48, 2))
        self.assertEqual(validity.shape, (2, 3))
        self.assertEqual((start, end), (0, 1))
        timestamps = [call.args[1] for call in pose.detect_for_video.call_args_list]
        self.assertTrue(all(a < b for a, b in zip(timestamps, timestamps[1:])))

    def test_s_when_not_ready_does_not_auto_start_later_or_overwrite_video(self):
        result, frames, make_writer, writer, _ = self.run_capture(
            [0, 1, 10],
            [ord("s"), -1, ord("q")],
            [[], centered_pose(), centered_pose()],
        )
        self.assertIsNone(result)
        self.assertEqual(frames, [])
        make_writer.assert_not_called()

    def test_q_during_countdown_does_not_save_video(self):
        result, frames, make_writer, writer, _ = self.run_capture(
            [0, 1], [ord("s"), ord("q")], [centered_pose(), centered_pose()],
        )
        self.assertIsNone(result)
        self.assertEqual(frames, [])
        make_writer.assert_not_called()
        writer.write.assert_not_called()

    def test_visible_shoulders_without_face_cannot_start(self):
        result, frames, make_writer, _, _ = self.run_capture(
            [0, 0.1], [ord("s"), ord("q")],
            [centered_pose(), centered_pose()], faces=[False, False],
        )
        self.assertIsNone(result)
        self.assertEqual(frames, [])
        make_writer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
