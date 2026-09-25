import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import evaluate
import record
from sign_eval import desktop


class EvaluationWorkflowTests(unittest.TestCase):
    def test_shared_pipeline_passes_current_video_and_saves_returned_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = root / "calibration.json"
            calibration.write_text("{}", encoding="utf-8")
            output = root / "evaluation.json"
            payload = {"target_label": "Bế mạc", "form": {"score": None}, "status": "incorrect_label"}
            evaluator = Mock()
            evaluator.evaluate.return_value.to_dict.return_value = payload
            recognition = {"predicted_label": "Cảm ơn", "confidence": 0.2}
            with patch.object(evaluate, "DEFAULT_CALIBRATION_PATH", calibration), \
                    patch.object(evaluate, "SignEvaluator", return_value=evaluator), \
                    patch("recognizer.predict_video", return_value=recognition) as predict:
                result = evaluate.evaluate_recording("Bế mạc", "new.npz", "new.mp4", output)
            predict.assert_called_once_with("new.mp4")
            evaluator.evaluate.assert_called_once_with("new.npz", "Bế mạc", recognition)
            self.assertEqual(result, payload)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), payload)

    def test_record_main_opens_processing_after_capture_then_evaluates_saved_pair(self):
        events = []
        recording = object()
        payload = {"form": {"score": 80}}
        def capture():
            events.append("capture_finished")
            return recording
        def save(value):
            self.assertIs(value, recording)
            events.append("save")
            return Path("fresh.npz"), Path("fresh.mp4")
        def process(task, target):
            self.assertEqual(target, "Bế mạc")
            events.append("processing_window")
            self.assertIs(task(), payload)
            events.append("result_window")
            return False
        with patch("sys.argv", ["record.py", "--target-label", "Bế mạc"]), \
                patch("sign_eval.evaluator.SignEvaluator") as evaluator, \
                patch.object(record, "record_sequence", side_effect=capture), \
                patch.object(desktop, "show_sample_window", return_value=True), \
                patch.object(record, "save_recording", side_effect=save), \
                patch.object(desktop, "show_evaluation_window", side_effect=process), \
                patch.object(evaluate, "evaluate_recording", return_value=payload) as score:
            evaluator.return_value.available_labels.return_value = ["Bế mạc"]
            record.main()
        self.assertEqual(events, ["capture_finished", "processing_window", "save", "result_window"])
        score.assert_called_once_with("Bế mạc", Path("fresh.npz"), Path("fresh.mp4"))

    def test_cancelled_or_invalid_capture_never_evaluates_old_recording(self):
        with patch("sys.argv", ["record.py"]), \
                patch("sign_eval.evaluator.SignEvaluator") as evaluator, \
                patch.object(desktop, "choose_target_label", return_value="Bế mạc"), \
                patch.object(desktop, "show_sample_window", return_value=True), \
                patch.object(desktop, "show_notice"), \
                patch.object(record, "record_sequence", return_value=None), \
                patch.object(record, "save_recording") as save, \
                patch.object(desktop, "show_evaluation_window") as window:
            evaluator.return_value.available_labels.return_value = ["Bế mạc"]
            record.main()
        save.assert_not_called()
        window.assert_not_called()

    def test_sample_window_can_return_to_label_picker_without_restarting(self):
        with patch("sys.argv", ["record.py", "--target-label", "Bế mạc"]), \
                patch("sign_eval.evaluator.SignEvaluator") as evaluator, \
                patch.object(desktop, "choose_target_label", return_value="Cảm ơn") as choose, \
                patch.object(desktop, "show_sample_window",
                             side_effect=["choose_another", False]) as sample, \
                patch.object(record, "record_sequence") as capture:
            evaluator.return_value.available_labels.return_value = ["Bế mạc", "Cảm ơn"]
            record.main()
        choose.assert_called_once_with(["Bế mạc", "Cảm ơn"])
        self.assertEqual(sample.call_args_list[0].args[0], "Bế mạc")
        self.assertEqual(sample.call_args_list[1].args[0], "Cảm ơn")
        capture.assert_not_called()


if __name__ == "__main__":
    unittest.main()
