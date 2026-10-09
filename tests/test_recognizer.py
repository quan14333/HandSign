import unittest
from importlib.util import find_spec
from types import SimpleNamespace
from unittest.mock import Mock, patch

from recognizer import LABEL_TEXT_CORRECTIONS, _repair_label_text, predict_video


class LabelTextRepairTests(unittest.TestCase):
    def test_repairs_text_without_reordering_class_ids(self) -> None:
        model = SimpleNamespace(config=SimpleNamespace(id2label={
            "0": "An ủi",
            "14": "Ch£ng ta",
            "28": "C†ch ly",
            "99": "Ủng hộ",
        }))

        _repair_label_text(model)

        self.assertEqual(model.config.id2label, {
            0: "An ủi",
            14: "Chúng ta",
            28: "Cách ly",
            99: "Ủng hộ",
        })
        self.assertEqual(model.config.label2id["Chúng ta"], 14)
        self.assertEqual(model.config.label2id["Cách ly"], 28)

    def test_corrections_cover_known_corrupted_labels(self) -> None:
        self.assertEqual(LABEL_TEXT_CORRECTIONS[16], "Chào")
        self.assertEqual(LABEL_TEXT_CORRECTIONS[27], "Cá")
        self.assertEqual(LABEL_TEXT_CORRECTIONS[39], "Khai báo")
        self.assertEqual(LABEL_TEXT_CORRECTIONS[87], "Xin phép")


class PredictionTests(unittest.TestCase):
    def test_rejects_invalid_top_k_before_loading_model(self):
        for top_k in (0, -1, True, 1.5):
            with patch("recognizer._load_model") as load, self.assertRaises(ValueError):
                predict_video("unused.mp4", top_k=top_k)
            load.assert_not_called()

    @unittest.skipUnless(find_spec("torch"), "torch is needed to test probability extraction")
    def test_top_one_request_still_reports_actual_runner_up_margin(self):
        import torch

        model = Mock()
        model.config.id2label = {0: "Target", 1: "Other"}
        model.return_value = SimpleNamespace(logits=torch.log(torch.tensor([[0.51, 0.49]])))
        with patch("recognizer._load_model", return_value=(lambda *args, **kwargs: {}, model, torch)), \
                patch("recognizer._load_video", return_value=[]):
            result = predict_video("unused.mp4", top_k=1)
        self.assertEqual(result["predicted_label"], "Target")
        self.assertEqual(len(result["top_predictions"]), 1)
        self.assertAlmostEqual(result["margin"], 0.02, places=6)


if __name__ == "__main__":
    unittest.main()

class LocalModelTests(unittest.TestCase):
    def test_complete_local_model_uses_local_source(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from recognizer import model_source
        with TemporaryDirectory() as directory:
            for name in ('config.json', 'preprocessor_config.json', 'classifier_sequential.pth', 'model.safetensors'):
                (Path(directory) / name).touch()
            with patch.dict('os.environ', {'HANDSIGN_MODEL_PATH': directory}):
                self.assertEqual(model_source(), (directory, True))

    def test_partial_local_model_reports_setup_error(self):
        from tempfile import TemporaryDirectory
        from recognizer import model_source, ModelSetupError
        with TemporaryDirectory() as directory, patch.dict('os.environ', {'HANDSIGN_MODEL_PATH': directory}):
            with self.assertRaisesRegex(ModelSetupError, 'download_model.py'):
                model_source()
