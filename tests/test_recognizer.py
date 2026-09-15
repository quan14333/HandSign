import unittest
from types import SimpleNamespace

from recognizer import LABEL_TEXT_CORRECTIONS, _repair_label_text


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


if __name__ == "__main__":
    unittest.main()
