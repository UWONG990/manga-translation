import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.translation.retranslation import (
    retranslate_table_rows,
    save_translation_results,
)
from src.typesetting.typesetter import build_editable_project


class FakeTranslator:
    def __init__(self):
        self.calls = []

    def translate(self, **kwargs):
        self.calls.append(kwargs)
        return f"English: {kwargs['text']}"


class RetranslationTest(unittest.TestCase):
    def test_saves_retranslated_results_as_utf8_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "translation.json"
            translations = [
                {
                    "index": 4,
                    "bbox": [1.0, 2.0, 30.0, 40.0],
                    "source_text": "こんにちは",
                    "translated_text": "Hello again",
                }
            ]

            saved_path = save_translation_results(translations, output_path)

            self.assertEqual(saved_path, str(output_path))
            self.assertEqual(
                json.loads(output_path.read_text(encoding="utf-8")),
                translations,
            )
            self.assertEqual(list(Path(temp_dir).glob("*.tmp")), [])

    def test_retranslates_edited_rows_and_adds_manual_rows_with_page_context(self):
        translator = FakeTranslator()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            page_path = temp_dir / "page.png"
            Image.new("RGB", (100, 100), "white").save(page_path)
            existing_crop = temp_dir / "existing.png"
            Image.new("RGB", (40, 60), "white").save(existing_crop)
            previous = [
                {
                    "index": 0,
                    "bbox": [10, 20, 50, 80],
                    "source_text": "old source",
                    "translated_text": "old translation",
                    "context": "Original content: old source",
                    "crop_path": str(existing_crop),
                    "page_image": str(page_path),
                    "scene_summary": "A short scene.",
                }
            ]

            rows, translations = retranslate_table_rows(
                [
                    [0, "corrected source", "ignore this old translation", 10, 20, 50, 80],
                    [float("nan"), "manually added line", "", 60, 10, 90, 40],
                    [None, None, None, None, None, None, None],
                ],
                previous,
                str(page_path),
                "A short scene.",
                translator,
                temp_dir / "contexts",
            )

            self.assertEqual(rows, [
                [0, "corrected source", "English: corrected source", 10.0, 20.0, 50.0, 80.0],
                [1, "manually added line", "English: manually added line", 60.0, 10.0, 90.0, 40.0],
            ])
            self.assertEqual(translations[0]["bbox"], [10, 20, 50, 80])
            project = build_editable_project(
                page_path,
                page_path,
                translations,
            )
            self.assertEqual(len(project["bubbles"]), 2)
            self.assertEqual(project["bubbles"][1]["bbox"], [60.0, 10.0, 90.0, 40.0])
            self.assertIn("Original content: corrected source", translator.calls[0]["context"])
            self.assertEqual(translator.calls[0]["image_path"], str(existing_crop))
            self.assertEqual(translator.calls[1]["page_image"], str(page_path))
            self.assertIn("Bubble 0: corrected source", translator.calls[1]["context"])
            self.assertTrue(Path(translator.calls[1]["image_path"]).exists())
            with Image.open(translator.calls[1]["image_path"]) as crop:
                self.assertEqual(crop.size, (30, 30))

    def test_rejects_duplicate_bubble_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            page_path = Path(temp_dir) / "page.png"
            Image.new("RGB", (100, 100), "white").save(page_path)
            with self.assertRaisesRegex(ValueError, "appears more than once"):
                retranslate_table_rows(
                    [
                        [0, "one", "", 1, 1, 10, 10],
                        [0, "two", "", 20, 20, 30, 30],
                    ],
                    [],
                    str(page_path),
                    "",
                    FakeTranslator(),
                    temp_dir,
                )

    def test_rejects_empty_table(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            page_path = Path(temp_dir) / "page.png"
            Image.new("RGB", (100, 100), "white").save(page_path)
            with self.assertRaisesRegex(ValueError, "at least one source-text row"):
                retranslate_table_rows(
                    [], [], str(page_path), "", FakeTranslator(), temp_dir
                )

    def test_requires_valid_coordinates_for_new_bubbles(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            page_path = Path(temp_dir) / "page.png"
            Image.new("RGB", (100, 100), "white").save(page_path)
            with self.assertRaisesRegex(ValueError, "all four coordinates"):
                retranslate_table_rows(
                    [[None, "new line", "", None, None, None, None]],
                    [],
                    str(page_path),
                    "",
                    FakeTranslator(),
                    temp_dir,
                )


if __name__ == "__main__":
    unittest.main()
