import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.typesetting.typesetter import render_translated_page, resolve_font_path


class TypesettingFontTest(unittest.TestCase):
    def test_resolve_font_rejects_missing_path(self):
        self.assertIsNone(resolve_font_path("C:/missing/custom.ttf"))

    def test_render_accepts_formatting_options(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            source = temp_dir / "page.png"
            output = temp_dir / "typeset.png"
            Image.new("RGB", (300, 200), "gray").save(source)

            result = render_translated_page(
                source,
                [{
                    "index": 0,
                    "bbox": [40, 40, 260, 120],
                    "translated_text": "Editable text",
                }],
                output,
                "C:/Windows/Fonts/arial.ttf",
                24,
                "#FFFFFF",
                "#333333",
                "#AAAAAA",
                1.2,
                "left",
            )

            self.assertTrue(Path(result).exists())

    def test_render_accepts_custom_font_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            source = temp_dir / "page.png"
            output = temp_dir / "typeset.png"
            Image.new("RGB", (200, 200), "white").save(source)

            result = render_translated_page(
                source,
                [{
                    "index": 0,
                    "bbox": [30, 30, 170, 100],
                    "translated_text": "Hello",
                }],
                output,
                "C:/Windows/Fonts/arial.ttf",
            )

            self.assertEqual(Path(result), output)
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
