import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.typesetting.typesetter import (
    build_editable_project,
    export_editor_html,
    render_translated_page,
    save_editable_project,
    resolve_font_path,
)


class TypesettingFontTest(unittest.TestCase):
    def test_build_project_and_editor_exports(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            source = temp_dir / "page.png"
            cleaned = temp_dir / "cleaned.png"
            project_path = temp_dir / "project.json"
            editor_path = temp_dir / "editor.html"
            Image.new("RGB", (300, 200), "white").save(source)
            Image.new("RGB", (300, 200), "white").save(cleaned)

            project = build_editable_project(
                source,
                cleaned,
                [{
                    "index": 0,
                    "bbox": [30, 30, 170, 100],
                    "source_text": "こんにちは",
                    "translated_text": "Hello",
                }],
            )

            save_editable_project(project, project_path)
            export_editor_html(project, editor_path)

            self.assertTrue(project_path.exists())
            self.assertTrue(editor_path.exists())
            self.assertIn("pointermove", editor_path.read_text(encoding="utf-8"))
            self.assertGreater(project["bubbles"][0]["font_size"], 0)
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
