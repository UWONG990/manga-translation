import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.evaluation.evaluate_advanced_metrics import (
    build_direct_assessment_prompt,
    load_evaluation_pages,
    parse_judge_response,
)


class DirectAssessmentTests(unittest.TestCase):
    def test_prompt_includes_source_candidate_and_reference(self):
        prompt = build_direct_assessment_prompt("日本語", "English candidate", "English reference")

        self.assertIn("日本語", prompt)
        self.assertIn("English candidate", prompt)
        self.assertIn("English reference", prompt)
        self.assertIn("[RESULT]", prompt)

    def test_prompt_handles_missing_reference(self):
        prompt = build_direct_assessment_prompt("日本語", "English candidate", None)

        self.assertIn("No reference answer is available.", prompt)

    def test_parses_score_and_feedback(self):
        result = parse_judge_response("Feedback: Accurate and fluent. [RESULT] 4")

        self.assertEqual(result["score"], 4)
        self.assertEqual(result["feedback"], "Accurate and fluent.")

    def test_rejects_unparseable_score(self):
        with self.assertRaisesRegex(ValueError, "no \\[RESULT\\] score"):
            parse_judge_response("Looks good, score four.")


class EvaluationPageLoadingTests(unittest.TestCase):
    def test_loads_translation_from_nested_ui_folder(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            annotation_path = root / "annotation.json"
            annotation_path.write_text(json.dumps([{
                "book_title": "sample_book",
                "pages": [{
                    "image_paths": {"ja": "images/sample_book/ja/000.jpg"},
                    "text": [{"text_ja": "日本語", "text_en": "English"}],
                }],
            }]), encoding="utf-8")
            page_dir = root / "ui" / "sample_book_000"
            page_dir.mkdir(parents=True)
            (page_dir / "translation.json").write_text(json.dumps([{
                "source_text": "日本語",
                "translated_text": "English candidate",
            }]), encoding="utf-8")

            pages = load_evaluation_pages(
                "sample_book",
                root / "ui",
                annotation_path,
            )

        self.assertEqual(pages[0]["image"], "000.jpg")
        self.assertEqual(pages[0]["references"][0]["text_en"], "English")
        self.assertEqual(
            pages[0]["translations"][0]["translated_text"],
            "English candidate",
        )


if __name__ == "__main__":
    unittest.main()
