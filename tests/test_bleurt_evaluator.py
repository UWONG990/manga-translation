import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.evaluation.batch_evaluate import find_translation_file
from src.evaluation.bleurt_evaluator import BLEURTEvaluator


class FakeBLEURTScorer:
    def __init__(self):
        self.references = None
        self.candidates = None

    def score(self, references, candidates):
        self.references = references
        self.candidates = candidates
        return [0.2, 0.8]


class BLEURTEvaluatorTests(unittest.TestCase):
    def test_score_batch_uses_bleurt_scores_and_reports_bleurt(self):
        scorer = FakeBLEURTScorer()
        evaluator = BLEURTEvaluator(scorer=scorer)

        result = evaluator.score_batch(
            hypotheses=["A candidate.", "Another candidate."],
            references=["A reference.", "Another reference."],
        )

        self.assertEqual(scorer.references, ["A reference.", "Another reference."])
        self.assertEqual(scorer.candidates, ["A candidate.", "Another candidate."])
        self.assertEqual(result["metric"], "BLEURT")
        self.assertEqual(result["scores"], [0.2, 0.8])
        self.assertAlmostEqual(result["mean"], 0.5)
        self.assertAlmostEqual(result["std"], 0.3)
        self.assertEqual(result["min"], 0.2)
        self.assertEqual(result["max"], 0.8)
        self.assertNotIn("corpus_bleu", result)

    def test_score_batch_rejects_mismatched_pairs(self):
        evaluator = BLEURTEvaluator(scorer=FakeBLEURTScorer())

        with self.assertRaisesRegex(ValueError, "same length"):
            evaluator.score_batch(["candidate"], ["reference", "extra"])

    def test_score_batch_rejects_empty_input(self):
        evaluator = BLEURTEvaluator(scorer=FakeBLEURTScorer())

        with self.assertRaisesRegex(ValueError, "must not be empty"):
            evaluator.score_batch([], [])


class TranslationFileLookupTests(unittest.TestCase):
    def test_finds_translation_in_ui_page_folder(self):
        with TemporaryDirectory() as temp_dir:
            translation_dir = Path(temp_dir)
            page_dir = translation_dir / "balloon_dream_017"
            page_dir.mkdir()
            expected = page_dir / "translation.json"
            expected.write_text("[]", encoding="utf-8")

            result = find_translation_file(translation_dir, "balloon_dream", "017.jpg")

            self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
