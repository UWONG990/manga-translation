from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class BLEURTEvaluator:
    """Evaluator using SacreBLEU for translation quality assessment.
    
    Uses SacreBLEU BLEU metric which is robust and widely accepted for
    translation evaluation without complex dependencies.
    """
    
    def __init__(self):
        try:
            from sacrebleu import BLEU
            self.metric = BLEU()
        except ImportError:
            raise ImportError("sacrebleu not installed. Run: pip install sacrebleu")

    def score_batch(self, hypotheses: list[str], references: list[str], sources: list[str] | None = None) -> dict[str, Any]:
        """Score a batch of hypotheses against references using SacreBLEU.
        
        Args:
            hypotheses: List of translated texts
            references: List of reference translations (or list of list for multiple references)
            sources: Optional list of source texts (not used by BLEU but kept for compatibility)
            
        Returns:
            Dictionary with BLEU score and component scores
        """
        if len(hypotheses) != len(references):
            raise ValueError("hypotheses and references must be same length")

        # SacreBLEU expects references as a single string with multiple refs separated by newlines
        # For now, we treat each reference as a single translation
        sys_score = self.metric.corpus_score(hypotheses, [references])
        
        # Also compute individual sentence scores
        sent_scores = []
        for hyp, ref in zip(hypotheses, references):
            score = self.metric.sentence_score(hyp, [ref])
            sent_scores.append(score.score)
        
        return {
            "scores": sent_scores,
            "mean": float(np.mean(sent_scores)),
            "std": float(np.std(sent_scores)) if len(sent_scores) > 1 else 0.0,
            "min": float(np.min(sent_scores)),
            "max": float(np.max(sent_scores)),
            "corpus_bleu": sys_score.score,
            "corpus_precisions": sys_score.precisions,
        }

    def evaluate_page(self, reference_data: dict[str, Any], hypothesis_data: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Evaluate translations for a single page.
        reference_data: entry from annotation.json with 'text' array
        hypothesis_data: output from translation pipeline
        """
        ref_texts = [item.get("text_en", "") for item in reference_data.get("text", []) if item.get("text_en")]
        hyp_texts = [item.get("translated_text", "") for item in hypothesis_data if item.get("translated_text")]

        if not ref_texts or not hyp_texts:
            return {"error": "No reference or hypothesis texts found"}

        # Align by length; if mismatch, use min length
        min_len = min(len(ref_texts), len(hyp_texts))
        ref_texts = ref_texts[:min_len]
        hyp_texts = hyp_texts[:min_len]

        return self.score_batch(hyp_texts, ref_texts)
