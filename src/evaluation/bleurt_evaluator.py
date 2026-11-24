from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


class BLEURTEvaluator:
    """Score translations against references with a BLEURT checkpoint."""

    def __init__(self, checkpoint: str | Path | None = None, scorer: Any | None = None):
        if scorer is not None:
            self.scorer = scorer
            return

        try:
            from bleurt.score import BleurtScorer
        except ImportError as exc:
            raise ImportError(
                "BLEURT is not installed. Install it with: "
                "pip install -r requirements-bleurt.txt"
            ) from exc

        checkpoint_value = checkpoint or os.environ.get("BLEURT_CHECKPOINT")
        checkpoint_path = Path(checkpoint_value) if checkpoint_value else (
            Path(__file__).resolve().parents[2] / "bleurt-checkpoints" / "BLEURT-20"
        )
        if not checkpoint_path.is_dir():
            raise FileNotFoundError(
                f"BLEURT checkpoint directory not found: {checkpoint_path}. "
                "Extract a BLEURT checkpoint directory and set BLEURT_CHECKPOINT "
                "to its path. BLEURT cannot load a .zip file directly."
            )

        self.scorer = BleurtScorer(checkpoint=str(checkpoint_path))

    def score_batch(self, hypotheses: list[str], references: list[str], sources: list[str] | None = None) -> dict[str, Any]:
        """Score candidate translations against reference translations.
        
        Args:
            hypotheses: Candidate translations.
            references: Reference translations, one per candidate.
            sources: Optional source texts, retained for call compatibility.
            
        Returns:
            BLEURT scores and summary statistics.
        """
        if len(hypotheses) != len(references):
            raise ValueError("hypotheses and references must be same length")
        if not hypotheses:
            raise ValueError("hypotheses and references must not be empty")

        scores = [
            float(score)
            for score in self.scorer.score(references=references, candidates=hypotheses)
        ]
        mean_score = float(np.mean(scores))
        return {
            "metric": "BLEURT",
            "scores": scores,
            "mean": mean_score,
            "std": float(np.std(scores)) if len(scores) > 1 else 0.0,
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
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
