"""Evaluate translations with reference-free COMET-QE and M-Prometheus."""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.batch_evaluate import find_translation_file


JUDGE_RESULT_PATTERN = re.compile(r"\[RESULT\]\s*([1-5])\b")


def build_direct_assessment_prompt(
    source: str,
    candidate: str,
    reference: str | None,
) -> str:
    reference_text = reference or "No reference answer is available."
    return f"""###Task Description: Evaluate a translation using the score rubric.
1. Give concise feedback based only on the rubric.
2. Then give an integer score from 1 to 5.
3. Use exactly this output format: Feedback: (your feedback) [RESULT] (score)
Do not add an opening or closing.

###The instruction to evaluate:
Translate the following text from Japanese to English: {source}

###Response to evaluate:
{candidate}

###Reference Answer (Score 5):
{reference_text}

###Score Rubric: Accuracy, Fluency, and Manga Dialogue Style
Score 1: Major meaning errors; barely comprehensible; style does not fit the source.
Score 2: Several inaccuracies; awkward or difficult to read; style rarely fits.
Score 3: Mostly accurate and understandable, with minor errors or awkwardness.
Score 4: Accurate and natural, with only negligible errors; style mostly fits.
Score 5: Fully accurate and fluent; naturally conveys the source's tone and register.

###Feedback:"""


def parse_judge_response(response: str) -> dict[str, Any]:
    match = JUDGE_RESULT_PATTERN.search(response)
    if match is None:
        raise ValueError(f"M-Prometheus response has no [RESULT] score: {response!r}")

    feedback = response[:match.start()].strip()
    if feedback.lower().startswith("feedback:"):
        feedback = feedback[len("feedback:"):].strip()

    return {"score": int(match.group(1)), "feedback": feedback, "raw_response": response}


def load_evaluation_pages(
    book_name: str,
    translation_dir: Path,
    annotation_path: Path,
) -> list[dict[str, Any]]:
    with annotation_path.open("r", encoding="utf-8") as file:
        annotations = json.load(file)

    book = next((entry for entry in annotations if entry.get("book_title") == book_name), None)
    if book is None:
        raise ValueError(f"Book {book_name!r} not found in {annotation_path}")

    pages = []
    for page in book.get("pages", []):
        image_path = page.get("image_paths", {}).get("ja", "")
        if not image_path:
            continue

        image_name = Path(image_path).name
        translation_path = find_translation_file(translation_dir, book_name, image_name)
        if translation_path is None:
            print(f"  [SKIP] Page {image_name}: No translation file found")
            continue

        with translation_path.open("r", encoding="utf-8") as file:
            translations = json.load(file)
        if not isinstance(translations, list):
            raise ValueError(f"Expected a list of translations in {translation_path}")

        pages.append({
            "image": image_name,
            "references": page.get("text", []),
            "translations": translations,
            "num_references": len(page.get("text", [])),
            "num_candidates": len(translations),
        })

    if not pages:
        raise ValueError(f"No matching translations found in {translation_dir}")
    return pages


def score_comet_qe(pages: list[dict[str, Any]], model_name: str) -> list[float]:
    try:
        import torch
        from comet import download_model, load_from_checkpoint
    except ImportError as exc:
        raise ImportError(
            "COMET is not installed. Install requirements-advanced-eval.txt "
            "in the evaluation environment."
        ) from exc

    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)
    gpu_count = 1 if torch.cuda.is_available() else 0
    all_scores = []

    for page in pages:
        translations = page["translations"]
        rows = []
        indices = []
        for index, item in enumerate(translations):
            source = item.get("source_text", "").strip()
            candidate = item.get("translated_text", "").strip()
            if source and candidate:
                rows.append({"src": source, "mt": candidate})
                indices.append(index)

        if not rows:
            page["comet_qe"] = {"scores": [], "mean": None}
            continue

        prediction = model.predict(rows, batch_size=8, gpus=gpu_count)
        scores = [float(score) for score in prediction.scores]
        page["comet_qe"] = {
            "scores": scores,
            "mean": float(np.mean(scores)),
            "translation_indices": indices,
        }
        all_scores.extend(scores)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return all_scores


class MPrometheusDirectAssessor:
    def __init__(self, model_name: str):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError(
                "The M-Prometheus runtime requires torch, transformers, accelerate, "
                "and bitsandbytes; see requirements-advanced-eval.txt."
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available in this Python environment. "
                "Use the CUDA-enabled evaluation environment for 4-bit M-Prometheus."
            )

        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        offload_folder = Path(
            os.environ.get("HF_HOME", REPO_ROOT / "outputs" / "hf-cache")
        ) / "m-prometheus-offload"
        offload_folder.mkdir(parents=True, exist_ok=True)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            max_memory={0: "4GiB", "cpu": "4GiB"},
            dtype=torch.float16,
            quantization_config=quantization,
            low_cpu_mem_usage=True,
            offload_state_dict=True,
            offload_folder=str(offload_folder),
        )
        self.model.eval()

    def evaluate(self, source: str, candidate: str, reference: str | None) -> dict[str, Any]:
        prompt = build_direct_assessment_prompt(source, candidate, reference)
        chat_prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            chat_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        )
        inputs = inputs.to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=192,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = output[0, inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return parse_judge_response(response)


def score_with_judge(
    pages: list[dict[str, Any]],
    model_name: str,
) -> list[float]:
    judge = MPrometheusDirectAssessor(model_name)
    all_scores = []

    for page in pages:
        references = page["references"]
        results = []
        for index, item in enumerate(page["translations"]):
            source = item.get("source_text", "").strip()
            candidate = item.get("translated_text", "").strip()
            if not source or not candidate:
                continue

            reference = ""
            if index < len(references):
                reference = references[index].get("text_en", "").strip()

            result = judge.evaluate(source, candidate, reference or None)
            results.append({"translation_index": index, **result})
            all_scores.append(result["score"])

        scores = [result["score"] for result in results]
        page["m_prometheus_direct_assessment"] = {
            "scores_1_to_5": scores,
            "mean": float(np.mean(scores)) if scores else None,
            "results": results,
        }

    del judge
    return all_scores


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate UI translations with COMET-QE and M-Prometheus."
    )
    parser.add_argument("--book", required=True, help="Book title from annotation.json")
    parser.add_argument("--translation-dir", required=True, type=Path)
    parser.add_argument("--annotation", default="open-mantra-dataset/annotation.json", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--qe-model", default="Unbabel/wmt20-comet-qe-da")
    parser.add_argument("--judge-model", default="Unbabel/M-Prometheus-3B")
    parser.add_argument("--skip-comet-qe", action="store_true")
    parser.add_argument("--skip-judge", action="store_true")
    args = parser.parse_args()

    if args.skip_comet_qe and args.skip_judge:
        parser.error("At least one evaluator must be enabled.")

    pages = load_evaluation_pages(args.book, args.translation_dir, args.annotation)
    print(f"Matched {len(pages)} translated pages for {args.book}.")

    stats: dict[str, Any] = {}
    models: dict[str, str] = {}
    if not args.skip_comet_qe:
        comet_scores = score_comet_qe(pages, args.qe_model)
        stats["mean_comet_qe"] = float(np.mean(comet_scores)) if comet_scores else None
        models["comet_qe"] = args.qe_model

    if not args.skip_judge:
        judge_scores = score_with_judge(pages, args.judge_model)
        stats["mean_m_prometheus_direct_assessment"] = (
            float(np.mean(judge_scores)) if judge_scores else None
        )
        models["m_prometheus_direct_assessment"] = args.judge_model

    result = {
        "book_name": args.book,
        "pages_evaluated": len(pages),
        "models": models,
        "stats": stats,
        "page_results": pages,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    print(json.dumps({"pages_evaluated": len(pages), "stats": stats}, indent=2))
    print(f"Detailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
