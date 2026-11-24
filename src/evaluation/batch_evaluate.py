"""Batch evaluation of manga translations across multiple pages."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.bleurt_evaluator import BLEURTEvaluator
from src.evaluation.evaluate_page import load_annotation_by_image


def find_translation_file(translation_dir: Path, book_name: str, image_name: str) -> Path | None:
    page_stem = Path(image_name).stem
    candidates = (
        translation_dir / f"{page_stem}_translation.json",
        translation_dir / f"{book_name}_{page_stem}" / "translation.json",
        translation_dir / "translation.json",
    )
    return next((path for path in candidates if path.is_file()), None)


def batch_evaluate(
    book_name: str,
    translation_dir: Path,
    annotation_path: Path,
    max_pages: int | None = None,
) -> dict[str, Any]:
    """
    Evaluate translations for multiple pages from a book.
    
    Args:
        book_name: Book title as in annotation (e.g., "balloon_dream")
        translation_dir: Directory containing translation.json files
        annotation_path: Path to annotation.json
        max_pages: Maximum number of pages to evaluate (None for all)
        
    Returns:
        Dictionary with aggregated evaluation metrics
    """
    with open(annotation_path, "r", encoding="utf-8") as f:
        annotation_data = json.load(f)

    # Find book in annotations
    book_entry = next((b for b in annotation_data if b.get("book_title") == book_name), None)
    if not book_entry:
        return {"error": f"Book '{book_name}' not found in annotations"}

    pages = book_entry.get("pages", [])
    if max_pages:
        pages = pages[:max_pages]

    evaluator = BLEURTEvaluator()
    page_results = []
    all_scores = []

    for page_idx, page_data in enumerate(pages):
        image_path = page_data.get("image_paths", {}).get("ja", "")
        if not image_path:
            continue

        # Extract filename (e.g., "000.jpg" from "images/balloon_dream/ja/000.jpg")
        image_name = Path(image_path).name

        translation_file = find_translation_file(translation_dir, book_name, image_name)
        if translation_file is None:
            print(f"  [SKIP] Page {image_name}: No translation file found")
            continue

        try:
            with open(translation_file, "r", encoding="utf-8") as f:
                translations = json.load(f)

            # Score the page
            result = evaluator.evaluate_page(page_data, translations)

            page_result = {
                "page_index": page_idx,
                "image": image_name,
                "num_bubbles": len(translations),
                "num_references": len(page_data.get("text", [])),
                **result,
            }
            page_results.append(page_result)
            all_scores.extend(result.get("scores", []))

            score_str = f"{result.get('mean', 0.0):.3f}"
            print(f"  Page {image_name:10s} ({page_idx:2d}): BLEURT={score_str}")

        except Exception as e:
            print(f"  [ERROR] Page {image_name}: {e}")
            continue

    # Aggregate statistics
    if not all_scores:
        return {"error": "No pages evaluated successfully"}

    return {
        "book_name": book_name,
        "pages_evaluated": len(page_results),
        "total_pages": len(pages),
        "stats": {
            "mean_bleurt": float(np.mean(all_scores)),
            "std_bleurt": float(np.std(all_scores)),
            "min_bleurt": float(np.min(all_scores)),
            "max_bleurt": float(np.max(all_scores)),
            "median_bleurt": float(np.median(all_scores)),
        },
        "page_results": page_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Batch evaluate manga translations with BLEURT.")
    parser.add_argument("--book", required=True, help="Book title (e.g., 'balloon_dream')")
    parser.add_argument("--translation-dir", default="outputs", help="Directory containing translation files")
    parser.add_argument("--annotation", default="open-mantra-dataset/annotation.json", help="Path to annotation.json")
    parser.add_argument("--max-pages", type=int, help="Maximum number of pages to evaluate")
    parser.add_argument("--output", help="Output JSON file for detailed results")
    
    args = parser.parse_args()

    print(f"Evaluating book: {args.book}")
    print(f"Translation directory: {args.translation_dir}")
    print(f"Annotation file: {args.annotation}")
    print()

    result = batch_evaluate(
        args.book,
        Path(args.translation_dir),
        Path(args.annotation),
        max_pages=args.max_pages,
    )

    # Print summary
    if "error" in result:
        print(f"ERROR: {result['error']}")
        return

    print(f"\n{'='*60}")
    print(f"Summary for {result['book_name']}")
    print(f"{'='*60}")
    print(f"Pages evaluated: {result['pages_evaluated']}/{result['total_pages']}")
    print(f"Mean BLEURT: {result['stats']['mean_bleurt']:.4f} ± {result['stats']['std_bleurt']:.4f}")
    print(f"BLEURT range: [{result['stats']['min_bleurt']:.4f}, {result['stats']['max_bleurt']:.4f}]")
    print(f"Median BLEURT: {result['stats']['median_bleurt']:.4f}")

    # Save detailed results if requested
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
