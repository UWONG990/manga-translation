import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.bleurt_evaluator import BLEURTEvaluator


def load_annotation_by_image(annotation_path: str | Path, image_name: str) -> dict | None:
    """Find the annotation entry for a specific image file."""
    with open(annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Assume annotation has structure: [{"book_title": ..., "pages": [...]}]
    for book in data:
        for page in book.get("pages", []):
            image_paths = page.get("image_paths", {})
            if image_paths.get("ja") and image_name in image_paths.get("ja"):
                return page
    return None


def evaluate_page(image_name: str, translation_json: str | Path, annotation_json: str | Path) -> dict:
    """Evaluate translations for a single page against ground truth."""
    with open(translation_json, "r", encoding="utf-8") as f:
        translations = json.load(f)

    reference = load_annotation_by_image(annotation_json, image_name)
    if not reference:
        return {"error": f"No annotation found for image: {image_name}"}

    evaluator = BLEURTEvaluator()
    result = evaluator.evaluate_page(reference, translations)

    return {
        "image": image_name,
        "num_bubbles": len(translations),
        "num_references": len(reference.get("text", [])),
        **result,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate manga translation quality using BLEURT.")
    parser.add_argument("--image", required=True, help="Image filename (e.g., '024.jpg')")
    parser.add_argument("--translation", required=True, help="Path to translation.json")
    parser.add_argument("--annotation", required=True, help="Path to annotation.json")
    args = parser.parse_args()

    result = evaluate_page(args.image, args.translation, args.annotation)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
