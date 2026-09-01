"""Batch processing pipeline for manga translation end-to-end.

Processes multiple pages from a book through the full pipeline:
detection → OCR → context → translation → inpainting → typesetting
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.prototype_pipeline import run_prototype
from src.pipeline.postprocess_pipeline import run_postprocess


def batch_process(
    book_name: str,
    annotation_path: Path,
    image_root: Path,
    model_path: Path,
    output_dir: Path,
    conf: float = 0.3,
    max_pages: int | None = None,
    skip_postprocess: bool = False,
) -> dict[str, Any]:
    """
    Process multiple pages through the full pipeline.
    
    Args:
        book_name: Book title as in annotation (e.g., "balloon_dream")
        annotation_path: Path to annotation.json
        image_root: Root directory for images
        model_path: Path to YOLO model
        output_dir: Output directory for results
        conf: Detection confidence threshold
        max_pages: Maximum number of pages to process
        skip_postprocess: Skip inpainting and typesetting
        
    Returns:
        Dictionary with processing results and statistics
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

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for each processing stage
    (output_dir / "translations").mkdir(exist_ok=True)
    (output_dir / "inpainted").mkdir(exist_ok=True)
    (output_dir / "typeset").mkdir(exist_ok=True)
    (output_dir / "contexts").mkdir(exist_ok=True)

    results = []
    successful = 0
    failed = 0

    for page_idx, page_data in enumerate(pages):
        image_path = page_data.get("image_paths", {}).get("ja", "")
        if not image_path:
            continue

        full_image_path = image_root / image_path
        if not full_image_path.exists():
            print(f"  [SKIP] Page {page_idx:2d}: Image not found at {full_image_path}")
            failed += 1
            continue

        image_name = Path(image_path).name
        stem = Path(image_name).stem

        try:
            print(f"  Processing page {page_idx:2d}: {image_name}...", end=" ", flush=True)

            # Stage 1: Detection + OCR + Translation
            proto_output = output_dir / f"stage1_{stem}"
            proto_output.mkdir(exist_ok=True)

            run_prototype(
                str(full_image_path),
                str(model_path),
                conf=conf,
                output_dir=str(proto_output),
            )

            # Move translation file to organized output
            translation_file = proto_output / "translation.json"
            if translation_file.exists():
                output_trans = output_dir / "translations" / f"{stem}_translation.json"
                translation_file.rename(output_trans)

            # Stage 2: Inpainting + Typesetting (optional)
            if not skip_postprocess:
                postprocess_output = output_dir / f"stage2_{stem}"
                postprocess_output.mkdir(exist_ok=True)

                run_postprocess(
                    str(full_image_path),
                    str(output_trans),
                    output_dir=str(postprocess_output),
                )

                # Move outputs to organized directories
                inpainted = postprocess_output / f"{stem}_inpainted.png"
                typeset = postprocess_output / f"{stem}_typeset.png"

                if inpainted.exists():
                    inpainted.rename(output_dir / "inpainted" / inpainted.name)
                if typeset.exists():
                    typeset.rename(output_dir / "typeset" / typeset.name)

            print("✓")
            successful += 1
            results.append({
                "page_index": page_idx,
                "image": image_name,
                "status": "success",
            })

        except Exception as e:
            print(f"✗ (Error: {type(e).__name__})")
            failed += 1
            results.append({
                "page_index": page_idx,
                "image": image_name,
                "status": "failed",
                "error": str(e),
            })
            continue

    return {
        "book_name": book_name,
        "pages_processed": successful,
        "pages_failed": failed,
        "total_pages": len(pages),
        "output_directory": str(output_dir),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Batch process manga pages through full pipeline.")
    parser.add_argument("--book", required=True, help="Book title (e.g., 'balloon_dream')")
    parser.add_argument("--annotation", default="open-mantra-dataset/annotation.json", help="Path to annotation.json")
    parser.add_argument("--images", default="open-mantra-dataset/images", help="Root directory for images")
    parser.add_argument("--model", default="manga_panel_detector_fp32.pt", help="Path to YOLO model")
    parser.add_argument("--output", default="batch_outputs", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.3, help="Detection confidence threshold")
    parser.add_argument("--max-pages", type=int, help="Maximum number of pages to process")
    parser.add_argument("--skip-postprocess", action="store_true", help="Skip inpainting and typesetting")

    args = parser.parse_args()

    print(f"Batch Processing Pipeline")
    print(f"{'='*60}")
    print(f"Book: {args.book}")
    print(f"Annotation: {args.annotation}")
    print(f"Images root: {args.images}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print()

    result = batch_process(
        args.book,
        Path(args.annotation),
        Path(args.images),
        Path(args.model),
        Path(args.output),
        conf=args.conf,
        max_pages=args.max_pages,
        skip_postprocess=args.skip_postprocess,
    )

    print(f"\n{'='*60}")
    print(f"Batch Processing Complete")
    print(f"{'='*60}")

    if "error" in result:
        print(f"ERROR: {result['error']}")
        return

    print(f"Book: {result['book_name']}")
    print(f"Successful: {result['pages_processed']}/{result['total_pages']}")
    print(f"Failed: {result['pages_failed']}")
    print(f"Output directory: {result['output_directory']}")
    print()
    print("Next step: Run batch_evaluate.py to assess translation quality")
    print(f"  python src/evaluation/batch_evaluate.py --book {args.book} --translation-dir {args.output}/translations --annotation {args.annotation}")


if __name__ == "__main__":
    main()
