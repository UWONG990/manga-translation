import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image

from src.context.context_builder import ContextBuilder
from src.detection.yolo_detector import MangaDetector
from src.ocr.manga_ocr import MangaOCR
from src.translation.translator import OllamaVisionTranslator


def _smart_sort_text_boxes(text_boxes: list[dict], image_width: int = 800, image_height: int = 1000):
    """
    Sort text boxes using manga reading conventions:
    1. Group into vertical regions (top/middle/bottom) by y-coordinate
    2. Within each region, sort right-to-left by x-coordinate (typical manga reading order)
    3. Within same x-column, sort top-to-bottom by y-coordinate
    """
    if not text_boxes:
        return []
    
    def get_bbox_center_y(box):
        return box["bbox"][1] + (box["bbox"][3] - box["bbox"][1]) / 2
    
    def get_bbox_center_x(box):
        return box["bbox"][0] + (box["bbox"][2] - box["bbox"][0]) / 2
    
    # Group into rough vertical bands (regions with similar y-coordinates)
    sorted_by_y = sorted(text_boxes, key=get_bbox_center_y)
    
    regions = []
    current_region = [sorted_by_y[0]]
    current_y_center = get_bbox_center_y(sorted_by_y[0])
    
    for box in sorted_by_y[1:]:
        box_y = get_bbox_center_y(box)
        # Group boxes within ~150px vertically (tolerance for multi-line text)
        if abs(box_y - current_y_center) < 150:
            current_region.append(box)
        else:
            if current_region:
                regions.append(current_region)
            current_region = [box]
            current_y_center = box_y
    if current_region:
        regions.append(current_region)
    
    # Within each region, apply manga reading order: right-to-left, then top-to-bottom
    ordered = []
    for region in regions:
        # Sort by x descending (right-to-left), then by y ascending (top-to-bottom)
        region_sorted = sorted(region, key=lambda b: (-get_bbox_center_x(b), get_bbox_center_y(b)))
        ordered.extend(region_sorted)
    
    return ordered


def run_prototype(image_path: str | Path, model_path: str | Path, conf: float = 0.25, output_dir: str | Path = "outputs"):
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = MangaDetector(model_path)
    detected = detector.detect(image_path, conf=conf)
    
    # Get image dimensions for smarter sorting
    page_image = Image.open(image_path).convert("RGB")
    text_boxes = _smart_sort_text_boxes(
        [box for box in detected if box["label"] == "text"],
        image_width=page_image.width,
        image_height=page_image.height
    )

    if not text_boxes:
        raise ValueError(f"No text boxes detected in {image_path}")

    ocr = MangaOCR()
    ocr_results = []

    for idx, box in enumerate(text_boxes):
        x1, y1, x2, y2 = [float(v) for v in box["bbox"]]
        crop = page_image.crop((max(0, x1), max(0, y1), min(page_image.width, x2), min(page_image.height, y2)))
        text = ocr.extract_text_from_image(crop)
        ocr_results.append(
            {
                "index": idx,
                "label": "text",
                "bbox": [x1, y1, x2, y2],
                "text": text,
                "confidence": box["confidence"],
            }
        )

    context_builder = ContextBuilder(output_dir / "contexts")
    context_data = context_builder.save_context_json(image_path, ocr_results, output_dir / "context.json")

    translator = OllamaVisionTranslator(model="qwen2.5vl:7b", target_language="English")
    translated = translator.translate_batch(context_data)

    with open(output_dir / "translation.json", "w", encoding="utf-8") as f:
        json.dump(translated, f, ensure_ascii=False, indent=2)

    return {
        "ocr_results": ocr_results,
        "context": context_data,
        "translation": translated,
    }


def main():
    parser = argparse.ArgumentParser(description="Run the manga translation prototype pipeline.")
    parser.add_argument("--image", default="open-mantra-dataset\images\balloon_dream\ja\010.jpg", help="Manga page image path.")
    parser.add_argument("--model", default="manga_panel_detector_fp32.pt", help="YOLO model path.")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    args = parser.parse_args()

    result = run_prototype(args.image, args.model, conf=args.conf, output_dir="outputs")
    for item in result["translation"]:
        print(f"{item['index']}: {item['source_text']} -> {item['translated_text']}")


if __name__ == "__main__":
    main()
