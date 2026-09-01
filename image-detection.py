import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image
from src.detection.yolo_detector import MangaDetector
from src.ocr.manga_ocr import MangaOCR


def main():
    parser = argparse.ArgumentParser(description="Run YOLO detection and OCR for a manga page.")
    parser.add_argument("--image", default="open-mantra-dataset\images\balloon_dream\ja\010.jpg", help="Path to manga page image.")
    parser.add_argument("--model", default=str(REPO_ROOT / "manga_panel_detector_fp32.pt"), help="Path to YOLO model file.")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    parser.add_argument("--save-json", action="store_true", help="Save OCR output to a JSON file.")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = REPO_ROOT / image_path

    detector = MangaDetector(args.model)
    detected = detector.detect(image_path, conf=args.conf)

    def reading_order_key(box):
        x1, y1, x2, y2 = box["bbox"]
        return (y1 + (y2 - y1) / 2, x1 + (x2 - x1) / 2)

    text_boxes = sorted(
        [box for box in detected if box["label"] == "text"],
        key=reading_order_key,
    )

    if not text_boxes:
        print("No text boxes detected.")
        return

    ocr = MangaOCR()
    page_image = Image.open(image_path).convert("RGB")
    ocr_results = []

    for idx, box in enumerate(text_boxes):
        x1, y1, x2, y2 = [float(v) for v in box["bbox"]]
        crop = page_image.crop((max(0, x1), max(0, y1), min(page_image.width, x2), min(page_image.height, y2)))
        text = ocr.extract_text_from_image(crop)
        result = {
            "index": idx,
            "label": "text",
            "bbox": [x1, y1, x2, y2],
            "text": text,
            "confidence": box["confidence"],
        }
        ocr_results.append(result)
        print(f"[{idx}] {text} | bbox={result['bbox']}")

    if args.save_json:
        output_path = image_path.with_suffix(".ocr.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"image": str(image_path), "results": ocr_results}, f, ensure_ascii=False, indent=2)
        print(f"Saved OCR JSON to {output_path}")


if __name__ == "__main__":
    main()
