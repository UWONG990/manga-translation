import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.detection.yolo_detector import MangaDetector


def main():
    parser = argparse.ArgumentParser(description="Run YOLO-based detection for manga pages.")
    parser.add_argument("--image", required=True, help="Path to the manga page image.")
    parser.add_argument(
        "--model",
        default=str(REPO_ROOT / "manga_panel_detector_fp32.pt"),
        help="Path to YOLO model file.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = REPO_ROOT / image_path

    detector = MangaDetector(args.model)
    detections = detector.detect(image_path, conf=args.conf)

    print(f"Detected {len(detections)} objects in {image_path}")
    for item in detections:
        print(item)


if __name__ == "__main__":
    main()
