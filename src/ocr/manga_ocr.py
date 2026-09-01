from __future__ import annotations

from pathlib import Path
from typing import Any, List

from PIL import Image
from manga_ocr import MangaOcr


class MangaOCR:
    def __init__(self) -> None:
        self.model = MangaOcr()

    def extract_text_from_image(self, image: Image.Image | str | Path) -> str:
        return self.model(image)

    def extract_text_from_boxes(self, image_path: str | Path, boxes: List[dict[str, Any]]) -> List[dict[str, Any]]:
        source = Image.open(image_path).convert("RGB")
        extracted: List[dict[str, Any]] = []

        for idx, box in enumerate(boxes):
            if str(box.get("label", "")).lower() != "text":
                continue

            x1, y1, x2, y2 = [float(v) for v in box["bbox"]]
            crop = source.crop((max(0, x1), max(0, y1), min(source.width, x2), min(source.height, y2)))

            if crop.width <= 2 or crop.height <= 2:
                continue

            text = self.model(crop)
            extracted.append(
                {
                    "index": idx,
                    "label": "text",
                    "bbox": [x1, y1, x2, y2],
                    "text": text,
                    "confidence": box.get("confidence"),
                }
            )

        return extracted
