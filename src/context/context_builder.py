from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

from PIL import Image


@dataclass
class BubbleContext:
    index: int
    bbox: list[float]
    text: str
    page_image: str
    crop_path: str | None = None
    context_summary: str | None = None


class ContextBuilder:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_context(self, image_path: str | Path, ocr_results: List[dict[str, Any]]) -> List[BubbleContext]:
        image_path = Path(image_path)
        page_image = Image.open(image_path).convert("RGB")
        contexts: List[BubbleContext] = []

        for result in ocr_results:
            x1, y1, x2, y2 = [float(v) for v in result["bbox"]]
            crop = page_image.crop((max(0, x1), max(0, y1), min(page_image.width, x2), min(page_image.height, y2)))

            crop_name = f"text_{result['index']}.png"
            crop_path = self.output_dir / crop_name
            crop.save(crop_path)

            context_text = (
                f"This bubble is located at bbox={result['bbox']}. "
                f"The original content is: {result['text']}. "
                f"Use the comic page context and the surrounding page to resolve translation ambiguity."
            )

            contexts.append(
                BubbleContext(
                    index=result["index"],
                    bbox=result["bbox"],
                    text=result["text"],
                    page_image=str(image_path),
                    crop_path=str(crop_path),
                    context_summary=context_text,
                )
            )

        return contexts

    def save_context_json(self, image_path: str | Path, ocr_results: List[dict[str, Any]], out_path: str | Path):
        contexts = self.build_context(image_path, ocr_results)
        payload = [
            {
                "index": item.index,
                "bbox": item.bbox,
                "text": item.text,
                "page_image": item.page_image,
                "crop_path": item.crop_path,
                "context_summary": item.context_summary,
            }
            for item in contexts
        ]

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        return payload
