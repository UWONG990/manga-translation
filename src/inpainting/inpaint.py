from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def build_text_mask(image_shape, boxes: list[list[float]]) -> np.ndarray:
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    for box in boxes:
        if not box or len(box) < 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(width, int(x2))
        y2 = min(height, int(y2))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)

    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def inpaint_text_regions(image_path: str | Path, boxes: list[list[float]], output_path: str | Path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    mask = build_text_mask(image.shape, boxes)
    result = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), result)
    return str(output)
