from pathlib import Path
from typing import List

from ultralytics import YOLO


class MangaDetector:
    def __init__(self, model_path: str | Path):
        self.model = YOLO(str(model_path))

    def detect(self, image_path: str | Path, conf: float = 0.25) -> List[dict]:
        results = self.model.predict(str(image_path), conf=conf)
        detections: List[dict] = []

        for box in results[0].boxes:
            cls = int(box.cls)
            conf_value = float(box.conf)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            label = "panel" if cls == 0 else "text"
            detections.append(
                {
                    "label": label,
                    "confidence": round(conf_value, 4),
                    "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                }
            )

        return detections
