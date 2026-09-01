from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, List

import requests


class OllamaVisionTranslator:
    def __init__(self, model: str = "qwen2.5vl:7b", host: str = "http://localhost:11434", target_language: str = "English"):
        self.model = model
        self.host = host.rstrip("/")
        self.target_language = target_language

    def _encode_image(self, image_path: str | Path) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _chat(self, model: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = requests.post(f"{self.host}/api/chat", json={"model": model, **payload}, timeout=180)
        response.raise_for_status()
        return response.json()

    def translate(self, text: str, context: str | None = None, image_path: str | Path | None = None) -> str:
        prompt = f"""
You are translating manga dialogue into {self.target_language}.
Use the image and the context to resolve ambiguity and keep the dialogue natural.
Return only the translated line, no explanation.

Source text:
{text}

Context:
{context or 'No extra context provided.'}
""".strip()

        if image_path:
            image_b64 = self._encode_image(image_path)
            payload = {
                "messages": [{
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64],
                }],
                "stream": False,
            }
            try:
                data = self._chat(self.model, payload)
            except requests.RequestException:
                fallback_payload = {
                    "messages": [{
                        "role": "user",
                        "content": prompt,
                    }],
                    "stream": False,
                }
                data = self._chat("qwen2.5:7b", fallback_payload)
        else:
            payload = {
                "messages": [{
                    "role": "user",
                    "content": prompt,
                }],
                "stream": False,
            }
            data = self._chat(self.model, payload)

        try:
            return data["message"]["content"].strip()
        except (KeyError, TypeError, ValueError):
            return str(data).strip()

    def translate_batch(self, contexts: List[dict[str, Any]]) -> List[dict[str, Any]]:
        translated = []
        for item in contexts:
            image_path = item.get("crop_path")
            text = item.get("text", "")
            context = item.get("context_summary")
            translated_text = self.translate(text=text, context=context, image_path=image_path)
            translated.append(
                {
                    "index": item.get("index"),
                    "bbox": item.get("bbox"),
                    "source_text": text,
                    "translated_text": translated_text,
                    "context": context,
                    "crop_path": image_path,
                }
            )
        return translated


SimpleContextTranslator = OllamaVisionTranslator
