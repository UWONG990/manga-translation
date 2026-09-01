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

    def summarize_scene(self, page_image: str | Path, page_texts: list[str] | None = None) -> str:
        if not page_image:
            return "No page scene available."

        prompt = """
You are analyzing a manga page before translation.
Write a short scene summary in 2-4 sentences.
Include the setting, the emotional mood, who seems to be speaking, and the situation.
Only describe the scene; do not translate the dialogue yet.
""".strip()

        if page_texts:
            prompt += "\n\nDetected text on the page:\n" + "\n".join(page_texts[:10])

        payload = {
            "messages": [{
                "role": "user",
                "content": prompt,
                "images": [self._encode_image(page_image)],
            }],
            "stream": False,
        }

        try:
            data = self._chat(self.model, payload)
        except requests.RequestException:
            data = self._chat("qwen2.5:7b", {"messages": [{"role": "user", "content": prompt}], "stream": False})

        try:
            return data["message"]["content"].strip()
        except (KeyError, TypeError, ValueError):
            return str(data).strip()

    def translate(self, text: str, context: str | None = None, image_path: str | Path | None = None, page_image: str | Path | None = None, scene_summary: str | None = None) -> str:
        prompt = f"""
You are translating manga dialogue into {self.target_language}.
Use the full page image, the bubble image, and the scene summary to resolve ambiguity and keep the dialogue natural.
Return only the translated line, no explanation.

Source text:
{text}

Scene summary:
{scene_summary or 'No page scene summary available.'}

Context:
{context or 'No extra context provided.'}
""".strip()

        image_b64s: list[str] = []
        if page_image:
            image_b64s.append(self._encode_image(page_image))
        if image_path:
            image_b64s.append(self._encode_image(image_path))

        if image_b64s:
            payload = {
                "messages": [{
                    "role": "user",
                    "content": prompt,
                    "images": image_b64s,
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
        page_summary_by_image: dict[str, str] = {}

        for item in contexts:
            page_image = item.get("page_image")
            if page_image and page_image not in page_summary_by_image:
                page_summary_by_image[page_image] = self.summarize_scene(
                    page_image,
                    [entry.get("text", "") for entry in contexts if entry.get("page_image") == page_image],
                )

        for item in contexts:
            image_path = item.get("crop_path")
            page_image = item.get("page_image")
            text = item.get("text", "")
            context = item.get("context_summary")
            scene_summary = item.get("scene_summary") or page_summary_by_image.get(page_image)
            translated_text = self.translate(
                text=text,
                context=context,
                image_path=image_path,
                page_image=page_image,
                scene_summary=scene_summary,
            )
            translated.append(
                {
                    "index": item.get("index"),
                    "bbox": item.get("bbox"),
                    "source_text": text,
                    "translated_text": translated_text,
                    "context": context,
                    "scene_summary": scene_summary,
                    "crop_path": image_path,
                    "page_image": page_image,
                }
            )
        return translated


SimpleContextTranslator = OllamaVisionTranslator
