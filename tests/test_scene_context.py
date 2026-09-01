import base64
import unittest

from src.translation.translator import OllamaVisionTranslator


class SceneContextTranslatorTest(unittest.TestCase):
    def test_translate_batch_uses_page_and_bubble_images(self):
        captured = {}

        def fake_chat(self, model, payload):
            captured["model"] = model
            captured["payload"] = payload
            return {"message": {"content": "translated"}}

        def fake_encode(self, image_path):
            return base64.b64encode(b"fake-image-bytes").decode("utf-8")

        translator = OllamaVisionTranslator(model="qwen2.5vl:7b")
        translator._chat = fake_chat.__get__(translator, OllamaVisionTranslator)
        translator._encode_image = fake_encode.__get__(translator, OllamaVisionTranslator)

        contexts = [
            {
                "index": 0,
                "bbox": [0, 0, 10, 10],
                "text": "hello",
                "context_summary": "some context",
                "crop_path": "crop.png",
                "page_image": "page.png",
            }
        ]

        result = translator.translate_batch(contexts)

        self.assertEqual(result[0]["translated_text"], "translated")
        self.assertEqual(captured["payload"]["messages"][0]["images"][0], base64.b64encode(b"fake-image-bytes").decode("utf-8"))
        self.assertEqual(len(captured["payload"]["messages"][0]["images"]), 2)


if __name__ == "__main__":
    unittest.main()
