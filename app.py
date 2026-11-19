import sys
import uuid
from pathlib import Path

import gradio as gr

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.postprocess_pipeline import run_postprocess
from src.pipeline.prototype_pipeline import run_prototype


def translate_page(
    image_path,
    confidence,
    postprocess,
    detector_model,
    translator_model,
    ollama_host,
):
    if image_path is None:
        raise gr.Error("Upload a manga page first.")

    detector_model = detector_model.strip() or str(REPO_ROOT / "manga_panel_detector_fp32.pt")
    detector_path = Path(detector_model)
    if not detector_path.is_absolute():
        detector_path = REPO_ROOT / detector_path
    if not detector_path.exists():
        raise gr.Error(f"Detector model not found: {detector_path}")

    try:
        output_root = REPO_ROOT / "outputs" / "ui" / uuid.uuid4().hex
        output_root.mkdir(parents=True, exist_ok=True)
        temp_root = output_root
        result = run_prototype(
            image_path=image_path,
            model_path=detector_path,
            conf=confidence,
            output_dir=temp_root,
            translator_model=translator_model.strip() or "qwen2.5vl:7b",
            ollama_host=ollama_host.strip() or "http://localhost:11434",
        )

        translations = result["translation"]
        rows = [
            {
                "Bubble": item["index"],
                "Source": item["source_text"],
                "Translation": item["translated_text"],
            }
            for item in sorted(translations, key=lambda item: item["index"])
        ]

        inpainted_path = None
        typeset_path = None
        if postprocess:
            postprocess_dir = temp_root / "postprocess"
            processed = run_postprocess(
                image_path,
                temp_root / "translation.json",
                postprocess_dir,
            )
            inpainted_path = processed["inpainted"]
            typeset_path = processed["typeset"]

        scene_summary = next(
            (item.get("scene_summary") for item in translations if item.get("scene_summary")),
            "No scene summary was generated.",
        )
        status = f"Translation complete: {len(translations)} bubble(s) processed."
        return typeset_path, inpainted_path, rows, scene_summary, status
    except Exception as exc:
        raise gr.Error(f"Translation failed: {type(exc).__name__}: {exc}") from exc


with gr.Blocks(title="Manga Translation") as demo:
    gr.Markdown("# Manga Translation")
    gr.Markdown("Upload a manga page and translate detected Japanese text using local OCR and Ollama.")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="filepath", label="Manga page")
            detector_model = gr.Textbox(
                value="manga_panel_detector_fp32.pt",
                label="Detector model path",
            )
            translator_model = gr.Textbox(
                value="qwen2.5vl:7b",
                label="Ollama vision model",
            )
            ollama_host = gr.Textbox(
                value="http://localhost:11434",
                label="Ollama host",
            )
            confidence = gr.Slider(
                minimum=0.05,
                maximum=0.95,
                value=0.25,
                step=0.05,
                label="Detection confidence",
            )
            postprocess = gr.Checkbox(
                value=True,
                label="Inpaint text and typeset translations",
            )
            translate_button = gr.Button("Translate page", variant="primary")

        with gr.Column(scale=2):
            typeset_output = gr.Image(label="Translated page")
            inpainted_output = gr.Image(label="Inpainted page")

    bubble_table = gr.Dataframe(
        headers=["Bubble", "Source", "Translation"],
        datatype=["number", "str", "str"],
        label="Translations",
        wrap=True,
    )
    scene_output = gr.Textbox(label="Scene summary", lines=5)
    status_output = gr.Textbox(label="Status")

    translate_button.click(
        fn=translate_page,
        inputs=[
            image_input,
            confidence,
            postprocess,
            detector_model,
            translator_model,
            ollama_host,
        ],
        outputs=[
            typeset_output,
            inpainted_output,
            bubble_table,
            scene_output,
            status_output,
        ],
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
