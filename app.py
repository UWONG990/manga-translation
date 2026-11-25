import sys
import uuid
import socket
from pathlib import Path
import os

import gradio as gr

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.prototype_pipeline import run_prototype
from src.inpainting.inpaint import inpaint_text_regions
from src.translation.retranslation import (
    retranslate_table_rows,
    save_translation_results,
)
from src.translation.translator import OllamaVisionTranslator
from src.typesetting.typesetter import (
    build_editable_project,
    export_editor_html,
    render_translated_page,
)

FONT_CHOICES = {
    "Automatic": None,
    "MS Gothic": "C:/Windows/Fonts/msgothic.ttc",
    "Arial": "C:/Windows/Fonts/arial.ttf",
    "Comic Sans": "C:/Windows/Fonts/comic.ttf",
    "Calibri": "C:/Windows/Fonts/calibri.ttf",
    "Consolas": "C:/Windows/Fonts/consola.ttf",
    "Segoe UI": "C:/Windows/Fonts/segoeui.ttf",
}
FONT_CHOICES = {
    name: path for name, path in FONT_CHOICES.items()
    if path is None or os.path.exists(path)
}


def translate_page(
    image_path,
    confidence,
    postprocess,
    detector_model,
    translator_model,
    ollama_host,
    font_choice,
    custom_font,
):
    if image_path is None:
        raise gr.Error("Upload a manga page first.")

    detector_model = detector_model.strip() or str(REPO_ROOT / "manga_panel_detector_fp32.pt")
    detector_path = Path(detector_model)
    if not detector_path.is_absolute():
        detector_path = REPO_ROOT / detector_path
    if not detector_path.exists():
        raise gr.Error(f"Detector model not found: {detector_path}")

    if isinstance(custom_font, dict):
        custom_font = custom_font.get("name") or custom_font.get("path")
    elif isinstance(custom_font, (list, tuple)):
        custom_font = custom_font[0] if custom_font else None

    selected_font = custom_font if custom_font else FONT_CHOICES.get(font_choice)
    if selected_font and not Path(selected_font).exists():
        raise gr.Error(f"Selected font file was not found: {selected_font}")

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
            [
                item["index"],
                item["source_text"],
                item["translated_text"],
                *item["bbox"],
            ]
            for item in sorted(translations, key=lambda item: item["index"])
        ]

        postprocess_dir = temp_root / "postprocess"
        postprocess_dir.mkdir(parents=True, exist_ok=True)
        inpainted_path = postprocess_dir / f"{Path(image_path).stem}_inpainted.png"
        if postprocess:
            inpaint_text_regions(
                image_path,
                [item["bbox"] for item in translations],
                inpainted_path,
            )
        else:
            inpainted_path = Path(image_path)

        project = build_editable_project(
            image_path,
            inpainted_path,
            translations,
            selected_font,
        )
        editor_path = export_editor_html(
            project,
            temp_root / "typesetter_editor.html",
        )

        scene_summary = next(
            (item.get("scene_summary") for item in translations if item.get("scene_summary")),
            "No scene summary was generated.",
        )
        status = (
            f"Translation complete: {len(translations)} bubble(s) processed. "
            "Interactive editor generated."
        )
        state = {
            "inpainted_path": inpainted_path,
            "translation_data": translations,
            "output_dir": str(temp_root),
            "image_path": str(image_path),
            "scene_summary": scene_summary,
            "font_path": selected_font,
            "postprocess": postprocess,
        }
        editor_embed = (
            f'<iframe src="/file={editor_path}" '
            'style="width:100%;height:520px;border:0;border-radius:8px;background:#fff;"></iframe>'
        )
        return (
            str(inpainted_path),
            rows,
            scene_summary,
            status,
            state,
            editor_embed,
        )
    except Exception as exc:
        raise gr.Error(f"Translation failed: {type(exc).__name__}: {exc}") from exc


def retranslate_page_rows(table_rows, state, translator_model, ollama_host):
    if not state or not state.get("translation_data"):
        raise gr.Error("Translate a page before using Retranslate.")

    try:
        translator = OllamaVisionTranslator(
            model=translator_model.strip() or "qwen2.5vl:7b",
            host=ollama_host.strip() or "http://localhost:11434",
        )
        rows, translations = retranslate_table_rows(
            table_rows,
            state["translation_data"],
            state.get("image_path", ""),
            state.get("scene_summary", ""),
            translator,
            Path(state["output_dir"]) / "contexts",
        )

        updated_state = dict(state)
        updated_state["translation_data"] = translations
        output_dir = Path(state["output_dir"])
        if state.get("postprocess"):
            inpainted_path = inpaint_text_regions(
                state["image_path"],
                [item["bbox"] for item in translations],
                output_dir / f"inpainted_{uuid.uuid4().hex}.png",
            )
        else:
            inpainted_path = state["image_path"]
        updated_state["inpainted_path"] = inpainted_path
        project = build_editable_project(
            state["image_path"],
            inpainted_path,
            translations,
            state.get("font_path"),
        )
        editor_path = export_editor_html(
            project,
            output_dir / f"typesetter_editor_{uuid.uuid4().hex}.html",
        )
        save_translation_results(
            translations,
            output_dir / "translation.json",
        )
        editor_embed = (
            f'<iframe src="/file={editor_path}" '
            'style="width:100%;height:520px;border:0;border-radius:8px;background:#fff;"></iframe>'
        )
        status = (
            f"Re-translated {len(translations)} row(s) with "
            f"{translator.model}. YOLO and OCR were not rerun. "
            "The typesetter includes all rows with their specified positions."
        )
        return str(inpainted_path), rows, updated_state, status, editor_embed
    except Exception as exc:
        raise gr.Error(f"Retranslation failed: {type(exc).__name__}: {exc}") from exc


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
            font_choice = gr.Dropdown(
                choices=list(FONT_CHOICES),
                value="Automatic",
                label="Typesetting font",
            )
            custom_font = gr.File(
                file_types=[".ttf", ".otf", ".ttc"],
                type="file",
                label="Custom font (overrides selection)",
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
                label="Inpaint original text",
            )
            translate_button = gr.Button("Translate page", variant="primary")

        with gr.Column(scale=2):
            inpainted_output = gr.Image(label="Inpainted page")
            editor_link = gr.HTML(
                "<p>The interactive editor appears here after translation.</p>"
            )

    bubble_table = gr.Dataframe(
        headers=["Bubble", "Source", "Translation", "X1", "Y1", "X2", "Y2"],
        datatype=["number", "str", "str", "number", "number", "number", "number"],
        label="Detected translations",
        wrap=True,
        interactive=True,
        row_count=(1, "dynamic"),
        col_count=(7, "fixed"),
        type="array",
    )
    gr.Markdown(
        "Edit the Source cells or add rows with the table's + control, then click "
        "**Retranslate rows**. For new rows, enter the bubble rectangle as X1, Y1, "
        "X2, Y2 in pixels from the top-left of the original page. These coordinates "
        "are used to create its crop, inpaint its original text, and place it in the typesetter."
    )
    translation_state = gr.State()
    retranslate_button = gr.Button("Retranslate rows")

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
            font_choice,
            custom_font,
        ],
        outputs=[
            inpainted_output,
            bubble_table,
            scene_output,
            status_output,
            translation_state,
            editor_link,
        ],
    )
    retranslate_button.click(
        fn=retranslate_page_rows,
        inputs=[
            bubble_table,
            translation_state,
            translator_model,
            ollama_host,
        ],
        outputs=[
            inpainted_output,
            bubble_table,
            translation_state,
            status_output,
            editor_link,
        ],
    )


def find_available_port(start_port: int = 7860, end_port: int = 7879) -> int:
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("No available Gradio port found between 7860 and 7879.")


if __name__ == "__main__":
    port = find_available_port()
    demo.launch(server_name="127.0.0.1", server_port=port, share=False)
