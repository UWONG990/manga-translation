import sys
import uuid
import socket
from pathlib import Path
import os

import gradio as gr

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.postprocess_pipeline import run_postprocess
from src.pipeline.prototype_pipeline import run_prototype
from src.typesetting.typesetter import render_translated_page

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
                selected_font,
            )
            inpainted_path = processed["inpainted"]
            typeset_path = processed["typeset"]

        scene_summary = next(
            (item.get("scene_summary") for item in translations if item.get("scene_summary")),
            "No scene summary was generated.",
        )
        used_font = processed.get("font", "Not used") if postprocess else "Not used"
        status = (
            f"Translation complete: {len(translations)} bubble(s) processed. "
            f"Font: {used_font}"
        )
        state = {
            "inpainted_path": inpainted_path,
            "translation_data": translations,
            "output_dir": str(temp_root),
        }
        return typeset_path, inpainted_path, rows, scene_summary, status, state
    except Exception as exc:
        raise gr.Error(f"Translation failed: {type(exc).__name__}: {exc}") from exc


def apply_formatting(
    bubble_rows,
    state,
    font_choice,
    custom_font,
    max_font_size,
    text_color,
    bubble_fill,
    bubble_outline,
    bubble_padding,
    alignment,
):
    if not state or not state.get("inpainted_path"):
        raise gr.Error("Translate a page before applying formatting.")

    if isinstance(custom_font, dict):
        custom_font = custom_font.get("name") or custom_font.get("path")
    elif isinstance(custom_font, (list, tuple)):
        custom_font = custom_font[0] if custom_font else None
    selected_font = custom_font if custom_font else FONT_CHOICES.get(font_choice)
    if selected_font and not Path(selected_font).exists():
        raise gr.Error(f"Selected font file was not found: {selected_font}")

    translations = list(state["translation_data"])
    edited_by_index = {}
    for row in bubble_rows or []:
        if isinstance(row, dict):
            bubble_index = row.get("Bubble")
            translated_text = row.get("Translation")
        elif isinstance(row, (list, tuple)) and len(row) >= 3:
            bubble_index = row[0]
            translated_text = row[2]
        else:
            continue
        if bubble_index is not None:
            edited_by_index[int(bubble_index)] = translated_text

    for item in translations:
        if item["index"] in edited_by_index:
            item["translated_text"] = edited_by_index[item["index"]]

    output_dir = Path(state["output_dir"])
    rendered_path = output_dir / "user_typeset.png"
    render_translated_page(
        state["inpainted_path"],
        translations,
        rendered_path,
        selected_font,
        int(max_font_size),
        text_color,
        bubble_fill,
        bubble_outline,
        float(bubble_padding),
        alignment,
    )

    state["translation_data"] = translations
    font_label = resolve_font_label(selected_font)
    return str(rendered_path), state, f"Formatting applied with font: {font_label}"


def resolve_font_label(font_path):
    if not font_path:
        return "Automatic"
    for name, path in FONT_CHOICES.items():
        if path == font_path:
            return name
    return str(font_path)


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
        interactive=True,
    )
    translation_state = gr.State()

    gr.Markdown("## Edit and typeset")
    with gr.Row():
        max_font_size = gr.Slider(
            minimum=8,
            maximum=34,
            value=34,
            step=1,
            label="Maximum font size",
        )
        text_color = gr.ColorPicker(value="#000000", label="Text color")
        bubble_fill = gr.ColorPicker(value="#FFFFFF", label="Bubble fill")
        bubble_outline = gr.ColorPicker(value="#141414", label="Bubble outline")
    with gr.Row():
        bubble_padding = gr.Slider(
            minimum=0.5,
            maximum=1.5,
            value=1.0,
            step=0.1,
            label="Bubble padding",
        )
        alignment = gr.Radio(
            choices=["left", "center", "right"],
            value="center",
            label="Text alignment",
        )
        apply_formatting_button = gr.Button("Apply text and formatting", variant="primary")

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
            typeset_output,
            inpainted_output,
            bubble_table,
            scene_output,
            status_output,
            translation_state,
        ],
    )

    apply_formatting_button.click(
        fn=apply_formatting,
        inputs=[
            bubble_table,
            translation_state,
            font_choice,
            custom_font,
            max_font_size,
            text_color,
            bubble_fill,
            bubble_outline,
            bubble_padding,
            alignment,
        ],
        outputs=[typeset_output, translation_state, status_output],
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
