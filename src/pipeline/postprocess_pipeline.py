import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inpainting.inpaint import inpaint_text_regions
from src.typesetting.typesetter import (
    build_editable_project,
    export_editor_html,
    render_translated_page,
    resolve_font_path,
    save_editable_project,
)


def run_postprocess(
    image_path: str | Path,
    translation_json: str | Path,
    output_dir: str | Path = "outputs/postprocess",
    font_path: str | Path | None = None,
    max_font_size: int = 34,
    text_color: str = "#000000",
    bubble_fill: str = "#FFFFFF",
    bubble_outline: str = "#141414",
    bubble_padding: float = 1.0,
    alignment: str = "center",
):
    image_path = Path(image_path)
    translation_json = Path(translation_json)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(translation_json, "r", encoding="utf-8") as f:
        translations = json.load(f)

    boxes = [item["bbox"] for item in translations]
    inpainted_path = output_dir / f"{image_path.stem}_inpainted.png"
    inpaint_text_regions(image_path, boxes, inpainted_path)

    rendered_path = output_dir / f"{image_path.stem}_typeset.png"
    render_translated_page(
        inpainted_path,
        translations,
        rendered_path,
        font_path,
        max_font_size,
        text_color,
        bubble_fill,
        bubble_outline,
        bubble_padding,
        alignment,
    )

    project = build_editable_project(
        image_path,
        inpainted_path,
        translations,
        font_path,
    )
    project_path = output_dir / "typesetter_project.json"
    editor_path = output_dir / "typesetter_editor.html"
    save_editable_project(project, project_path)
    export_editor_html(project, editor_path)

    return {
        "inpainted": str(inpainted_path),
        "typeset": str(rendered_path),
        "font": resolve_font_path(font_path) or "Automatic",
        "project": str(project_path),
        "editor": str(editor_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Apply post-processing: inpainting and typesetting for translated manga pages.")
    parser.add_argument("--image", required=True, help="Original manga page image")
    parser.add_argument("--translation", required=True, help="JSON translation output")
    parser.add_argument("--out-dir", default="outputs/postprocess", help="Directory for rendered outputs")
    parser.add_argument("--font", help="Path to a custom TTF or TTC font")
    args = parser.parse_args()

    result = run_postprocess(args.image, args.translation, args.out_dir, args.font)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
