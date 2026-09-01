from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _load_font(size: int) -> ImageFont.ImageFont:
    candidate_paths = [
        "C:/Windows/Fonts/msgothic.ttc",
        "C:/Windows/Fonts/meiryo.ttc",
        "C:/Windows/Fonts/mingliu.ttc",
        "C:/Windows/Fonts/NotoSansCJK-Regular.ttc",
        "C:/Windows/Fonts/simsun.ttc",
    ]
    for path in candidate_paths:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _fit_font(draw: ImageDraw.ImageDraw, text: str, box: tuple[float, float, float, float]) -> ImageFont.ImageFont:
    left, top, right, bottom = box
    max_w = max(1.0, right - left)
    max_h = max(1.0, bottom - top)

    for size in range(30, 8, -1):
        font = _load_font(size)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        if text_w <= max_w * 0.9 and text_h <= max_h * 0.9:
            return font
    return _load_font(10)


def render_translated_page(page_image_path: str | Path, translated_items: list[dict], output_path: str | Path):
    page_image = Image.open(page_image_path).convert("RGBA")
    canvas = Image.new("RGBA", page_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    for item in sorted(translated_items, key=lambda x: x.get("index", 0)):
        bbox = item.get("bbox")
        translated = item.get("translated_text", "")
        if not bbox or len(bbox) < 4 or not translated:
            continue

        x1, y1, x2, y2 = [float(v) for v in bbox]
        draw.rounded_rectangle([x1, y1, x2, y2], radius=max(8, min(18, (y2 - y1) / 6)), fill=(255, 255, 255, 120))

        font = _fit_font(draw, translated, (x1, y1, x2, y2))
        bbox_box = draw.textbbox((0, 0), translated, font=font)
        text_w = bbox_box[2] - bbox_box[0]
        text_h = bbox_box[3] - bbox_box[1]

        x = x1 + max(0, (x2 - x1 - text_w) / 2)
        y = y1 + max(0, (y2 - y1 - text_h) / 2)
        draw.text((x, y), translated, fill=(0, 0, 0, 255), font=font)

    result = Image.alpha_composite(page_image, canvas).convert("RGB")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.save(output)
    return str(output)
