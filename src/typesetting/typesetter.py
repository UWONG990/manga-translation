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


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: float) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return [""]

    words = stripped.split()
    if not words:
        return [stripped]

    lines: list[str] = []
    current = words[0]

    for word in words[1:]:
        candidate = f"{current} {word}"
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word

    lines.append(current)

    # Fallback for long single words / punctuation-heavy strings.
    if len(lines) == 1 and draw.textbbox((0, 0), lines[0], font=font)[2] > max_width:
        chars = []
        current = ""
        for ch in stripped:
            candidate = current + ch
            if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
                current = candidate
            else:
                if current:
                    chars.append(current)
                current = ch
        if current:
            chars.append(current)
        return chars

    return lines


def _fit_font(draw: ImageDraw.ImageDraw, text: str, box: tuple[float, float, float, float]) -> ImageFont.ImageFont:
    left, top, right, bottom = box
    max_w = max(12.0, right - left)
    max_h = max(12.0, bottom - top)

    for size in range(34, 7, -1):
        font = _load_font(size)
        wrapped = _wrap_text(draw, text, font, max_w * 0.9)
        line_height = draw.textbbox((0, 0), "A", font=font)[3]
        total_h = len(wrapped) * line_height * 1.25
        if total_h <= max_h * 0.9:
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
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        if x2 <= x1 or y2 <= y1:
            continue

        pad = max(8.0, min(18.0, (y2 - y1) * 0.12))
        bubble_x1 = max(0.0, x1 - pad * 0.6)
        bubble_y1 = max(0.0, y1 - pad * 0.4)
        bubble_x2 = max(0.0, x2 + pad * 0.6)
        bubble_y2 = max(0.0, y2 + pad * 0.4)

        if bubble_x2 <= bubble_x1 or bubble_y2 <= bubble_y1:
            continue

        radius = max(6, min(26, min((bubble_x2 - bubble_x1) / 4, (bubble_y2 - bubble_y1) / 4)))
        draw.rounded_rectangle(
            [bubble_x1, bubble_y1, bubble_x2, bubble_y2],
            radius=int(radius),
            fill=(255, 255, 255, 180),
            outline=(20, 20, 20, 220),
            width=2,
        )

        inner_x1 = bubble_x1 + pad * 0.7
        inner_y1 = bubble_y1 + pad * 0.6
        inner_x2 = bubble_x2 - pad * 0.7
        inner_y2 = bubble_y2 - pad * 0.6
        font = _fit_font(draw, translated, (inner_x1, inner_y1, inner_x2, inner_y2))
        lines = _wrap_text(draw, translated, font, inner_x2 - inner_x1)
        line_height = draw.textbbox((0, 0), "A", font=font)[3]
        total_h = len(lines) * line_height * 1.25
        start_y = inner_y1 + max(0, (inner_y2 - inner_y1 - total_h) / 2)

        for i, line in enumerate(lines):
            line_w = draw.textbbox((0, 0), line, font=font)[2]
            x = inner_x1 + max(0, (inner_x2 - inner_x1 - line_w) / 2)
            y = start_y + i * line_height * 1.25
            draw.text((x, y), line, fill=(0, 0, 0, 255), font=font, stroke_width=1, stroke_fill=(255, 255, 255, 180))

    result = Image.alpha_composite(page_image, canvas).convert("RGB")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.save(output)
    return str(output)
