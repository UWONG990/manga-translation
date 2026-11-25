from __future__ import annotations

import math
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image


def save_translation_results(
    translations: list[dict[str, Any]],
    output_path: str | Path,
) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(translations, temporary_file, ensure_ascii=False, indent=2)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()

    return str(output_path)


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return isinstance(value, float) and math.isnan(value)


def _bubble_index(value: Any) -> int:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Bubble number must be an integer: {value!r}") from exc
    if not numeric_value.is_integer():
        raise ValueError(f"Bubble number must be an integer: {value!r}")
    return int(numeric_value)


def _bbox_values(values: list[Any], image_width: int, image_height: int) -> list[float]:
    if any(_is_empty(value) for value in values):
        raise ValueError(
            "Each new bubble needs all four coordinates (X1, Y1, X2, Y2) "
            "in original-image pixels."
        )
    try:
        x1, y1, x2, y2 = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError("Bubble coordinates must be numbers.") from exc
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        raise ValueError("Bubble coordinates must be finite numbers.")
    if x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height:
        raise ValueError(
            f"Bubble coordinates must be within the page bounds "
            f"(width {image_width}, height {image_height})."
        )
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Bubble coordinates must have X2 > X1 and Y2 > Y1.")
    return [x1, y1, x2, y2]


def retranslate_table_rows(
    table_rows: Any,
    previous_items: list[dict[str, Any]],
    page_image: str,
    scene_summary: str,
    translator: Any,
    crop_dir: str | Path,
) -> tuple[list[list[Any]], list[dict[str, Any]]]:
    if hasattr(table_rows, "tolist"):
        table_rows = table_rows.tolist()
    if not isinstance(table_rows, (list, tuple)):
        raise ValueError("The translations table did not contain rows.")
    if table_rows and not isinstance(table_rows[0], (list, tuple)):
        table_rows = [table_rows]

    previous_by_index = {
        _bubble_index(item["index"]): item
        for item in previous_items
        if item.get("index") is not None
    }
    used_indices = set(previous_by_index)
    next_index = max(used_indices, default=-1) + 1
    pending: list[tuple[int, str, dict[str, Any] | None, list[float]]] = []

    with Image.open(page_image) as page:
        image_width, image_height = page.size

    for row in table_rows:
        cells = list(row) if isinstance(row, (list, tuple)) else []
        cells += [None] * (7 - len(cells))
        raw_index, raw_source = cells[:2]
        if _is_empty(raw_source):
            continue

        source_text = str(raw_source).strip()
        if _is_empty(raw_index):
            while next_index in used_indices:
                next_index += 1
            index = next_index
            next_index += 1
        else:
            index = _bubble_index(raw_index)
            if index in {item[0] for item in pending}:
                raise ValueError(f"Bubble {index} appears more than once in the table.")

        previous = previous_by_index.get(index)
        row_bbox = cells[3:7]
        if any(not _is_empty(value) for value in row_bbox):
            bbox = _bbox_values(row_bbox, image_width, image_height)
        elif previous and previous.get("bbox"):
            bbox = [float(value) for value in previous["bbox"][:4]]
        else:
            bbox = _bbox_values(row_bbox, image_width, image_height)

        used_indices.add(index)
        pending.append((index, source_text, previous, bbox))

    if not pending:
        raise ValueError("Add or enter at least one source-text row before retranslating.")

    sources_by_index = {index: text for index, text, _, _ in pending}
    translated_items = []

    crop_dir = Path(crop_dir)
    crop_dir.mkdir(parents=True, exist_ok=True)
    for index, source_text, previous, bbox in pending:
        page_path = (
            (previous or {}).get("page_image")
            or page_image
        )
        if not page_path:
            raise ValueError("The original manga page is unavailable; translate the page again.")

        context_parts = []
        if previous:
            prior_context = previous.get("context") or previous.get("context_summary")
            prior_source = previous.get("source_text")
            if prior_context and prior_source:
                prior_context = prior_context.replace(prior_source, source_text)
            if prior_context:
                context_parts.append(prior_context)
            context_parts.append(f"Bubble location on the page: {bbox}.")
        else:
            context_parts.append(
                "This is a manually added text line. Use the full-page image and scene "
                "summary to understand its context."
            )

        surrounding_text = [
            f"Bubble {other_index}: {other_text}"
            for other_index, other_text in sources_by_index.items()
            if other_index != index
        ]
        if surrounding_text:
            context_parts.append(
                "Other dialogue on this page:\n" + "\n".join(surrounding_text)
            )

        previous_bbox = (previous or {}).get("bbox")
        bbox_changed = previous_bbox and [
            float(value) for value in previous_bbox[:4]
        ] != bbox
        crop_path = None if bbox_changed else (previous or {}).get("crop_path")
        if not crop_path:
            with Image.open(page_path) as page:
                crop_path = crop_dir / f"text_{index}.png"
                page.crop(tuple(int(round(value)) for value in bbox)).save(crop_path)

        translated_text = translator.translate(
            text=source_text,
            context="\n".join(context_parts) or None,
            image_path=crop_path,
            page_image=page_path,
            scene_summary=(
                (previous or {}).get("scene_summary")
                or scene_summary
                or None
            ),
        )

        translated_item = dict(previous or {})
        translated_item.update(
            {
                "index": index,
                "bbox": bbox,
                "source_text": source_text,
                "translated_text": translated_text,
                "context": "\n".join(context_parts),
                "scene_summary": (
                    (previous or {}).get("scene_summary")
                    or scene_summary
                ),
                "page_image": page_path,
                "crop_path": str(crop_path),
            }
        )
        translated_items.append(translated_item)

    translated_items.sort(key=lambda item: item["index"])
    result_rows = [
        [
            item["index"],
            item["source_text"],
            item["translated_text"],
            *item["bbox"],
        ]
        for item in translated_items
    ]
    return result_rows, translated_items
