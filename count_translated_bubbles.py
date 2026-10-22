"""Count translated bubbles grouped by manga title."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _title_from_page_image(page_image: Any) -> str:
    if not isinstance(page_image, str) or not page_image.strip():
        return "unknown"

    parts = Path(page_image).parts
    lowered_parts = [part.lower() for part in parts]
    try:
        images_index = lowered_parts.index("images")
    except ValueError:
        return "unknown"

    if images_index + 1 < len(parts):
        return parts[images_index + 1]
    return "unknown"


def _title_for_record(record: dict[str, Any]) -> str:
    for key in ("book_title", "title"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return _title_from_page_image(record.get("page_image"))


def _is_translated(record: dict[str, Any]) -> bool:
    translated_text = record.get("translated_text")
    return isinstance(translated_text, str) and bool(translated_text.strip())


def count_translated_bubbles(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Return total and translated bubble counts grouped by manga title."""
    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"translated_bubbles": 0, "total_bubbles": 0}
    )

    for record in records:
        title = _title_for_record(record)
        counts[title]["total_bubbles"] += 1
        if _is_translated(record):
            counts[title]["translated_bubbles"] += 1

    return dict(sorted(counts.items()))


def _load_records(input_path: Path) -> list[dict[str, Any]]:
    with input_path.open("r", encoding="utf-8") as input_file:
        data = json.load(input_file)

    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError("Input JSON must contain a list of bubble objects.")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count translated bubbles grouped by manga title."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("outputs/translation.json"),
        help="Translation JSON file (default: outputs/translation.json)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Print results as JSON instead of a table.",
    )
    args = parser.parse_args()

    counts = count_translated_bubbles(_load_records(args.input))
    if args.as_json:
        print(json.dumps(counts, ensure_ascii=False, indent=2))
        return

    print(f"{'Title':<30} {'Translated':>10} {'Total':>10}")
    print("-" * 54)
    for title, title_counts in counts.items():
        print(
            f"{title:<30} "
            f"{title_counts['translated_bubbles']:>10} "
            f"{title_counts['total_bubbles']:>10}"
        )


if __name__ == "__main__":
    main()
