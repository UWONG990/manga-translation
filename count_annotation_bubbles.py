"""Count English-translated bubbles in the manga annotation file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def count_text_en(annotation: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Count total bubbles and non-empty text_en values per book title."""
    counts: dict[str, dict[str, int]] = {}

    for book in annotation:
        title = book.get("book_title", "unknown")
        if not isinstance(title, str) or not title.strip():
            title = "unknown"

        title_counts = counts.setdefault(
            title, {"translated_bubbles": 0, "total_bubbles": 0}
        )
        for page in book.get("pages", []):
            for bubble in page.get("text", []):
                title_counts["total_bubbles"] += 1
                text_en = bubble.get("text_en")
                if isinstance(text_en, str) and text_en.strip():
                    title_counts["translated_bubbles"] += 1

    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count non-empty text_en bubbles per manga title."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("open-mantra-dataset/annotation.json"),
        help="Annotation JSON file (default: open-mantra-dataset/annotation.json)",
    )
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as input_file:
        annotation = json.load(input_file)

    if not isinstance(annotation, list) or not all(
        isinstance(book, dict) for book in annotation
    ):
        raise ValueError("Input JSON must contain a list of book objects.")

    counts = count_text_en(annotation)
    print(f"{'Title/folder':<30} {'text_en':>10} {'Total bubbles':>14}")
    print("-" * 58)
    for title, title_counts in counts.items():
        print(
            f"{title:<30} "
            f"{title_counts['translated_bubbles']:>10} "
            f"{title_counts['total_bubbles']:>14}"
        )


if __name__ == "__main__":
    main()
