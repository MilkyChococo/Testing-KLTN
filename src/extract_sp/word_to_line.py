from __future__ import annotations

import json
from pathlib import Path

from src.type.base import BoundingBox, OCRLine, OCRWord


def load_words_from_ocr(ocr_path: str | Path, page_number: int = 1) -> list[OCRWord]:
    path = Path(ocr_path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    words: list[OCRWord] = []
    for page in payload.get("recognitionResults", []):
        page_idx = int(page.get("page", 0))
        if page_idx != page_number:
            continue

        for line_idx, line in enumerate(page.get("lines", [])):
            for word_idx, word in enumerate(line.get("words", [])):
                text = word.get("text", "").strip()
                if not text:
                    continue
                words.append(
                    OCRWord(
                        id=f"p{page_idx}_l{line_idx:03d}_w{word_idx:03d}",
                        page=page_idx,
                        text=text,
                        bbox=BoundingBox.from_polygon(word["boundingBox"]),
                    )
                )
    return words


def _line_accepts_word(
    line_words: list[OCRWord],
    candidate: OCRWord,
    center_y_threshold_ratio: float,
    min_vertical_overlap_ratio: float,
) -> bool:
    if not line_words:
        return False

    line_box = BoundingBox.merge(word.bbox for word in line_words)
    avg_height = (
        sum(word.bbox.height for word in line_words) / len(line_words)
        if line_words
        else candidate.bbox.height
    )
    center_y_gap = abs(candidate.bbox.center_y - line_box.center_y)
    max_center_y_gap = max(avg_height, candidate.bbox.height) * center_y_threshold_ratio

    if center_y_gap > max_center_y_gap:
        return False

    overlap_ratio = candidate.bbox.vertical_overlap_ratio(line_box)
    return overlap_ratio >= min_vertical_overlap_ratio


def group_words_to_lines(
    words: list[OCRWord],
    center_y_threshold_ratio: float = 0.6,
    min_vertical_overlap_ratio: float = 0.25,
) -> list[OCRLine]:
    if not words:
        return []

    grouped: list[list[OCRWord]] = []
    ordered_words = sorted(
        words, key=lambda word: (word.page, word.bbox.center_y, word.bbox.left)
    )

    for word in ordered_words:
        matched_group: list[OCRWord] | None = None
        best_gap: float | None = None

        for group in grouped:
            if group[0].page != word.page:
                continue
            if not _line_accepts_word(
                group,
                word,
                center_y_threshold_ratio=center_y_threshold_ratio,
                min_vertical_overlap_ratio=min_vertical_overlap_ratio,
            ):
                continue

            group_box = BoundingBox.merge(item.bbox for item in group)
            center_gap = abs(word.bbox.center_y - group_box.center_y)
            if best_gap is None or center_gap < best_gap:
                best_gap = center_gap
                matched_group = group

        if matched_group is None:
            grouped.append([word])
        else:
            matched_group.append(word)

    lines: list[OCRLine] = []
    grouped.sort(
        key=lambda group: (
            group[0].page,
            BoundingBox.merge(item.bbox for item in group).top,
            BoundingBox.merge(item.bbox for item in group).left,
        )
    )

    for line_idx, group in enumerate(grouped):
        sorted_words = sorted(group, key=lambda word: word.bbox.left)
        bbox = BoundingBox.merge(word.bbox for word in sorted_words)
        text = " ".join(word.text for word in sorted_words).strip()
        page = sorted_words[0].page
        lines.append(
            OCRLine(
                id=f"p{page}_line_{line_idx:03d}",
                page=page,
                text=text,
                bbox=bbox,
                words=sorted_words,
                metadata={"source": "grouped_from_words"},
            )
        )

    return lines
