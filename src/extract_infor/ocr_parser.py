from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from src.type.base import BoundingBox, OCRLine, OCRWord


def _normalized_bbox_to_pixels(
    bbox: dict[str, float], image_width: int, image_height: int
) -> BoundingBox:
    left = float(bbox["Left"]) * image_width
    top = float(bbox["Top"]) * image_height
    right = (float(bbox["Left"]) + float(bbox["Width"])) * image_width
    bottom = (float(bbox["Top"]) + float(bbox["Height"])) * image_height
    return BoundingBox(left=left, top=top, right=right, bottom=bottom)


def load_infographic_words_and_lines(
    image_path: str | Path,
    ocr_path: str | Path,
    page_number: int = 1,
) -> tuple[list[OCRWord], list[OCRLine]]:
    image_path = Path(image_path)
    ocr_path = Path(ocr_path)

    payload = json.loads(ocr_path.read_text(encoding="utf-8"))
    with Image.open(image_path) as image:
        image_width, image_height = image.size

    word_blocks = payload.get("WORD", [])
    line_blocks = payload.get("LINE", [])

    words: list[OCRWord] = []
    word_by_raw_id: dict[str, OCRWord] = {}
    for index, block in enumerate(word_blocks):
        geometry = block.get("Geometry", {})
        bbox = geometry.get("BoundingBox")
        if not bbox:
            continue
        word = OCRWord(
            id=f"p{page_number}_word_{index:04d}",
            page=page_number,
            text=str(block.get("Text", "")).strip(),
            bbox=_normalized_bbox_to_pixels(bbox, image_width, image_height),
            confidence=(
                float(block["Confidence"]) if block.get("Confidence") is not None else None
            ),
            metadata={"raw_id": block.get("Id", ""), "raw_block": block},
        )
        words.append(word)
        raw_id = str(block.get("Id", "")).strip()
        if raw_id:
            word_by_raw_id[raw_id] = word

    lines: list[OCRLine] = []
    for index, block in enumerate(line_blocks):
        geometry = block.get("Geometry", {})
        bbox = geometry.get("BoundingBox")
        if not bbox:
            continue

        child_ids: list[str] = []
        for relation in block.get("Relationships", []):
            if relation.get("Type") == "CHILD":
                child_ids.extend(str(item) for item in relation.get("Ids", []))

        line_words = [word_by_raw_id[word_id] for word_id in child_ids if word_id in word_by_raw_id]
        line = OCRLine(
            id=f"p{page_number}_line_{index:04d}",
            page=page_number,
            text=str(block.get("Text", "")).strip(),
            bbox=_normalized_bbox_to_pixels(bbox, image_width, image_height),
            words=line_words,
            metadata={"raw_id": block.get("Id", ""), "raw_block": block},
        )
        lines.append(line)

    lines.sort(key=lambda item: (item.page, item.bbox.top, item.bbox.left))
    words.sort(key=lambda item: (item.page, item.bbox.top, item.bbox.left))
    return words, lines
