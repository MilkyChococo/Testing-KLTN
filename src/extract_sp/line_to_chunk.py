from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.type.base import BoundingBox, OCRLine, OCRWord

from .line_to_line import LineRelation, build_text_relations_from_ocr


@dataclass(slots=True)
class TextChunk:
    id: str
    page: int
    text: str
    bbox: BoundingBox
    lines: list[OCRLine]


def _should_merge_to_next_line(
    source: OCRLine,
    target: OCRLine,
    max_chunk_gap_ratio: float,
    max_left_delta_ratio: float,
    min_horizontal_overlap_ratio: float,
) -> bool:
    avg_height = max((source.bbox.height + target.bbox.height) / 2.0, 1.0)
    vertical_gap_ratio = source.bbox.vertical_gap(target.bbox) / avg_height
    if vertical_gap_ratio > max_chunk_gap_ratio:
        return False

    page_width = max(source.bbox.right, target.bbox.right, 1.0)
    left_delta_ratio = abs(source.bbox.left - target.bbox.left) / page_width
    overlap_ratio = source.bbox.horizontal_overlap_ratio(target.bbox)

    return (
        left_delta_ratio <= max_left_delta_ratio
        or overlap_ratio >= min_horizontal_overlap_ratio
    )


def build_chunks_from_lines(
    lines: list[OCRLine],
    relations: list[LineRelation],
    max_chunk_gap_ratio: float = 1.6,
    max_left_delta_ratio: float = 0.06,
    min_horizontal_overlap_ratio: float = 0.35,
) -> list[TextChunk]:
    ordered_lines = sorted(
        lines, key=lambda line: (line.page, line.bbox.top, line.bbox.left)
    )
    line_by_id = {line.id: line for line in ordered_lines}
    next_line_map = {
        relation.source_id: relation.target_id
        for relation in relations
        if relation.relation == "next_line"
    }

    chunks: list[TextChunk] = []
    visited: set[str] = set()

    for line in ordered_lines:
        if line.id in visited:
            continue

        chunk_lines = [line]
        visited.add(line.id)
        current = line

        while current.id in next_line_map:
            next_id = next_line_map[current.id]
            if next_id in visited:
                break

            next_line = line_by_id.get(next_id)
            if next_line is None or next_line.page != current.page:
                break

            if not _should_merge_to_next_line(
                source=current,
                target=next_line,
                max_chunk_gap_ratio=max_chunk_gap_ratio,
                max_left_delta_ratio=max_left_delta_ratio,
                min_horizontal_overlap_ratio=min_horizontal_overlap_ratio,
            ):
                break

            chunk_lines.append(next_line)
            visited.add(next_id)
            current = next_line

        chunks.append(
            TextChunk(
                id=f"p{line.page}_chunk_{len(chunks):03d}",
                page=line.page,
                text="\n".join(item.text for item in chunk_lines if item.text).strip(),
                bbox=BoundingBox.merge(item.bbox for item in chunk_lines),
                lines=chunk_lines,
            )
        )

    return chunks


def build_text_chunks_from_ocr(
    ocr_path: str | Path,
    page_number: int = 1,
) -> tuple[list[OCRWord], list[OCRLine], list[LineRelation], list[TextChunk]]:
    words, lines, relations = build_text_relations_from_ocr(
        ocr_path=ocr_path,
        page_number=page_number,
    )
    chunks = build_chunks_from_lines(lines=lines, relations=relations)
    return words, lines, relations, chunks
