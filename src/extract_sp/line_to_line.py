from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.type.base import OCRLine, OCRWord

from .word_to_line import group_words_to_lines, load_words_from_ocr


@dataclass(slots=True)
class LineRelation:
    source_id: str
    target_id: str
    relation: str
    score: float


def _next_line_score(source: OCRLine, candidate: OCRLine) -> float:
    source_box = source.bbox
    candidate_box = candidate.bbox
    avg_height = max((source_box.height + candidate_box.height) / 2.0, 1.0)
    vertical_gap = source_box.vertical_gap(candidate_box) / avg_height
    left_gap = abs(source_box.left - candidate_box.left) / max(
        source_box.width, candidate_box.width, 1.0
    )
    horizontal_penalty = 1.0 - min(
        1.0, source_box.horizontal_overlap_ratio(candidate_box)
    )
    return vertical_gap + (0.75 * left_gap) + (0.5 * horizontal_penalty)


def build_next_line_relations(
    lines: list[OCRLine],
    max_vertical_gap_ratio: float = 3.5,
    min_horizontal_overlap_ratio: float = 0.05,
) -> list[LineRelation]:
    relations: list[LineRelation] = []
    ordered_lines = sorted(
        lines, key=lambda line: (line.page, line.bbox.top, line.bbox.left)
    )

    for source in ordered_lines:
        best_candidate: OCRLine | None = None
        best_score: float | None = None

        for candidate in ordered_lines:
            if candidate.id == source.id or candidate.page != source.page:
                continue
            if candidate.bbox.top <= source.bbox.top:
                continue

            avg_height = max((source.bbox.height + candidate.bbox.height) / 2.0, 1.0)
            vertical_gap_ratio = source.bbox.vertical_gap(candidate.bbox) / avg_height
            if vertical_gap_ratio > max_vertical_gap_ratio:
                continue

            overlap_ratio = source.bbox.horizontal_overlap_ratio(candidate.bbox)
            if (
                overlap_ratio < min_horizontal_overlap_ratio
                and abs(source.bbox.left - candidate.bbox.left) > avg_height * 4
            ):
                continue

            score = _next_line_score(source, candidate)
            if best_score is None or score < best_score:
                best_score = score
                best_candidate = candidate

        if best_candidate is None or best_score is None:
            continue

        relations.append(
            LineRelation(
                source_id=source.id,
                target_id=best_candidate.id,
                relation="next_line",
                score=round(1.0 / (1.0 + best_score), 4),
            )
        )

    return relations


def build_text_relations_from_ocr(
    ocr_path: str | Path,
    page_number: int = 1,
) -> tuple[list[OCRWord], list[OCRLine], list[LineRelation]]:
    words = load_words_from_ocr(ocr_path=ocr_path, page_number=page_number)
    lines = group_words_to_lines(words)
    relations = build_next_line_relations(lines)
    return words, lines, relations
