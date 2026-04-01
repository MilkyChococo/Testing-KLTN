from __future__ import annotations

import math
from dataclasses import dataclass

from src.extract_sp.graph_types import GraphRelation
from src.type.base import OCRLine

from .line_to_chunk import InfographicChunk


@dataclass(slots=True)
class InfographicLineRelation:
    source_id: str
    target_id: str
    relation: str
    score: float


@dataclass(slots=True, frozen=True)
class InfographicRelationConfig:
    next_line_max_vertical_gap_ratio: float = 2.0
    next_line_max_left_delta_ratio: float = 0.10
    next_line_min_horizontal_overlap_ratio: float = 0.10
    near_line_max_center_distance_ratio: float = 0.14
    near_line_top_k: int = 1
    next_chunk_max_vertical_gap_ratio: float = 3.0
    next_chunk_max_left_delta_ratio: float = 0.18
    next_chunk_min_horizontal_overlap_ratio: float = 0.05
    near_chunk_max_center_distance_ratio: float = 0.30
    near_chunk_top_k: int = 1


DEFAULT_INFOGRAPHIC_RELATION_CONFIG = InfographicRelationConfig()


def _page_extent_from_lines(lines: list[OCRLine]) -> tuple[float, float]:
    if not lines:
        return 1.0, 1.0
    page_width = max(line.bbox.right for line in lines)
    page_height = max(line.bbox.bottom for line in lines)
    return max(page_width, 1.0), max(page_height, 1.0)


def _page_extent_from_chunks(chunks: list[InfographicChunk]) -> tuple[float, float]:
    if not chunks:
        return 1.0, 1.0
    page_width = max(chunk.bbox.right for chunk in chunks)
    page_height = max(chunk.bbox.bottom for chunk in chunks)
    return max(page_width, 1.0), max(page_height, 1.0)


def _height_similarity(source_height: float, target_height: float) -> float:
    return min(source_height, target_height) / max(max(source_height, target_height), 1.0)


def build_next_line_relations(
    lines: list[OCRLine],
    config: InfographicRelationConfig = DEFAULT_INFOGRAPHIC_RELATION_CONFIG,
) -> list[InfographicLineRelation]:
    page_width, _ = _page_extent_from_lines(lines)
    ordered_lines = sorted(lines, key=lambda item: (item.page, item.bbox.top, item.bbox.left))
    relations: list[InfographicLineRelation] = []

    for source in ordered_lines:
        best_candidate: OCRLine | None = None
        best_score = 0.0
        for candidate in ordered_lines:
            if candidate.id == source.id or candidate.page != source.page:
                continue
            if candidate.bbox.top <= source.bbox.top:
                continue

            avg_height = max((source.bbox.height + candidate.bbox.height) / 2.0, 1.0)
            vertical_gap_ratio = source.bbox.vertical_gap(candidate.bbox) / avg_height
            if vertical_gap_ratio > config.next_line_max_vertical_gap_ratio:
                continue

            overlap_ratio = source.bbox.horizontal_overlap_ratio(candidate.bbox)
            left_delta_ratio = abs(source.bbox.left - candidate.bbox.left) / page_width
            center_delta_ratio = abs(source.bbox.center_x - candidate.bbox.center_x) / page_width
            aligned = (
                overlap_ratio >= config.next_line_min_horizontal_overlap_ratio
                or left_delta_ratio <= config.next_line_max_left_delta_ratio
                or center_delta_ratio <= config.next_line_max_left_delta_ratio
            )
            if not aligned:
                continue

            gap_score = max(
                0.0,
                1.0 - (vertical_gap_ratio / max(config.next_line_max_vertical_gap_ratio, 1e-6)),
            )
            align_score = max(
                overlap_ratio,
                max(0.0, 1.0 - (left_delta_ratio / max(config.next_line_max_left_delta_ratio, 1e-6))),
                max(0.0, 1.0 - (center_delta_ratio / max(config.next_line_max_left_delta_ratio, 1e-6))),
            )
            score = (
                0.45 * gap_score
                + 0.35 * align_score
                + 0.20 * _height_similarity(source.bbox.height, candidate.bbox.height)
            )
            if score > best_score:
                best_candidate = candidate
                best_score = score

        if best_candidate is not None:
            relations.append(
                InfographicLineRelation(
                    source_id=source.id,
                    target_id=best_candidate.id,
                    relation="next_line",
                    score=round(best_score, 4),
                )
            )
    return relations


def build_line_chunk_membership_relations(
    chunks: list[InfographicChunk],
) -> list[GraphRelation]:
    relations: list[GraphRelation] = []
    for chunk in chunks:
        for line in chunk.lines:
            relations.append(
                GraphRelation(
                    source_id=line.id,
                    target_id=chunk.id,
                    relation="line_to_chunk",
                    score=1.0,
                )
            )
            relations.append(
                GraphRelation(
                    source_id=chunk.id,
                    target_id=line.id,
                    relation="chunk_to_line",
                    score=1.0,
                )
            )
    return relations


def build_near_line_relations(
    lines: list[OCRLine],
    chunks: list[InfographicChunk],
    config: InfographicRelationConfig = DEFAULT_INFOGRAPHIC_RELATION_CONFIG,
) -> list[InfographicLineRelation]:
    page_width, page_height = _page_extent_from_lines(lines)
    page_diag = math.hypot(page_width, page_height)
    line_to_chunk = {
        line.id: chunk.id
        for chunk in chunks
        for line in chunk.lines
    }
    ordered_lines = sorted(lines, key=lambda item: (item.page, item.bbox.top, item.bbox.left))
    seen_pairs: set[tuple[str, str]] = set()
    relations: list[InfographicLineRelation] = []

    for source in ordered_lines:
        candidates: list[tuple[float, OCRLine]] = []
        for candidate in ordered_lines:
            if candidate.id == source.id or candidate.page != source.page:
                continue
            if line_to_chunk.get(candidate.id) == line_to_chunk.get(source.id):
                continue

            center_distance_ratio = math.hypot(
                source.bbox.center_x - candidate.bbox.center_x,
                source.bbox.center_y - candidate.bbox.center_y,
            ) / max(page_diag, 1.0)
            if center_distance_ratio > config.near_line_max_center_distance_ratio:
                continue

            overlap_ratio = source.bbox.horizontal_overlap_ratio(candidate.bbox)
            vertical_overlap_ratio = source.bbox.vertical_overlap_ratio(candidate.bbox)
            distance_score = max(
                0.0,
                1.0 - (
                    center_distance_ratio
                    / max(config.near_line_max_center_distance_ratio, 1e-6)
                ),
            )
            alignment_score = max(overlap_ratio, vertical_overlap_ratio)
            score = (
                0.60 * distance_score
                + 0.20 * alignment_score
                + 0.20 * _height_similarity(source.bbox.height, candidate.bbox.height)
            )
            candidates.append((score, candidate))

        for score, candidate in sorted(candidates, key=lambda item: item[0], reverse=True)[
            : config.near_line_top_k
        ]:
            pair = tuple(sorted((source.id, candidate.id)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            relations.append(
                InfographicLineRelation(
                    source_id=source.id,
                    target_id=candidate.id,
                    relation="near_line",
                    score=round(score, 4),
                )
            )
    return relations


def build_next_chunk_relations(
    chunks: list[InfographicChunk],
    config: InfographicRelationConfig = DEFAULT_INFOGRAPHIC_RELATION_CONFIG,
) -> list[GraphRelation]:
    page_width, _ = _page_extent_from_chunks(chunks)
    ordered_chunks = sorted(chunks, key=lambda item: (item.page, item.bbox.top, item.bbox.left))
    relations: list[GraphRelation] = []

    for source in ordered_chunks:
        best_candidate: InfographicChunk | None = None
        best_score = 0.0
        for candidate in ordered_chunks:
            if candidate.id == source.id or candidate.page != source.page:
                continue
            if candidate.bbox.top <= source.bbox.top:
                continue

            avg_height = max((source.bbox.height + candidate.bbox.height) / 2.0, 1.0)
            vertical_gap_ratio = source.bbox.vertical_gap(candidate.bbox) / avg_height
            if vertical_gap_ratio > config.next_chunk_max_vertical_gap_ratio:
                continue

            overlap_ratio = source.bbox.horizontal_overlap_ratio(candidate.bbox)
            left_delta_ratio = abs(source.bbox.left - candidate.bbox.left) / page_width
            center_delta_ratio = abs(source.bbox.center_x - candidate.bbox.center_x) / page_width
            aligned = (
                overlap_ratio >= config.next_chunk_min_horizontal_overlap_ratio
                or left_delta_ratio <= config.next_chunk_max_left_delta_ratio
                or center_delta_ratio <= config.next_chunk_max_left_delta_ratio
            )
            if not aligned:
                continue

            gap_score = max(
                0.0,
                1.0 - (vertical_gap_ratio / max(config.next_chunk_max_vertical_gap_ratio, 1e-6)),
            )
            align_score = max(
                overlap_ratio,
                max(0.0, 1.0 - (left_delta_ratio / max(config.next_chunk_max_left_delta_ratio, 1e-6))),
                max(0.0, 1.0 - (center_delta_ratio / max(config.next_chunk_max_left_delta_ratio, 1e-6))),
            )
            score = (
                0.45 * gap_score
                + 0.35 * align_score
                + 0.20 * _height_similarity(source.bbox.height, candidate.bbox.height)
            )
            if score > best_score:
                best_candidate = candidate
                best_score = score

        if best_candidate is not None:
            relations.append(
                GraphRelation(
                    source_id=source.id,
                    target_id=best_candidate.id,
                    relation="next_chunk",
                    score=round(best_score, 4),
                )
            )
    return relations


def build_near_chunk_relations(
    chunks: list[InfographicChunk],
    config: InfographicRelationConfig = DEFAULT_INFOGRAPHIC_RELATION_CONFIG,
) -> list[GraphRelation]:
    page_width, page_height = _page_extent_from_chunks(chunks)
    page_diag = math.hypot(page_width, page_height)
    ordered_chunks = sorted(chunks, key=lambda item: (item.page, item.bbox.top, item.bbox.left))
    seen_pairs: set[tuple[str, str]] = set()
    relations: list[GraphRelation] = []

    for source in ordered_chunks:
        candidates: list[tuple[float, InfographicChunk]] = []
        for candidate in ordered_chunks:
            if candidate.id == source.id or candidate.page != source.page:
                continue

            center_distance_ratio = math.hypot(
                source.bbox.center_x - candidate.bbox.center_x,
                source.bbox.center_y - candidate.bbox.center_y,
            ) / max(page_diag, 1.0)
            if center_distance_ratio > config.near_chunk_max_center_distance_ratio:
                continue

            overlap_ratio = max(
                source.bbox.horizontal_overlap_ratio(candidate.bbox),
                source.bbox.vertical_overlap_ratio(candidate.bbox),
            )
            distance_score = max(
                0.0,
                1.0 - (
                    center_distance_ratio
                    / max(config.near_chunk_max_center_distance_ratio, 1e-6)
                ),
            )
            score = (
                0.70 * distance_score
                + 0.30 * overlap_ratio
            )
            candidates.append((score, candidate))

        for score, candidate in sorted(candidates, key=lambda item: item[0], reverse=True)[
            : config.near_chunk_top_k
        ]:
            pair = tuple(sorted((source.id, candidate.id)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            relations.append(
                GraphRelation(
                    source_id=source.id,
                    target_id=candidate.id,
                    relation="near_chunk",
                    score=round(score, 4),
                )
            )
    return relations


def build_infographic_text_relations(
    lines: list[OCRLine],
    chunks: list[InfographicChunk],
    config: InfographicRelationConfig = DEFAULT_INFOGRAPHIC_RELATION_CONFIG,
) -> tuple[list[InfographicLineRelation], list[GraphRelation]]:
    line_relations = [
        *build_next_line_relations(lines, config=config),
        *build_near_line_relations(lines, chunks=chunks, config=config),
    ]
    graph_relations = [
        *build_line_chunk_membership_relations(chunks),
        *build_next_chunk_relations(chunks, config=config),
        *build_near_chunk_relations(chunks, config=config),
    ]
    return line_relations, graph_relations
