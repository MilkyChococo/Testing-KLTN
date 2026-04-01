from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from src.type.base import BoundingBox, OCRLine

from .graph_types import GraphRelation
from .line_to_chunk import TextChunk
from .line_to_line import LineRelation


@dataclass(slots=True)
class LayoutRegion:
    id: str
    page: int
    label: str
    bbox: BoundingBox
    score: float
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def build_layout_regions(
    detections: list[dict[str, Any]],
    page_number: int = 1,
) -> list[LayoutRegion]:
    regions: list[LayoutRegion] = []
    for index, detection in enumerate(detections):
        x1, y1, x2, y2 = [float(value) for value in detection.get("bbox_xyxy", [0, 0, 0, 0])]
        if x2 <= x1 or y2 <= y1:
            continue

        label = str(detection.get("label", "region")).strip().lower()
        regions.append(
            LayoutRegion(
                id=f"p{page_number}_{label}_{index:03d}",
                page=page_number,
                label=label,
                bbox=BoundingBox(x1, y1, x2, y2),
                score=float(detection.get("score", 0.0)),
                content=str(detection.get("content", "")).strip(),
                metadata={
                    "raw_label": detection.get("label", label),
                    "crop_path": detection.get("crop_path", ""),
                    "content_lines": list(detection.get("content_lines", [])),
                    "ocr_content": detection.get("ocr_content", ""),
                    "gemini_analysis": detection.get("gemini_analysis", {}),
                    "gemini_model": detection.get("gemini_model", ""),
                    "gemini_text": detection.get("gemini_text", ""),
                    "qwen_analysis": detection.get("qwen_analysis", {}),
                    "qwen_model": detection.get("qwen_model", ""),
                    "qwen_text": detection.get("qwen_text", ""),
                },
            )
        )
    return regions


def _area(box: BoundingBox) -> float:
    return max(0.0, box.width) * max(0.0, box.height)


def _inter_area(box_a: BoundingBox, box_b: BoundingBox) -> float:
    ix1 = max(box_a.left, box_b.left)
    iy1 = max(box_a.top, box_b.top)
    ix2 = min(box_a.right, box_b.right)
    iy2 = min(box_a.bottom, box_b.bottom)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def build_region_chunk_relations(
    regions: list[LayoutRegion],
    chunks: list[TextChunk],
    line_relations: list[LineRelation] | None = None,
    line_region_relations: list[GraphRelation] | None = None,
    min_line_overlap_ratio: float = 0.35,
    min_chunk_overlap_ratio: float = 0.10,
    min_spatial_score: float = 0.25,
) -> list[GraphRelation]:
    relations: list[GraphRelation] = []
    bridge_scores = _build_chunk_region_bridge_scores(
        chunks=chunks,
        line_relations=line_relations or [],
        line_region_relations=line_region_relations or [],
    )

    for region in regions:
        for chunk in chunks:
            if region.page != chunk.page:
                continue

            chunk_area = _area(chunk.bbox)
            region_area = _area(region.bbox)
            if chunk_area <= 0.0 or region_area <= 0.0:
                continue

            chunk_inter = _inter_area(region.bbox, chunk.bbox)
            chunk_overlap_ratio = chunk_inter / min(chunk_area, region_area)
            spatial_score = _chunk_region_spatial_score(chunk.bbox, region.bbox)
            bridge_score = bridge_scores.get((chunk.id, region.id), 0.0)

            best_line_overlap_ratio = 0.0
            for line in chunk.lines:
                line_area = _area(line.bbox)
                if line_area <= 0.0:
                    continue
                line_inter = _inter_area(region.bbox, line.bbox)
                overlap_ratio = line_inter / line_area
                if overlap_ratio > best_line_overlap_ratio:
                    best_line_overlap_ratio = overlap_ratio

            if (
                bridge_score <= 0.0
                and spatial_score < min_spatial_score
                and
                best_line_overlap_ratio < min_line_overlap_ratio
                and chunk_overlap_ratio < min_chunk_overlap_ratio
            ):
                continue

            score = round(
                min(
                    1.0,
                    max(
                        chunk_overlap_ratio,
                        best_line_overlap_ratio,
                        spatial_score,
                        bridge_score,
                    ),
                ),
                4,
            )
            relations.append(
                GraphRelation(
                    source_id=region.id,
                    target_id=chunk.id,
                    relation="region_to_chunk",
                    score=score,
                )
            )
            relations.append(
                GraphRelation(
                    source_id=chunk.id,
                    target_id=region.id,
                    relation="chunk_to_region",
                    score=score,
                )
            )

    return relations


def build_line_region_relations(
    lines: list[OCRLine],
    regions: list[LayoutRegion],
    min_line_overlap_ratio: float = 0.5,
) -> list[GraphRelation]:
    relations: list[GraphRelation] = []

    for line in lines:
        best_region: LayoutRegion | None = None
        best_overlap_ratio = 0.0

        for region in regions:
            if region.page != line.page:
                continue

            line_area = _area(line.bbox)
            if line_area <= 0.0:
                continue

            overlap_ratio = _inter_area(line.bbox, region.bbox) / line_area
            if overlap_ratio < min_line_overlap_ratio or overlap_ratio <= best_overlap_ratio:
                continue

            best_region = region
            best_overlap_ratio = overlap_ratio

        if best_region is None:
            continue

        score = round(min(1.0, best_overlap_ratio), 4)
        relations.append(
            GraphRelation(
                source_id=line.id,
                target_id=best_region.id,
                relation="line_to_region",
                score=score,
            )
        )
        relations.append(
            GraphRelation(
                source_id=best_region.id,
                target_id=line.id,
                relation="region_to_line",
                score=score,
            )
        )

    return relations


def build_next_region_relations(regions: list[LayoutRegion]) -> list[GraphRelation]:
    ordered_regions = sorted(
        regions,
        key=lambda region: (region.page, region.bbox.top, region.bbox.left),
    )
    relations: list[GraphRelation] = []

    for source, target in zip(ordered_regions, ordered_regions[1:]):
        relations.append(
            GraphRelation(
                source_id=source.id,
                target_id=target.id,
                relation="next_region",
            )
        )

    return relations


def build_region_graph_relations(
    lines: list[OCRLine],
    line_relations: list[LineRelation],
    chunks: list[TextChunk],
    regions: list[LayoutRegion],
    min_line_overlap_ratio: float = 0.5,
    min_chunk_overlap_ratio: float = 0.10,
    min_spatial_score: float = 0.25,
    min_lexical_score: float = 0.08,
    top_k_lexical_chunks: int = 2,
) -> list[GraphRelation]:
    line_region_relations = build_line_region_relations(
        lines=lines,
        regions=regions,
        min_line_overlap_ratio=min_line_overlap_ratio,
    )
    chunk_region_relations = build_region_chunk_relations(
        regions=regions,
        chunks=chunks,
        line_relations=line_relations,
        line_region_relations=line_region_relations,
        min_line_overlap_ratio=min_line_overlap_ratio,
        min_chunk_overlap_ratio=min_chunk_overlap_ratio,
        min_spatial_score=min_spatial_score,
    )
    lexical_relations = build_region_chunk_lexical_relations(
        chunks=chunks,
        regions=regions,
        min_lexical_score=min_lexical_score,
        top_k_chunks=top_k_lexical_chunks,
    )
    region_region_relations = build_next_region_relations(regions)
    return _dedupe_relations(
        [
            *line_region_relations,
            *chunk_region_relations,
            *lexical_relations,
            *region_region_relations,
        ]
    )


def _center_alignment_score(box_a: BoundingBox, box_b: BoundingBox) -> float:
    base = max(box_a.width, box_b.width, 1.0)
    return max(0.0, 1.0 - (abs(box_a.center_x - box_b.center_x) / base))


def _chunk_region_spatial_score(chunk_box: BoundingBox, region_box: BoundingBox) -> float:
    chunk_area = _area(chunk_box)
    region_area = _area(region_box)
    if chunk_area <= 0.0 or region_area <= 0.0:
        return 0.0

    inter_ratio = _inter_area(chunk_box, region_box) / min(chunk_area, region_area)
    horizontal_overlap = chunk_box.horizontal_overlap_ratio(region_box)
    center_alignment = _center_alignment_score(chunk_box, region_box)
    avg_height = max((chunk_box.height + region_box.height) / 2.0, 1.0)
    vertical_gap_ratio = chunk_box.vertical_gap(region_box) / avg_height
    proximity = 1.0 / (1.0 + vertical_gap_ratio)

    return min(
        1.0,
        max(
            inter_ratio,
            (0.65 * horizontal_overlap + 0.35 * center_alignment) * proximity,
        ),
    )


def _build_chunk_region_bridge_scores(
    chunks: list[TextChunk],
    line_relations: list[LineRelation],
    line_region_relations: list[GraphRelation],
) -> dict[tuple[str, str], float]:
    chunk_by_line_id = {
        line.id: chunk.id
        for chunk in chunks
        for line in chunk.lines
    }
    region_score_by_line_id = {
        relation.source_id: (relation.target_id, relation.score)
        for relation in line_region_relations
        if relation.relation == "line_to_region"
    }
    pair_scores: dict[tuple[str, str], float] = {}

    for relation in line_relations:
        source_chunk_id = chunk_by_line_id.get(relation.source_id)
        target_chunk_id = chunk_by_line_id.get(relation.target_id)
        source_region = region_score_by_line_id.get(relation.source_id)
        target_region = region_score_by_line_id.get(relation.target_id)

        if source_chunk_id is not None and target_region is not None:
            region_id, region_score = target_region
            key = (source_chunk_id, region_id)
            pair_scores[key] = max(
                pair_scores.get(key, 0.0),
                round(min(1.0, relation.score * region_score), 4),
            )

        if target_chunk_id is not None and source_region is not None:
            region_id, region_score = source_region
            key = (target_chunk_id, region_id)
            pair_scores[key] = max(
                pair_scores.get(key, 0.0),
                round(min(1.0, relation.score * region_score), 4),
            )

    return pair_scores


def _dedupe_relations(relations: list[GraphRelation]) -> list[GraphRelation]:
    best_by_key: dict[tuple[str, str, str], GraphRelation] = {}

    for relation in relations:
        key = (relation.source_id, relation.target_id, relation.relation)
        current = best_by_key.get(key)
        if current is None or relation.score > current.score:
            best_by_key[key] = relation

    return list(best_by_key.values())


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def build_region_chunk_lexical_relations(
    chunks: list[TextChunk],
    regions: list[LayoutRegion],
    min_lexical_score: float = 0.08,
    top_k_chunks: int = 2,
) -> list[GraphRelation]:
    documents: list[list[str]] = []
    chunk_tokens: dict[str, list[str]] = {}
    region_tokens: dict[str, list[str]] = {}

    for chunk in chunks:
        tokens = _tokenize_text(chunk.text)
        chunk_tokens[chunk.id] = tokens
        if tokens:
            documents.append(tokens)

    for region in regions:
        text = region.content or str(region.metadata.get("qwen_text", "")).strip()
        tokens = _tokenize_text(text)
        region_tokens[region.id] = tokens
        if tokens:
            documents.append(tokens)

    if not documents:
        return []

    idf = _build_idf(documents)
    chunk_vectors = {
        chunk.id: _tfidf_vector(chunk_tokens[chunk.id], idf)
        for chunk in chunks
        if chunk_tokens.get(chunk.id)
    }
    region_vectors = {
        region.id: _tfidf_vector(region_tokens[region.id], idf)
        for region in regions
        if region_tokens.get(region.id)
    }

    relations: list[GraphRelation] = []
    for region in regions:
        region_vector = region_vectors.get(region.id)
        if not region_vector:
            continue

        scored_chunks: list[tuple[float, TextChunk]] = []
        for chunk in chunks:
            if chunk.page != region.page:
                continue

            chunk_vector = chunk_vectors.get(chunk.id)
            if not chunk_vector:
                continue

            lexical_score = _cosine_sparse(chunk_vector, region_vector)
            if lexical_score < min_lexical_score:
                continue
            scored_chunks.append((lexical_score, chunk))

        scored_chunks.sort(key=lambda item: item[0], reverse=True)
        for lexical_score, chunk in scored_chunks[: max(1, top_k_chunks)]:
            score = round(min(1.0, lexical_score), 4)
            relations.append(
                GraphRelation(
                    source_id=chunk.id,
                    target_id=region.id,
                    relation="chunk_to_region_lexical",
                    score=score,
                )
            )
            relations.append(
                GraphRelation(
                    source_id=region.id,
                    target_id=chunk.id,
                    relation="region_to_chunk_lexical",
                    score=score,
                )
            )

    return relations


def _tokenize_text(text: str) -> list[str]:
    return [
        token.lower()
        for token in _TOKEN_PATTERN.findall(text)
        if len(token) > 1
    ]


def _build_idf(documents: list[list[str]]) -> dict[str, float]:
    doc_count = max(1, len(documents))
    document_frequency: Counter[str] = Counter()
    for tokens in documents:
        document_frequency.update(set(tokens))
    return {
        token: math.log((1.0 + doc_count) / (1.0 + frequency)) + 1.0
        for token, frequency in document_frequency.items()
    }


def _tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    counts = Counter(tokens)
    vector = {
        token: float(count) * idf.get(token, 0.0)
        for token, count in counts.items()
        if token in idf
    }
    norm = math.sqrt(sum(value * value for value in vector.values()))
    if norm <= 0.0:
        return {}
    return {token: value / norm for token, value in vector.items()}


def _cosine_sparse(
    vector_a: dict[str, float],
    vector_b: dict[str, float],
) -> float:
    if not vector_a or not vector_b:
        return 0.0
    if len(vector_a) > len(vector_b):
        vector_a, vector_b = vector_b, vector_a
    return sum(value * vector_b.get(token, 0.0) for token, value in vector_a.items())
