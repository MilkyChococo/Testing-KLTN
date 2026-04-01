from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.type.base import OCRLine, OCRWord

from .component_nodes import ComponentNode, build_coarse_component_nodes
from .fine_nodes import FineNode, build_coarse_to_fine_relations, build_fine_component_nodes
from .graph_types import GraphRelation
from .line_to_chunk import TextChunk, build_text_chunks_from_ocr
from .line_to_line import LineRelation
from .region_to_chunk import LayoutRegion


def build_chunk_membership_relations(chunks: list[TextChunk]) -> list[GraphRelation]:
    relations: list[GraphRelation] = []
    for chunk in chunks:
        for line in chunk.lines:
            relations.append(
                GraphRelation(
                    source_id=line.id,
                    target_id=chunk.id,
                    relation="line_to_chunk",
                )
            )
            relations.append(
                GraphRelation(
                    source_id=chunk.id,
                    target_id=line.id,
                    relation="chunk_to_line",
                )
            )
    return relations


def build_next_chunk_relations(chunks: list[TextChunk]) -> list[GraphRelation]:
    ordered_chunks = sorted(
        chunks, key=lambda chunk: (chunk.page, chunk.bbox.top, chunk.bbox.left)
    )
    relations: list[GraphRelation] = []

    for source, target in zip(ordered_chunks, ordered_chunks[1:]):
        relations.append(
            GraphRelation(
                source_id=source.id,
                target_id=target.id,
                relation="next_chunk",
            )
        )

    return relations


def build_chunk_graph_relations(chunks: list[TextChunk]) -> list[GraphRelation]:
    return [
        *build_chunk_membership_relations(chunks),
        *build_next_chunk_relations(chunks),
    ]


def build_text_graph_from_ocr(
    ocr_path: str | Path,
    page_number: int = 1,
) -> tuple[
    list[OCRWord],
    list[OCRLine],
    list[LineRelation],
    list[TextChunk],
    list[GraphRelation],
]:
    words, lines, line_relations, chunks = build_text_chunks_from_ocr(
        ocr_path=ocr_path,
        page_number=page_number,
    )
    graph_relations = build_chunk_graph_relations(chunks)
    return words, lines, line_relations, chunks, graph_relations


def build_layered_component_graph(
    chunks: list[TextChunk],
    regions: list[LayoutRegion] | None = None,
    include_visual_fine_nodes: bool = False,
    image_grid_rows: int = 1,
    image_grid_cols: int = 1,
) -> tuple[list[ComponentNode], list[FineNode], list[GraphRelation]]:
    coarse_nodes = build_coarse_component_nodes(chunks=chunks, regions=regions)
    fine_nodes = build_fine_component_nodes(
        coarse_nodes=coarse_nodes,
        include_image_patches=include_visual_fine_nodes,
        image_grid_rows=image_grid_rows,
        image_grid_cols=image_grid_cols,
    )
    relations = build_coarse_to_fine_relations(fine_nodes)
    return coarse_nodes, fine_nodes, relations
