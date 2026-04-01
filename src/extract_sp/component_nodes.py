from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.type.base import BoundingBox

from .line_to_chunk import TextChunk
from .region_to_chunk import LayoutRegion, build_layout_regions


@dataclass(slots=True)
class ComponentNode:
    id: str
    page: int
    modality: str
    text: str
    bbox: BoundingBox
    source_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


def build_text_chunk_nodes(chunks: list[TextChunk]) -> list[ComponentNode]:
    nodes: list[ComponentNode] = []
    for chunk in chunks:
        nodes.append(
            ComponentNode(
                id=chunk.id,
                page=chunk.page,
                modality="paragraph",
                text=chunk.text,
                bbox=chunk.bbox,
                source_id=chunk.id,
                metadata={
                    "chunk_id": chunk.id,
                    "num_lines": len(chunk.lines),
                },
            )
        )
    return nodes


def build_paragraph_nodes(chunks: list[TextChunk]) -> list[ComponentNode]:
    return build_text_chunk_nodes(chunks)


def _region_to_modality(label: str) -> str:
    norm = label.strip().lower()
    if norm in {"table", "bordered", "borderless"}:
        return "table"
    return "image"


def build_table_image_nodes(regions: list[LayoutRegion]) -> list[ComponentNode]:
    nodes: list[ComponentNode] = []
    for region in regions:
        nodes.append(
            ComponentNode(
                id=region.id,
                page=region.page,
                modality=_region_to_modality(region.label),
                text=region.content,
                bbox=region.bbox,
                source_id=region.id,
                metadata={
                    "label": region.label,
                    "score": region.score,
                    **region.metadata,
                },
            )
        )
    return nodes


def build_table_image_nodes_from_detections(
    detections: list[dict[str, Any]],
    page_number: int = 1,
) -> list[ComponentNode]:
    regions = build_layout_regions(detections=detections, page_number=page_number)
    return build_table_image_nodes(regions)


def build_coarse_component_nodes(
    chunks: list[TextChunk],
    regions: list[LayoutRegion] | None = None,
) -> list[ComponentNode]:
    nodes = build_text_chunk_nodes(chunks)
    if regions:
        nodes.extend(build_table_image_nodes(regions))
    return nodes
