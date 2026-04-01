from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.type.base import BoundingBox

from .component_nodes import ComponentNode
from .graph_types import GraphRelation


@dataclass(slots=True)
class FineNode:
    id: str
    page: int
    modality: str
    text: str
    bbox: BoundingBox
    parent_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _split_sentences(text: str) -> list[str]:
    parts: list[str] = []
    for block in text.splitlines():
        block = block.strip()
        if not block:
            continue
        pieces = re.split(r"(?<=[.!?])\s+", block)
        for piece in pieces:
            cleaned = piece.strip()
            if cleaned:
                parts.append(cleaned)
    return parts or ([text.strip()] if text.strip() else [])


def build_sentence_nodes(paragraph_nodes: list[ComponentNode]) -> list[FineNode]:
    nodes: list[FineNode] = []
    for paragraph in paragraph_nodes:
        if paragraph.modality != "paragraph":
            continue

        sentences = _split_sentences(paragraph.text)
        for index, sentence in enumerate(sentences):
            nodes.append(
                FineNode(
                    id=f"{paragraph.id}_sent_{index:03d}",
                    page=paragraph.page,
                    modality="sentence",
                    text=sentence,
                    bbox=paragraph.bbox,
                    parent_id=paragraph.id,
                    metadata={"sentence_index": index},
                )
            )
    return nodes


def build_image_patch_nodes(
    component_nodes: list[ComponentNode],
    grid_rows: int = 1,
    grid_cols: int = 1,
) -> list[FineNode]:
    nodes: list[FineNode] = []
    safe_rows = max(1, grid_rows)
    safe_cols = max(1, grid_cols)

    for component in component_nodes:
        if component.modality != "image":
            continue

        patch_width = component.bbox.width / safe_cols
        patch_height = component.bbox.height / safe_rows
        patch_index = 0

        for row in range(safe_rows):
            for col in range(safe_cols):
                left = component.bbox.left + (col * patch_width)
                top = component.bbox.top + (row * patch_height)
                right = component.bbox.right if col == safe_cols - 1 else left + patch_width
                bottom = (
                    component.bbox.bottom
                    if row == safe_rows - 1
                    else top + patch_height
                )

                nodes.append(
                    FineNode(
                        id=f"{component.id}_patch_{patch_index:03d}",
                        page=component.page,
                        modality="patch",
                        text=component.text,
                        bbox=BoundingBox(left, top, right, bottom),
                        parent_id=component.id,
                        metadata={
                            "patch_index": patch_index,
                            "grid_row": row,
                            "grid_col": col,
                            "grid_rows": safe_rows,
                            "grid_cols": safe_cols,
                        },
                    )
                )
                patch_index += 1
    return nodes


def build_fine_component_nodes(
    coarse_nodes: list[ComponentNode],
    include_image_patches: bool = False,
    image_grid_rows: int = 1,
    image_grid_cols: int = 1,
) -> list[FineNode]:
    fine_nodes: list[FineNode] = []
    fine_nodes.extend(build_sentence_nodes(coarse_nodes))
    if include_image_patches:
        fine_nodes.extend(
            build_image_patch_nodes(
                coarse_nodes,
                grid_rows=image_grid_rows,
                grid_cols=image_grid_cols,
            )
        )
    return fine_nodes


def build_coarse_to_fine_relations(fine_nodes: list[FineNode]) -> list[GraphRelation]:
    relations: list[GraphRelation] = []
    for node in fine_nodes:
        relations.append(
            GraphRelation(
                source_id=node.parent_id,
                target_id=node.id,
                relation="coarse_to_fine",
            )
        )
        relations.append(
            GraphRelation(
                source_id=node.id,
                target_id=node.parent_id,
                relation="fine_to_coarse",
            )
        )
    return relations
