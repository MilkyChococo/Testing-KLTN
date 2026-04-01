from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.type.base import OCRLine, OCRWord

from src.api.qwen_vl_region_analysis import (
    DEFAULT_QWEN_MODEL,
    annotate_detections_with_qwen,
)

from .component_nodes import ComponentNode
from .doclayout_yolo import attach_ocr_content, parse_ocr_lines, run_doclayout_yolo
from .fine_nodes import FineNode
from .graph_builder import (
    GraphRelation,
    build_chunk_graph_relations,
    build_layered_component_graph,
)
from .line_to_chunk import TextChunk, build_chunks_from_lines
from .line_to_line import LineRelation
from .line_to_line import build_next_line_relations
from .region_to_chunk import LayoutRegion, build_layout_regions, build_region_graph_relations
from .valid_expand_region import build_expanded_region, line_region_overlap_ratio
from .word_to_line import load_words_from_ocr, group_words_to_lines


@dataclass(slots=True)
class DocumentPipelineResult:
    image_path: Path
    ocr_path: Path
    words: list[OCRWord]
    lines: list[OCRLine]
    paragraph_lines: list[OCRLine]
    region_lines: list[OCRLine]
    line_relations: list[LineRelation]
    chunks: list[TextChunk]
    chunk_relations: list[GraphRelation]
    detections: list[dict[str, Any]] = field(default_factory=list)
    original_regions: list[LayoutRegion] = field(default_factory=list)
    regions: list[LayoutRegion] = field(default_factory=list)
    region_relations: list[GraphRelation] = field(default_factory=list)
    coarse_nodes: list[ComponentNode] = field(default_factory=list)
    fine_nodes: list[FineNode] = field(default_factory=list)
    hierarchical_relations: list[GraphRelation] = field(default_factory=list)


def load_layout_detections(layout_path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(layout_path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        detections = payload.get("detections", [])
    elif isinstance(payload, list):
        detections = payload
    else:
        raise ValueError(f"Unsupported layout payload type: {type(payload)!r}")

    if not isinstance(detections, list):
        raise ValueError("Layout detections must be a list.")
    return detections


def _prepare_detections(
    image_path: Path,
    ocr_path: Path,
    page_number: int,
    layout_path: str | Path | None,
    detect_layout: bool,
    layout_labels: list[str] | None,
    layout_threshold: float,
    local_model_path: str | Path | None,
    analyze_regions_with_qwen: bool = False,
    qwen_labels: list[str] | None = None,
    qwen_model: str = DEFAULT_QWEN_MODEL,
    qwen_device: str = "auto",
    qwen_dtype: str = "auto",
    qwen_max_new_tokens: int = 1024,
    qwen_temperature: float = 0.1,
    qwen_crops_dir: str | Path | None = None,
    qwen_batch_size: int = 1,
) -> list[dict[str, Any]]:
    if layout_path is not None:
        detections = load_layout_detections(layout_path)
    elif detect_layout:
        result = run_doclayout_yolo(
            image_path=image_path,
            labels=layout_labels,
            score_threshold=layout_threshold,
            local_model_path=local_model_path,
        )
        detections = result.get("detections", [])
    else:
        return []

    ocr_lines = parse_ocr_lines(ocr_path)
    attach_ocr_content(detections, ocr_lines)

    for index, detection in enumerate(detections):
        label = str(detection.get("label", "region")).strip().lower()
        detection.setdefault("id", f"p{page_number}_{label}_{index:03d}")

    if analyze_regions_with_qwen and detections:
        annotate_detections_with_qwen(
            image_path=image_path,
            detections=detections,
            labels=tuple(qwen_labels or layout_labels or ()),
            model=qwen_model,
            device=qwen_device,
            dtype=qwen_dtype,
            max_new_tokens=qwen_max_new_tokens,
            temperature=qwen_temperature,
            save_crops_dir=qwen_crops_dir,
            merge_into_content=True,
            batch_size=qwen_batch_size,
        )

    return detections


def expand_regions_with_lines(
    regions: list[LayoutRegion],
    lines: list[OCRLine],
    min_line_overlap_ratio: float = 0.5,
    horizontal_padding: float = 0.0,
    clamp_to_region: bool = False,
) -> list[LayoutRegion]:
    expanded_regions: list[LayoutRegion] = []
    for region in regions:
        expansion = build_expanded_region(
            lines=lines,
            region=region,
            min_line_overlap_ratio=min_line_overlap_ratio,
            horizontal_padding=horizontal_padding,
            clamp_to_region=clamp_to_region,
        )
        expanded_regions.append(
            LayoutRegion(
                id=region.id,
                page=region.page,
                label=region.label,
                bbox=expansion.expanded_bbox,
                score=region.score,
                content=region.content,
                metadata={
                    **region.metadata,
                    "original_bbox": region.bbox,
                    "expanded_line_count": expansion.num_lines,
                    "expanded_line_ids": expansion.line_ids,
                    "min_line_left": expansion.min_line_left,
                    "max_line_right": expansion.max_line_right,
                },
            )
        )
    return expanded_regions


def split_lines_by_regions(
    lines: list[OCRLine],
    regions: list[LayoutRegion],
    min_line_overlap_ratio: float = 0.5,
) -> tuple[list[OCRLine], list[OCRLine]]:
    if not regions:
        return lines, []

    outside_regions: list[OCRLine] = []
    inside_regions: list[OCRLine] = []

    for line in lines:
        best_region: LayoutRegion | None = None
        best_overlap_ratio = 0.0

        for region in regions:
            overlap_ratio = line_region_overlap_ratio(line, region)
            if (
                overlap_ratio >= min_line_overlap_ratio
                and overlap_ratio > best_overlap_ratio
            ):
                best_region = region
                best_overlap_ratio = overlap_ratio

        if best_region is not None:
            line.metadata["inside_region"] = True
            line.metadata["region_id"] = best_region.id
            line.metadata["region_label"] = best_region.label
            line.metadata["region_overlap_ratio"] = round(best_overlap_ratio, 4)
            inside_regions.append(line)
        else:
            outside_regions.append(line)

    return outside_regions, inside_regions


def run_document_pipeline(
    image_path: str | Path,
    ocr_path: str | Path,
    page_number: int = 1,
    layout_path: str | Path | None = None,
    detect_layout: bool = False,
    layout_labels: list[str] | None = None,
    layout_threshold: float = 0.25,
    local_model_path: str | Path | None = None,
    min_region_line_overlap_ratio: float = 0.5,
    region_horizontal_padding: float = 0.0,
    clamp_expanded_region_to_original: bool = False,
    include_visual_fine_nodes: bool = False,
    image_grid_rows: int = 1,
    image_grid_cols: int = 1,
    analyze_regions_with_qwen: bool = False,
    qwen_labels: list[str] | None = None,
    qwen_model: str = DEFAULT_QWEN_MODEL,
    qwen_device: str = "auto",
    qwen_dtype: str = "auto",
    qwen_max_new_tokens: int = 1024,
    qwen_temperature: float = 0.1,
    qwen_crops_dir: str | Path | None = None,
    qwen_batch_size: int = 1,
) -> DocumentPipelineResult:
    image_path = Path(image_path)
    ocr_path = Path(ocr_path)

    words = load_words_from_ocr(
        ocr_path=ocr_path,
        page_number=page_number,
    )
    lines = group_words_to_lines(words)
    all_line_relations = build_next_line_relations(lines)

    detections = _prepare_detections(
        image_path=image_path,
        ocr_path=ocr_path,
        page_number=page_number,
        layout_path=layout_path,
        detect_layout=detect_layout,
        layout_labels=layout_labels,
        layout_threshold=layout_threshold,
        local_model_path=local_model_path,
        analyze_regions_with_qwen=analyze_regions_with_qwen,
        qwen_labels=qwen_labels,
        qwen_model=qwen_model,
        qwen_device=qwen_device,
        qwen_dtype=qwen_dtype,
        qwen_max_new_tokens=qwen_max_new_tokens,
        qwen_temperature=qwen_temperature,
        qwen_crops_dir=qwen_crops_dir,
        qwen_batch_size=qwen_batch_size,
    )
    original_regions = build_layout_regions(detections=detections, page_number=page_number)
    regions = expand_regions_with_lines(
        regions=original_regions,
        lines=lines,
        min_line_overlap_ratio=min_region_line_overlap_ratio,
        horizontal_padding=region_horizontal_padding,
        clamp_to_region=clamp_expanded_region_to_original,
    )
    paragraph_lines, region_lines = split_lines_by_regions(
        lines=lines,
        regions=regions,
        min_line_overlap_ratio=min_region_line_overlap_ratio,
    )
    line_relations = build_next_line_relations(paragraph_lines)
    chunks = build_chunks_from_lines(lines=paragraph_lines, relations=line_relations)
    chunk_relations = build_chunk_graph_relations(chunks)
    region_relations = build_region_graph_relations(
        lines=lines,
        line_relations=all_line_relations,
        chunks=chunks,
        regions=regions,
        min_line_overlap_ratio=min_region_line_overlap_ratio,
    )

    coarse_nodes, fine_nodes, hierarchical_relations = build_layered_component_graph(
        chunks=chunks,
        regions=regions,
        include_visual_fine_nodes=include_visual_fine_nodes,
        image_grid_rows=image_grid_rows,
        image_grid_cols=image_grid_cols,
    )

    return DocumentPipelineResult(
        image_path=image_path,
        ocr_path=ocr_path,
        words=words,
        lines=lines,
        paragraph_lines=paragraph_lines,
        region_lines=region_lines,
        line_relations=line_relations,
        chunks=chunks,
        chunk_relations=chunk_relations,
        detections=detections,
        original_regions=original_regions,
        regions=regions,
        region_relations=region_relations,
        coarse_nodes=coarse_nodes,
        fine_nodes=fine_nodes,
        hierarchical_relations=hierarchical_relations,
    )
