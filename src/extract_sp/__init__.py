"""Extraction helpers for OCR, layout, and relation building."""

from src.api.qwen_vl_region_analysis import (
    DEFAULT_QWEN_MODEL,
    DEFAULT_SUPPORTED_LABELS,
    QwenRegionAnalysis,
    analyze_region_with_qwen,
    annotate_detections_with_qwen,
    crop_detection_region,
)

from .component_nodes import (
    ComponentNode,
    build_coarse_component_nodes,
    build_paragraph_nodes,
    build_text_chunk_nodes,
    build_table_image_nodes,
    build_table_image_nodes_from_detections,
)
from .fine_nodes import (
    FineNode,
    build_coarse_to_fine_relations,
    build_fine_component_nodes,
    build_image_patch_nodes,
    build_sentence_nodes,
)
from .graph_builder import (
    build_chunk_graph_relations,
    build_layered_component_graph,
    build_chunk_membership_relations,
    build_next_chunk_relations,
    build_text_graph_from_ocr,
)
from .graph_types import GraphRelation
from .doclayout_yolo import (
    attach_ocr_content,
    parse_ocr_lines,
    render_bboxes,
    run_doclayout_yolo,
    save_region_crops,
)
from .document_pipeline import (
    DocumentPipelineResult,
    load_layout_detections,
    run_document_pipeline,
)
from .line_to_chunk import TextChunk, build_chunks_from_lines, build_text_chunks_from_ocr
from .line_to_line import (
    LineRelation,
    build_next_line_relations,
    build_text_relations_from_ocr,
)
from .region_to_chunk import (
    LayoutRegion,
    build_layout_regions,
    build_line_region_relations,
    build_next_region_relations,
    build_region_chunk_relations,
    build_region_chunk_lexical_relations,
    build_region_graph_relations,
)
from .valid_expand_region import (
    ExpandedRegion,
    build_expanded_region,
    expand_bbox_horizontally_to_line,
    expand_region_horizontally_to_lines,
    line_belongs_to_region,
    line_region_overlap_ratio,
    select_lines_in_region,
)
from .word_to_line import group_words_to_lines, load_words_from_ocr

__all__ = [
    "ComponentNode",
    "DocumentPipelineResult",
    "ExpandedRegion",
    "FineNode",
    "QwenRegionAnalysis",
    "GraphRelation",
    "LineRelation",
    "LayoutRegion",
    "TextChunk",
    "DEFAULT_QWEN_MODEL",
    "DEFAULT_SUPPORTED_LABELS",
    "analyze_region_with_qwen",
    "annotate_detections_with_qwen",
    "attach_ocr_content",
    "build_coarse_component_nodes",
    "build_coarse_to_fine_relations",
    "build_chunk_graph_relations",
    "build_chunk_membership_relations",
    "build_chunks_from_lines",
    "build_fine_component_nodes",
    "build_image_patch_nodes",
    "build_layered_component_graph",
    "build_next_chunk_relations",
    "build_next_line_relations",
    "build_layout_regions",
    "build_line_region_relations",
    "build_paragraph_nodes",
    "build_next_region_relations",
    "build_region_chunk_relations",
    "build_region_chunk_lexical_relations",
    "build_region_graph_relations",
    "build_sentence_nodes",
    "build_expanded_region",
    "build_text_chunk_nodes",
    "build_table_image_nodes",
    "build_table_image_nodes_from_detections",
    "build_text_chunks_from_ocr",
    "build_text_graph_from_ocr",
    "build_text_relations_from_ocr",
    "crop_detection_region",
    "group_words_to_lines",
    "line_belongs_to_region",
    "line_region_overlap_ratio",
    "load_layout_detections",
    "load_words_from_ocr",
    "parse_ocr_lines",
    "render_bboxes",
    "run_document_pipeline",
    "run_doclayout_yolo",
    "save_region_crops",
    "expand_region_horizontally_to_lines",
    "select_lines_in_region",
    "expand_bbox_horizontally_to_line",
]
