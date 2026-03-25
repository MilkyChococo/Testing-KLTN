from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extract.document_pipeline import run_document_pipeline
from src.extract.graph_builder import build_text_graph_from_ocr
from src.extract.line_to_chunk import build_text_chunks_from_ocr
from src.extract.line_to_line import build_text_relations_from_ocr
from src.type.base import OCRLine
from src.utils.config import DEFAULT_QWEN_VL_MODEL


DEFAULT_OCR_DIR = PROJECT_ROOT / "dataset" / "spdocvqa" / "spdocvqa_ocr"
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "dataset" / "spdocvqa" / "spdocvqa_images"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "visualizations"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot OCR line bounding boxes on a document image."
    )
    parser.add_argument(
        "image",
        type=Path,
        help="Path to the input image. If relative, it is resolved from the project root.",
    )
    parser.add_argument(
        "--ocr",
        type=Path,
        default=None,
        help="Optional path to the OCR JSON file. Defaults to the file with the same stem in spdocvqa_ocr.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image path. If omitted, the figure is only shown.",
    )
    parser.add_argument(
        "--page",
        type=int,
        default=1,
        help="OCR page number to visualize. Default is 1.",
    )
    parser.add_argument(
        "--hide-text",
        action="store_true",
        help="Hide OCR line text labels and draw only the polygons.",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=2,
        help="Bounding polygon line width.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the overlay without opening a matplotlib window.",
    )
    parser.add_argument(
        "--plot-next-line",
        action="store_true",
        help="Draw rebuilt line boxes and their next_line relations.",
    )
    parser.add_argument(
        "--hide-relation-text",
        action="store_true",
        help="Hide relation score labels when drawing next_line relations.",
    )
    parser.add_argument(
        "--plot-chunks",
        action="store_true",
        help="Draw chunk boxes built from next_line expansion.",
    )
    parser.add_argument(
        "--plot-graph",
        action="store_true",
        help="Draw line-to-line, line-to-chunk, and chunk-to-line relations together.",
    )
    parser.add_argument(
        "--plot-layered",
        action="store_true",
        help="Draw coarse and fine component nodes.",
    )
    parser.add_argument(
        "--layout-json",
        type=Path,
        default=None,
        help="Optional layout detection JSON to include table/image nodes.",
    )
    parser.add_argument(
        "--detect-layout",
        action="store_true",
        help="Run DocLayout-YOLO directly instead of loading layout JSON.",
    )
    parser.add_argument(
        "--layout-threshold",
        type=float,
        default=0.65,
        help="Detection score threshold for DocLayout-YOLO.",
    )
    parser.add_argument(
        "--layout-labels",
        default="table,figure,chart,image",
        help="Comma-separated labels to keep when running layout detection.",
    )
    parser.add_argument(
        "--local-model-path",
        type=Path,
        default=None,
        help="Optional local DocLayout-YOLO model path.",
    )
    parser.add_argument(
        "--analyze-regions-with-qwen",
        action="store_true",
        help="Call Qwen2.5-VL to analyze detected table/figure/chart/image regions.",
    )
    parser.add_argument(
        "--qwen-model",
        default=DEFAULT_QWEN_VL_MODEL,
        help="Qwen2.5-VL model name used for region analysis.",
    )
    parser.add_argument(
        "--qwen-device",
        default="auto",
        help="Qwen inference device: auto, cuda, or cpu.",
    )
    parser.add_argument(
        "--qwen-dtype",
        default="auto",
        help="Qwen inference dtype: auto, float16, bfloat16, or float32.",
    )
    parser.add_argument(
        "--qwen-max-new-tokens",
        type=int,
        default=1024,
        help="Max generated tokens for Qwen region analysis.",
    )
    parser.add_argument(
        "--qwen-temperature",
        type=float,
        default=0.1,
        help="Temperature for Qwen region analysis.",
    )
    parser.add_argument(
        "--qwen-crops-dir",
        type=Path,
        default=None,
        help="Optional directory to save region crops sent to Qwen.",
    )
    return parser.parse_args()


def resolve_image_path(image_path: Path) -> Path:
    if image_path.is_absolute():
        return image_path
    if image_path.exists():
        return image_path.resolve()
    candidate = PROJECT_ROOT / image_path
    if candidate.exists():
        return candidate.resolve()
    candidate = DEFAULT_IMAGE_DIR / image_path.name
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f"Image not found: {image_path}")


def resolve_ocr_path(image_path: Path, ocr_path: Path | None) -> Path:
    if ocr_path is not None:
        if ocr_path.is_absolute():
            return ocr_path
        if ocr_path.exists():
            return ocr_path.resolve()
        candidate = PROJECT_ROOT / ocr_path
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(f"OCR file not found: {ocr_path}")

    candidate = DEFAULT_OCR_DIR / f"{image_path.stem}.json"
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(
        f"Could not infer OCR JSON for image stem '{image_path.stem}' in {DEFAULT_OCR_DIR}"
    )


def resolve_optional_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    candidate = PROJECT_ROOT / path
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f"File not found: {path}")


def load_ocr_lines(ocr_path: Path, page_number: int) -> list[dict[str, Any]]:
    payload = json.loads(ocr_path.read_text(encoding="utf-8"))
    pages = payload.get("recognitionResults", [])
    for page in pages:
        if int(page.get("page", 0)) == page_number:
            return page.get("lines", [])
    raise ValueError(f"Page {page_number} not found in OCR file: {ocr_path}")


def polygon_points(bounding_box: list[float]) -> list[tuple[float, float]]:
    if len(bounding_box) < 8 or len(bounding_box) % 2 != 0:
        raise ValueError("Expected an OCR polygon bounding box with 8 coordinates.")
    return [
        (float(bounding_box[index]), float(bounding_box[index + 1]))
        for index in range(0, len(bounding_box), 2)
    ]


def polygon_anchor(points: list[tuple[float, float]]) -> tuple[float, float]:
    min_x = min(point[0] for point in points)
    min_y = min(point[1] for point in points)
    return min_x, min_y


def draw_ocr_lines(
    image: Image.Image,
    lines: list[dict[str, Any]],
    show_text: bool,
    line_width: int,
) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for index, line in enumerate(lines, start=1):
        points = polygon_points(line["boundingBox"])
        polygon = points + [points[0]]
        draw.line(polygon, fill="red", width=line_width)

        if not show_text:
            continue

        anchor_x, anchor_y = polygon_anchor(points)
        label = f"{index}. {line.get('text', '').strip()}"
        draw.text((anchor_x + 2, max(0, anchor_y - 12)), label, fill="yellow", font=font)

    return canvas


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    width: int = 2,
    arrow_size: int = 8,
) -> None:
    draw.line([start, end], fill=color, width=width)

    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length <= 0.0:
        return

    ux = dx / length
    uy = dy / length
    left = (
        end[0] - arrow_size * ux + (arrow_size * 0.5) * uy,
        end[1] - arrow_size * uy - (arrow_size * 0.5) * ux,
    )
    right = (
        end[0] - arrow_size * ux - (arrow_size * 0.5) * uy,
        end[1] - arrow_size * uy + (arrow_size * 0.5) * ux,
    )
    draw.polygon([end, left, right], fill=color)


def line_center(line: OCRLine) -> tuple[float, float]:
    return line.bbox.center_x, line.bbox.center_y


def line_anchor(line: OCRLine) -> tuple[float, float]:
    return line.bbox.left, line.bbox.top


def draw_grouped_lines_and_relations(
    image: Image.Image,
    lines: list[OCRLine],
    show_text: bool,
    show_relation_text: bool,
    line_width: int,
    relation_lookup: dict[str, str],
    relation_scores: dict[str, float],
) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    id_to_line = {line.id: line for line in lines}

    for line in lines:
        rectangle = [
            (line.bbox.left, line.bbox.top),
            (line.bbox.right, line.bbox.bottom),
        ]
        draw.rectangle(rectangle, outline="cyan", width=line_width)

        if show_text:
            anchor_x, anchor_y = line_anchor(line)
            label = line.text
            draw.text(
                (anchor_x + 2, max(0, anchor_y - 12)),
                label,
                fill="cyan",
                font=font,
            )

    for source_id, target_id in relation_lookup.items():
        source = id_to_line[source_id]
        target = id_to_line[target_id]
        start = line_center(source)
        end = line_center(target)
        draw_arrow(draw, start, end, color="lime", width=max(1, line_width))

        if show_relation_text:
            score = relation_scores[source_id]
            mid_x = (start[0] + end[0]) / 2.0
            mid_y = (start[1] + end[1]) / 2.0
            draw.text((mid_x + 2, max(0, mid_y - 10)), f"{score:.2f}", fill="lime", font=font)

    return canvas


def draw_chunks(
    image: Image.Image,
    chunks: list[Any],
    show_text: bool,
    line_width: int,
) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for chunk in chunks:
        rectangle = [
            (chunk.bbox.left, chunk.bbox.top),
            (chunk.bbox.right, chunk.bbox.bottom),
        ]
        draw.rectangle(rectangle, outline="orange", width=line_width)

        if not show_text:
            continue

        label = f"{chunk.id} ({len(chunk.lines)} lines)"
        draw.text(
            (chunk.bbox.left + 2, max(0, chunk.bbox.top - 12)),
            label,
            fill="orange",
            font=font,
        )

    return canvas


def chunk_center(chunk: Any) -> tuple[float, float]:
    return chunk.bbox.center_x, chunk.bbox.center_y


def bbox_center(bbox: Any) -> tuple[float, float]:
    return bbox.center_x, bbox.center_y


def virtual_fine_bbox(fine_node: Any, sibling_count: int, sibling_index: int) -> Any:
    if fine_node.modality == "patch":
        return fine_node.bbox

    parent_box = fine_node.bbox
    count = max(1, sibling_count)
    index = max(0, min(sibling_index, count - 1))
    slot_height = parent_box.height / count
    top = parent_box.top + (index * slot_height)
    bottom = parent_box.bottom if index == count - 1 else top + slot_height
    return type(parent_box)(parent_box.left, top, parent_box.right, bottom)


def draw_text_graph(
    image: Image.Image,
    lines: list[OCRLine],
    chunks: list[Any],
    next_line_relations: list[Any],
    graph_relations: list[Any],
    regions: list[Any] | None,
    region_relations: list[Any] | None,
    show_text: bool,
    show_relation_text: bool,
    line_width: int,
) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    line_by_id = {line.id: line for line in lines}
    chunk_by_id = {chunk.id: chunk for chunk in chunks}
    region_by_id = {region.id: region for region in (regions or [])}

    for line in lines:
        draw.rectangle(
            [(line.bbox.left, line.bbox.top), (line.bbox.right, line.bbox.bottom)],
            outline="cyan",
            width=line_width,
        )
        if show_text:
            draw.text(
                (line.bbox.left + 2, max(0, line.bbox.top - 12)),
                line.text,
                fill="cyan",
                font=font,
            )

    for chunk in chunks:
        draw.rectangle(
            [(chunk.bbox.left, chunk.bbox.top), (chunk.bbox.right, chunk.bbox.bottom)],
            outline="orange",
            width=line_width,
        )
        if show_text:
            draw.text(
                (chunk.bbox.left + 2, max(0, chunk.bbox.top - 26)),
                chunk.id,
                fill="orange",
                font=font,
            )

    for region in regions or []:
        original_bbox = region.metadata.get("original_bbox")
        if original_bbox is not None:
            draw.rectangle(
                [
                    (original_bbox.left, original_bbox.top),
                    (original_bbox.right, original_bbox.bottom),
                ],
                outline="yellow",
                width=1,
            )
        draw.rectangle(
            [(region.bbox.left, region.bbox.top), (region.bbox.right, region.bbox.bottom)],
            outline="springgreen",
            width=line_width,
        )
        if show_text:
            draw.text(
                (region.bbox.left + 2, max(0, region.bbox.top - 12)),
                region.id,
                fill="springgreen",
                font=font,
            )

    for relation in next_line_relations:
        source = line_by_id[relation.source_id]
        target = line_by_id[relation.target_id]
        start = line_center(source)
        end = line_center(target)
        draw_arrow(draw, start, end, color="lime", width=max(1, line_width))
        if show_relation_text:
            mid_x = (start[0] + end[0]) / 2.0
            mid_y = (start[1] + end[1]) / 2.0
            draw.text(
                (mid_x + 2, max(0, mid_y - 10)),
                f"{relation.score:.2f}",
                fill="lime",
                font=font,
            )

    for relation in graph_relations:
        if relation.relation == "line_to_chunk":
            source = line_by_id[relation.source_id]
            target = chunk_by_id[relation.target_id]
            start = line_center(source)
            end = chunk_center(target)
            draw_arrow(draw, start, end, color="magenta", width=1, arrow_size=6)
        elif relation.relation == "chunk_to_line":
            source = chunk_by_id[relation.source_id]
            target = line_by_id[relation.target_id]
            start = chunk_center(source)
            end = line_center(target)
            draw_arrow(draw, start, end, color="deepskyblue", width=1, arrow_size=6)
        elif relation.relation == "next_chunk":
            source = chunk_by_id[relation.source_id]
            target = chunk_by_id[relation.target_id]
            start = chunk_center(source)
            end = chunk_center(target)
            draw_arrow(draw, start, end, color="gold", width=2, arrow_size=7)

    for relation in region_relations or []:
        if relation.relation == "line_to_region":
            source = line_by_id.get(relation.source_id)
            target = region_by_id.get(relation.target_id)
            if source is None or target is None:
                continue
            draw_arrow(draw, line_center(source), bbox_center(target.bbox), color="orchid", width=1, arrow_size=6)
        elif relation.relation == "chunk_to_region":
            source = chunk_by_id.get(relation.source_id)
            target = region_by_id.get(relation.target_id)
            if source is None or target is None:
                continue
            draw_arrow(draw, chunk_center(source), bbox_center(target.bbox), color="tomato", width=2, arrow_size=7)
        elif relation.relation == "next_region":
            source = region_by_id.get(relation.source_id)
            target = region_by_id.get(relation.target_id)
            if source is None or target is None:
                continue
            draw_arrow(draw, bbox_center(source.bbox), bbox_center(target.bbox), color="limegreen", width=2, arrow_size=7)
        elif relation.relation == "chunk_to_region_lexical":
            source = chunk_by_id.get(relation.source_id)
            target = region_by_id.get(relation.target_id)
            if source is None or target is None:
                continue
            draw_arrow(draw, chunk_center(source), bbox_center(target.bbox), color="mediumorchid", width=2, arrow_size=7)

    return canvas


def draw_layered_graph(
    image: Image.Image,
    lines: list[Any],
    coarse_nodes: list[Any],
    fine_nodes: list[Any],
    region_relations: list[Any],
    hierarchical_relations: list[Any],
    show_text: bool,
    line_width: int,
) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    line_by_id = {line.id: line for line in lines}
    coarse_by_id = {node.id: node for node in coarse_nodes}
    fine_children: dict[str, list[Any]] = {}
    for fine_node in fine_nodes:
        fine_children.setdefault(fine_node.parent_id, []).append(fine_node)

    fine_box_by_id: dict[str, Any] = {}
    for parent_id, items in fine_children.items():
        for index, fine_node in enumerate(items):
            fine_box_by_id[fine_node.id] = virtual_fine_bbox(
                fine_node=fine_node,
                sibling_count=len(items),
                sibling_index=index,
            )

    for node in coarse_nodes:
        color = "orange" if node.modality == "paragraph" else "springgreen"
        original_bbox = node.metadata.get("original_bbox")
        if original_bbox is not None:
            draw.rectangle(
                [
                    (original_bbox.left, original_bbox.top),
                    (original_bbox.right, original_bbox.bottom),
                ],
                outline="yellow",
                width=1,
            )
        draw.rectangle(
            [(node.bbox.left, node.bbox.top), (node.bbox.right, node.bbox.bottom)],
            outline=color,
            width=line_width,
        )
        if show_text:
            label = f"{node.id} [{node.modality}]"
            draw.text(
                (node.bbox.left + 2, max(0, node.bbox.top - 12)),
                label,
                fill=color,
                font=font,
            )

    for fine_node in fine_nodes:
        fine_box = fine_box_by_id[fine_node.id]
        color = "cyan" if fine_node.modality == "sentence" else "magenta"
        if fine_node.modality == "patch":
            color = "deepskyblue"
        draw.rectangle(
            [(fine_box.left, fine_box.top), (fine_box.right, fine_box.bottom)],
            outline=color,
            width=1,
        )
        if show_text:
            label = fine_node.text[:32].replace("\n", " | ")
            draw.text(
                (fine_box.left + 2, max(0, fine_box.top + 2)),
                label,
                fill=color,
                font=font,
            )

    for relation in region_relations:
        if relation.relation != "line_to_region":
            continue
        line = line_by_id.get(relation.source_id)
        region_node = coarse_by_id.get(relation.target_id)
        if line is None or region_node is None:
            continue

        draw.rectangle(
            [(line.bbox.left, line.bbox.top), (line.bbox.right, line.bbox.bottom)],
            outline="deepskyblue",
            width=1,
        )
        draw_arrow(
            draw,
            line_center(line),
            bbox_center(region_node.bbox),
            color="orchid",
            width=1,
            arrow_size=6,
        )

    for relation in region_relations:
        if relation.relation != "chunk_to_region":
            continue
        chunk_node = coarse_by_id.get(relation.source_id)
        region_node = coarse_by_id.get(relation.target_id)
        if chunk_node is None or region_node is None:
            continue
        draw_arrow(
            draw,
            bbox_center(chunk_node.bbox),
            bbox_center(region_node.bbox),
            color="tomato",
            width=2,
            arrow_size=7,
        )

    for relation in region_relations:
        if relation.relation != "chunk_to_region_lexical":
            continue
        chunk_node = coarse_by_id.get(relation.source_id)
        region_node = coarse_by_id.get(relation.target_id)
        if chunk_node is None or region_node is None:
            continue
        draw_arrow(
            draw,
            bbox_center(chunk_node.bbox),
            bbox_center(region_node.bbox),
            color="mediumorchid",
            width=2,
            arrow_size=7,
        )

    for relation in region_relations:
        if relation.relation != "next_region":
            continue
        source = coarse_by_id.get(relation.source_id)
        target = coarse_by_id.get(relation.target_id)
        if source is None or target is None:
            continue
        draw_arrow(
            draw,
            bbox_center(source.bbox),
            bbox_center(target.bbox),
            color="limegreen",
            width=2,
            arrow_size=7,
        )

    for relation in hierarchical_relations:
        if relation.relation != "coarse_to_fine":
            continue
        parent = coarse_by_id.get(relation.source_id)
        fine_box = fine_box_by_id.get(relation.target_id)
        if parent is None or fine_box is None:
            continue
        draw_arrow(
            draw,
            bbox_center(parent.bbox),
            bbox_center(fine_box),
            color="yellow",
            width=1,
            arrow_size=6,
        )

    return canvas


def show_image(image: Image.Image, title: str) -> None:
    plt.figure(figsize=(14, 10))
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def print_region_analyses(regions: list[Any]) -> None:
    printed_any = False
    for region in regions:
        analysis = region.metadata.get("qwen_analysis", {})
        if not isinstance(analysis, dict) or not analysis:
            continue

        printed_any = True
        title = str(analysis.get("title_or_topic", "")).strip()
        summary = str(analysis.get("summary", "")).strip()
        structured = str(analysis.get("structured_content", "")).strip()

        print()
        print(f"[{region.id}] label={region.label} score={region.score:.2f}")
        if title:
            print(f"Topic: {title}")
        if summary:
            print(f"Summary: {summary}")
        if structured:
            print("Structured content:")
            print(structured)

    if not printed_any:
        print()
        print("No Qwen analysis found in region metadata.")


def main() -> None:
    args = parse_args()
    image_path = resolve_image_path(args.image)
    ocr_path = resolve_ocr_path(image_path, args.ocr)
    layout_path = resolve_optional_path(args.layout_json)

    image = Image.open(image_path)
    if args.plot_layered:
        pipeline = run_document_pipeline(
            image_path=image_path,
            ocr_path=ocr_path,
            page_number=args.page,
            layout_path=layout_path,
            detect_layout=args.detect_layout,
            layout_labels=[
                label.strip()
                for label in args.layout_labels.split(",")
                if label.strip()
            ],
            layout_threshold=args.layout_threshold,
            local_model_path=resolve_optional_path(args.local_model_path),
            include_visual_fine_nodes=False,
            analyze_regions_with_qwen=args.analyze_regions_with_qwen,
            qwen_labels=[
                label.strip()
                for label in args.layout_labels.split(",")
                if label.strip()
            ],
            qwen_model=args.qwen_model,
            qwen_device=args.qwen_device,
            qwen_dtype=args.qwen_dtype,
            qwen_max_new_tokens=args.qwen_max_new_tokens,
            qwen_temperature=args.qwen_temperature,
            qwen_crops_dir=resolve_optional_path(args.qwen_crops_dir),
        )
        plotted = draw_layered_graph(
            image=image,
            lines=pipeline.lines,
            coarse_nodes=pipeline.coarse_nodes,
            fine_nodes=pipeline.fine_nodes,
            region_relations=pipeline.region_relations,
            hierarchical_relations=pipeline.hierarchical_relations,
            show_text=not args.hide_text,
            line_width=args.line_width,
        )
        title = (
            f"{image_path.name} | coarse: {len(pipeline.coarse_nodes)} | "
            f"fine: {len(pipeline.fine_nodes)} | hierarchy: {len(pipeline.hierarchical_relations)} | "
            f"regions: {len(pipeline.regions)} | paragraph_lines: {len(pipeline.paragraph_lines)} | "
            f"region_relations: {len(pipeline.region_relations)}"
        )
        output_name = f"{image_path.stem}_layered_overlay.png"
        if args.analyze_regions_with_qwen:
            print_region_analyses(pipeline.regions)
    elif args.plot_graph:
        if layout_path is not None or args.detect_layout:
            pipeline = run_document_pipeline(
                image_path=image_path,
                ocr_path=ocr_path,
                page_number=args.page,
                layout_path=layout_path,
                detect_layout=args.detect_layout,
                layout_labels=[
                    label.strip()
                    for label in args.layout_labels.split(",")
                    if label.strip()
                ],
                layout_threshold=args.layout_threshold,
                local_model_path=resolve_optional_path(args.local_model_path),
                include_visual_fine_nodes=False,
                analyze_regions_with_qwen=args.analyze_regions_with_qwen,
                qwen_labels=[
                    label.strip()
                    for label in args.layout_labels.split(",")
                    if label.strip()
                ],
                qwen_model=args.qwen_model,
                qwen_device=args.qwen_device,
                qwen_dtype=args.qwen_dtype,
                qwen_max_new_tokens=args.qwen_max_new_tokens,
                qwen_temperature=args.qwen_temperature,
                qwen_crops_dir=resolve_optional_path(args.qwen_crops_dir),
            )
            lines_for_graph = pipeline.paragraph_lines
            chunks = pipeline.chunks
            line_relations = pipeline.line_relations
            graph_relations = pipeline.chunk_relations
            regions = pipeline.regions
            region_relations = pipeline.region_relations
        else:
            _, lines_for_graph, line_relations, chunks, graph_relations = build_text_graph_from_ocr(
                ocr_path=ocr_path,
                page_number=args.page,
            )
            regions = []
            region_relations = []

        plotted = draw_text_graph(
            image=image,
            lines=lines_for_graph,
            chunks=chunks,
            next_line_relations=line_relations,
            graph_relations=graph_relations,
            regions=regions,
            region_relations=region_relations,
            show_text=not args.hide_text,
            show_relation_text=not args.hide_relation_text,
            line_width=args.line_width,
        )
        title = (
            f"{image_path.name} | lines: {len(lines_for_graph)} | "
            f"next_line: {len(line_relations)} | chunks: {len(chunks)} | "
            f"chunk_rel: {len(graph_relations)} | regions: {len(regions)} | "
            f"region_rel: {len(region_relations)}"
        )
        output_name = f"{image_path.stem}_graph_overlay.png"
    elif args.plot_chunks:
        _, grouped_lines, relations, chunks = build_text_chunks_from_ocr(
            ocr_path=ocr_path,
            page_number=args.page,
        )
        plotted = draw_chunks(
            image=image,
            chunks=chunks,
            show_text=not args.hide_text,
            line_width=args.line_width,
        )
        title = (
            f"{image_path.name} | grouped lines: {len(grouped_lines)} | "
            f"next_line: {len(relations)} | chunks: {len(chunks)}"
        )
        output_name = f"{image_path.stem}_chunk_overlay.png"
    elif args.plot_next_line:
        _, grouped_lines, relations = build_text_relations_from_ocr(
            ocr_path=ocr_path,
            page_number=args.page,
        )
        relation_lookup = {
            relation.source_id: relation.target_id
            for relation in relations
            if relation.relation == "next_line"
        }
        relation_scores = {
            relation.source_id: relation.score
            for relation in relations
            if relation.relation == "next_line"
        }
        plotted = draw_grouped_lines_and_relations(
            image=image,
            lines=grouped_lines,
            show_text=not args.hide_text,
            show_relation_text=not args.hide_relation_text,
            line_width=args.line_width,
            relation_lookup=relation_lookup,
            relation_scores=relation_scores,
        )
        title = f"{image_path.name} | grouped lines: {len(grouped_lines)} | next_line: {len(relations)}"
        output_name = f"{image_path.stem}_next_line_overlay.png"
    else:
        lines = load_ocr_lines(ocr_path, page_number=args.page)
        plotted = draw_ocr_lines(
            image=image,
            lines=lines,
            show_text=not args.hide_text,
            line_width=args.line_width,
        )
        title = f"{image_path.name} | OCR lines: {len(lines)}"
        output_name = f"{image_path.stem}_ocr_overlay.png"

    if args.output is not None:
        output_path = args.output
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
    else:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = DEFAULT_OUTPUT_DIR / output_name

    plotted.save(output_path)
    print(f"Saved overlay to: {output_path}")
    if not args.no_show:
        show_image(plotted, title=title)


if __name__ == "__main__":
    main()

