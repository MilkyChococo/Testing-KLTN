from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extract_infor import (
    DEFAULT_INFOGRAPHIC_CHUNK_CONFIG,
    DEFAULT_INFOGRAPHIC_RELATION_CONFIG,
    InfographicChunkConfig,
    InfographicRelationConfig,
    build_infographic_chunks_from_ocr,
    build_infographic_text_relations,
)


DEFAULT_IMAGE_DIR = PROJECT_ROOT / "dataset" / "infographic" / "infographicsvqa_images"
DEFAULT_OCR_DIR = PROJECT_ROOT / "dataset" / "infographic" / "infographicsvqa_ocr"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "visualizations"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot infographic OCR line boxes and clustered chunks."
    )
    parser.add_argument("image", type=Path, help="Path to the infographic image.")
    parser.add_argument(
        "--ocr",
        type=Path,
        default=None,
        help="Optional OCR JSON path. Defaults to the same stem in infographicsvqa_ocr.",
    )
    parser.add_argument(
        "--hide-line-text",
        action="store_true",
        help="Hide line labels.",
    )
    parser.add_argument(
        "--hide-chunk-text",
        action="store_true",
        help="Hide chunk labels.",
    )
    parser.add_argument(
        "--line-limit",
        type=int,
        default=None,
        help="Optional maximum number of lines to draw.",
    )
    parser.add_argument(
        "--chunk-limit",
        type=int,
        default=None,
        help="Optional maximum number of chunks to draw.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Build the figure without opening a window. Useful for quick verification.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image path. Defaults to artifacts/visualizations/<stem>_infographic_chunks.png",
    )
    parser.add_argument(
        "--max-vertical-gap-ratio",
        type=float,
        default=DEFAULT_INFOGRAPHIC_CHUNK_CONFIG.max_vertical_gap_ratio,
        help="Max vertical gap ratio for merging lines into a chunk.",
    )
    parser.add_argument(
        "--max-horizontal-gap-ratio",
        type=float,
        default=DEFAULT_INFOGRAPHIC_CHUNK_CONFIG.max_horizontal_gap_ratio,
        help="Max horizontal gap ratio for merging lines into a chunk.",
    )
    parser.add_argument(
        "--max-left-delta-ratio",
        type=float,
        default=DEFAULT_INFOGRAPHIC_CHUNK_CONFIG.max_left_delta_ratio,
        help="Max left/center alignment delta ratio for merging lines into a chunk.",
    )
    parser.add_argument(
        "--min-horizontal-overlap-ratio",
        type=float,
        default=DEFAULT_INFOGRAPHIC_CHUNK_CONFIG.min_horizontal_overlap_ratio,
        help="Min horizontal overlap ratio used as an alignment cue.",
    )
    parser.add_argument(
        "--min-merge-score",
        type=float,
        default=DEFAULT_INFOGRAPHIC_CHUNK_CONFIG.min_merge_score,
        help="Minimum pairwise merge score for clustering lines into chunks.",
    )
    parser.add_argument(
        "--hide-line-relations",
        action="store_true",
        help="Hide line-to-line relations.",
    )
    parser.add_argument(
        "--hide-membership-relations",
        action="store_true",
        help="Hide line-to-chunk relations.",
    )
    parser.add_argument(
        "--hide-chunk-relations",
        action="store_true",
        help="Hide chunk-to-chunk relations.",
    )
    parser.add_argument(
        "--next-line-max-vertical-gap-ratio",
        type=float,
        default=DEFAULT_INFOGRAPHIC_RELATION_CONFIG.next_line_max_vertical_gap_ratio,
        help="Max vertical gap ratio for next_line relation.",
    )
    parser.add_argument(
        "--near-line-max-center-distance-ratio",
        type=float,
        default=DEFAULT_INFOGRAPHIC_RELATION_CONFIG.near_line_max_center_distance_ratio,
        help="Max center distance ratio for near_line relation.",
    )
    parser.add_argument(
        "--next-chunk-max-vertical-gap-ratio",
        type=float,
        default=DEFAULT_INFOGRAPHIC_RELATION_CONFIG.next_chunk_max_vertical_gap_ratio,
        help="Max vertical gap ratio for next_chunk relation.",
    )
    parser.add_argument(
        "--near-chunk-max-center-distance-ratio",
        type=float,
        default=DEFAULT_INFOGRAPHIC_RELATION_CONFIG.near_chunk_max_center_distance_ratio,
        help="Max center distance ratio for near_chunk relation.",
    )
    parser.add_argument(
        "--cross-chunk-only",
        action="store_true",
        help="Draw only line relations that connect different chunks.",
    )
    return parser.parse_args()


def resolve_path(path: Path, default_dir: Path | None = None) -> Path:
    if path.is_absolute() and path.exists():
        return path.resolve()
    if path.exists():
        return path.resolve()
    candidate = PROJECT_ROOT / path
    if candidate.exists():
        return candidate.resolve()
    if default_dir is not None:
        candidate = default_dir / path.name
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Path not found: {path}")


def resolve_image_path(image_path: Path) -> Path:
    return resolve_path(image_path, default_dir=DEFAULT_IMAGE_DIR)


def resolve_ocr_path(image_path: Path, ocr_path: Path | None) -> Path:
    if ocr_path is not None:
        return resolve_path(ocr_path, default_dir=DEFAULT_OCR_DIR)
    candidate = DEFAULT_OCR_DIR / f"{image_path.stem}.json"
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(
        f"Could not infer OCR JSON for image stem '{image_path.stem}' in {DEFAULT_OCR_DIR}"
    )


def resolve_output_path(image_path: Path, output: Path | None) -> Path:
    if output is not None:
        return output if output.is_absolute() else (PROJECT_ROOT / output).resolve()
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return (DEFAULT_OUTPUT_DIR / f"{image_path.stem}_infographic_chunks.png").resolve()


def enable_scroll_zoom(axis: plt.Axes, base_scale: float = 1.2) -> None:
    def on_scroll(event) -> None:
        if event.inaxes != axis or event.xdata is None or event.ydata is None:
            return

        current_xlim = axis.get_xlim()
        current_ylim = axis.get_ylim()
        xdata = float(event.xdata)
        ydata = float(event.ydata)

        if event.button == "up":
            scale_factor = 1.0 / base_scale
        elif event.button == "down":
            scale_factor = base_scale
        else:
            scale_factor = 1.0

        new_width = (current_xlim[1] - current_xlim[0]) * scale_factor
        new_height = (current_ylim[1] - current_ylim[0]) * scale_factor

        rel_x = 0.5 if current_xlim[1] == current_xlim[0] else (xdata - current_xlim[0]) / (current_xlim[1] - current_xlim[0])
        rel_y = 0.5 if current_ylim[1] == current_ylim[0] else (ydata - current_ylim[0]) / (current_ylim[1] - current_ylim[0])

        axis.set_xlim([xdata - new_width * rel_x, xdata + new_width * (1.0 - rel_x)])
        axis.set_ylim([ydata - new_height * rel_y, ydata + new_height * (1.0 - rel_y)])
        axis.figure.canvas.draw_idle()

    axis.figure.canvas.mpl_connect("scroll_event", on_scroll)


def draw_relation(
    axis: plt.Axes,
    source_xy: tuple[float, float],
    target_xy: tuple[float, float],
    *,
    color: str,
    linewidth: float,
    alpha: float,
    linestyle: str = "-",
    arrow: bool = False,
    curve: float = 0.0,
    zorder: float = 2.0,
) -> None:
    if arrow:
        axis.annotate(
            "",
            xy=target_xy,
            xytext=source_xy,
            arrowprops={
                "arrowstyle": "->",
                "color": color,
                "lw": linewidth,
                "alpha": alpha,
                "linestyle": linestyle,
                "shrinkA": 3,
                "shrinkB": 3,
                "connectionstyle": f"arc3,rad={curve}",
                "zorder": zorder,
            },
        )
        return
    axis.plot(
        [source_xy[0], target_xy[0]],
        [source_xy[1], target_xy[1]],
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        linestyle=linestyle,
        zorder=zorder,
    )


def draw_relation_label(
    axis: plt.Axes,
    source_xy: tuple[float, float],
    target_xy: tuple[float, float],
    text: str,
    *,
    color: str,
    fontsize: int = 8,
    zorder: float = 8.0,
) -> None:
    mid_x = (source_xy[0] + target_xy[0]) / 2.0
    mid_y = (source_xy[1] + target_xy[1]) / 2.0
    axis.text(
        mid_x,
        mid_y,
        text,
        fontsize=fontsize,
        color=color,
        ha="center",
        va="center",
        bbox={"facecolor": "black", "alpha": 0.75, "pad": 1.0},
        zorder=zorder,
    )


def main() -> None:
    args = parse_args()
    image_path = resolve_image_path(args.image)
    ocr_path = resolve_ocr_path(image_path, args.ocr)
    output_path = resolve_output_path(image_path, args.output)
    config = InfographicChunkConfig(
        max_vertical_gap_ratio=args.max_vertical_gap_ratio,
        max_horizontal_gap_ratio=args.max_horizontal_gap_ratio,
        max_left_delta_ratio=args.max_left_delta_ratio,
        min_horizontal_overlap_ratio=args.min_horizontal_overlap_ratio,
        min_merge_score=args.min_merge_score,
    )
    relation_config = InfographicRelationConfig(
        next_line_max_vertical_gap_ratio=args.next_line_max_vertical_gap_ratio,
        next_line_max_left_delta_ratio=DEFAULT_INFOGRAPHIC_RELATION_CONFIG.next_line_max_left_delta_ratio,
        next_line_min_horizontal_overlap_ratio=DEFAULT_INFOGRAPHIC_RELATION_CONFIG.next_line_min_horizontal_overlap_ratio,
        near_line_max_center_distance_ratio=args.near_line_max_center_distance_ratio,
        near_line_top_k=DEFAULT_INFOGRAPHIC_RELATION_CONFIG.near_line_top_k,
        next_chunk_max_vertical_gap_ratio=args.next_chunk_max_vertical_gap_ratio,
        next_chunk_max_left_delta_ratio=DEFAULT_INFOGRAPHIC_RELATION_CONFIG.next_chunk_max_left_delta_ratio,
        next_chunk_min_horizontal_overlap_ratio=DEFAULT_INFOGRAPHIC_RELATION_CONFIG.next_chunk_min_horizontal_overlap_ratio,
        near_chunk_max_center_distance_ratio=args.near_chunk_max_center_distance_ratio,
        near_chunk_top_k=DEFAULT_INFOGRAPHIC_RELATION_CONFIG.near_chunk_top_k,
    )

    _, lines, chunks = build_infographic_chunks_from_ocr(
        image_path=image_path,
        ocr_path=ocr_path,
        page_number=1,
        config=config,
    )
    line_relations, graph_relations = build_infographic_text_relations(
        lines=lines,
        chunks=chunks,
        config=relation_config,
    )

    image = Image.open(image_path).convert("RGB")
    figure, axis = plt.subplots(figsize=(14, 10))
    axis.imshow(image)
    axis.set_title(f"{image_path.name} | lines={len(lines)} | chunks={len(chunks)}")
    axis.axis("off")
    enable_scroll_zoom(axis)

    display_lines = lines if args.line_limit is None else lines[: args.line_limit]
    display_chunks = chunks if args.chunk_limit is None else chunks[: args.chunk_limit]
    display_line_ids = {line.id for line in display_lines}
    display_chunk_ids = {chunk.id for chunk in display_chunks}
    line_by_id = {line.id: line for line in lines}
    chunk_by_id = {chunk.id: chunk for chunk in chunks}
    line_to_chunk = {
        line.id: chunk.id
        for chunk in chunks
        for line in chunk.lines
    }

    if not args.hide_line_relations:
        for relation in line_relations:
            if relation.source_id not in display_line_ids or relation.target_id not in display_line_ids:
                continue
            if args.cross_chunk_only and (
                line_to_chunk.get(relation.source_id) == line_to_chunk.get(relation.target_id)
            ):
                continue
            source = line_by_id[relation.source_id]
            target = line_by_id[relation.target_id]
            source_xy = (source.bbox.center_x, source.bbox.center_y)
            target_xy = (target.bbox.center_x, target.bbox.center_y)
            if relation.relation == "next_line":
                draw_relation(
                    axis,
                    source_xy,
                    target_xy,
                    color="lime",
                    linewidth=0.9,
                    alpha=0.22,
                    arrow=True,
                    curve=0.05,
                    zorder=1.0,
                )
            else:
                draw_relation(
                    axis,
                    source_xy,
                    target_xy,
                    color="violet",
                    linewidth=1.0,
                    alpha=0.32,
                    linestyle="--",
                    zorder=1.0,
                )

    for index, line in enumerate(display_lines):
        bbox = line.bbox
        axis.add_patch(
            patches.Rectangle(
                (bbox.left, bbox.top),
                bbox.width,
                bbox.height,
                linewidth=1.0,
                edgecolor="deepskyblue",
                facecolor="none",
                zorder=3.0,
            )
        )
        if not args.hide_line_text:
            axis.text(
                bbox.left,
                max(0.0, bbox.top - 3.0),
                f"L{index}",
                fontsize=7,
                color="deepskyblue",
                bbox={"facecolor": "black", "alpha": 0.55, "pad": 1.5},
                zorder=4.0,
            )

    for index, chunk in enumerate(display_chunks):
        bbox = chunk.bbox
        axis.add_patch(
            patches.Rectangle(
                (bbox.left, bbox.top),
                bbox.width,
                bbox.height,
                linewidth=2.6,
                edgecolor="orange",
                facecolor=(1.0, 0.55, 0.0, 0.06),
                zorder=5.0,
            )
        )
        axis.scatter(
            [bbox.center_x],
            [bbox.center_y],
            s=42,
            c="red",
            edgecolors="white",
            linewidths=0.9,
            zorder=7.0,
        )
        if not args.hide_chunk_text:
            axis.text(
                bbox.left,
                bbox.top + 10.0,
                f"C{index}",
                fontsize=10,
                color="orange",
                bbox={"facecolor": "black", "alpha": 0.80, "pad": 1.8},
                zorder=8.0,
            )

    if not args.hide_membership_relations:
        for relation in graph_relations:
            if relation.relation != "line_to_chunk":
                continue
            if relation.source_id not in display_line_ids or relation.target_id not in display_chunk_ids:
                continue
            source = line_by_id[relation.source_id]
            target = chunk_by_id[relation.target_id]
            draw_relation(
                axis,
                (source.bbox.center_x, source.bbox.center_y),
                (target.bbox.center_x, target.bbox.center_y),
                color="gold",
                linewidth=1.6,
                alpha=0.70,
                linestyle=":",
                zorder=6.0,
            )

    if not args.hide_chunk_relations:
        for relation in graph_relations:
            if relation.source_id not in display_chunk_ids or relation.target_id not in display_chunk_ids:
                continue
            if relation.relation not in {"next_chunk", "near_chunk"}:
                continue
            source = chunk_by_id[relation.source_id]
            target = chunk_by_id[relation.target_id]
            source_xy = (source.bbox.center_x, source.bbox.center_y)
            target_xy = (target.bbox.center_x, target.bbox.center_y)
            if relation.relation == "next_chunk":
                draw_relation(
                    axis,
                    source_xy,
                    target_xy,
                    color="tomato",
                    linewidth=3.0,
                    alpha=0.92,
                    arrow=True,
                    curve=0.12,
                    zorder=9.0,
                )
                draw_relation_label(
                    axis,
                    source_xy,
                    target_xy,
                    f"{relation.score:.2f}",
                    color="tomato",
                    fontsize=8,
                    zorder=10.0,
                )
            else:
                draw_relation(
                    axis,
                    source_xy,
                    target_xy,
                    color="springgreen",
                    linewidth=1.6,
                    alpha=0.50,
                    linestyle="--",
                    zorder=8.0,
                )

    print(f"Image: {image_path}")
    print(f"OCR: {ocr_path}")
    print(f"Lines: {len(lines)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Chunk config: {config}")
    print(
        "Relations: "
        f"line={len(line_relations)} "
        f"membership={sum(1 for item in graph_relations if item.relation == 'line_to_chunk')} "
        f"chunk={sum(1 for item in graph_relations if item.relation in {'next_chunk', 'near_chunk'})}"
    )
    print(f"Relation config: {relation_config}")
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved overlay to: {output_path}")
    print("Use mouse wheel to zoom in/out around the cursor.")
    if not args.no_show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
