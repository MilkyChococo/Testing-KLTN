from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.type.base import BoundingBox, OCRLine, OCRWord

from .ocr_parser import load_infographic_words_and_lines


@dataclass(slots=True)
class InfographicChunk:
    id: str
    page: int
    text: str
    bbox: BoundingBox
    lines: list[OCRLine]


@dataclass(slots=True, frozen=True)
class InfographicChunkConfig:
    max_vertical_gap_ratio: float = 1.2
    max_horizontal_gap_ratio: float = 0.10
    max_left_delta_ratio: float = 0.04
    min_horizontal_overlap_ratio: float = 0.35
    min_merge_score: float = 0.55


DEFAULT_INFOGRAPHIC_CHUNK_CONFIG = InfographicChunkConfig()


def _line_merge_score(
    source: OCRLine,
    target: OCRLine,
    max_vertical_gap_ratio: float,
    max_horizontal_gap_ratio: float,
    max_left_delta_ratio: float,
    min_horizontal_overlap_ratio: float,
) -> float:
    avg_height = max((source.bbox.height + target.bbox.height) / 2.0, 1.0)
    avg_width = max((source.bbox.width + target.bbox.width) / 2.0, 1.0)

    vertical_gap_ratio = source.bbox.vertical_gap(target.bbox) / avg_height
    if vertical_gap_ratio > max_vertical_gap_ratio:
        return 0.0

    horizontal_gap = max(0.0, max(source.bbox.left, target.bbox.left) - min(source.bbox.right, target.bbox.right))
    horizontal_gap_ratio = horizontal_gap / avg_width
    if horizontal_gap_ratio > max_horizontal_gap_ratio:
        return 0.0

    page_width = max(source.bbox.right, target.bbox.right, 1.0)
    left_delta_ratio = abs(source.bbox.left - target.bbox.left) / page_width
    center_delta_ratio = abs(source.bbox.center_x - target.bbox.center_x) / page_width
    overlap_ratio = source.bbox.horizontal_overlap_ratio(target.bbox)
    height_ratio = min(source.bbox.height, target.bbox.height) / max(
        max(source.bbox.height, target.bbox.height), 1.0
    )

    aligned = (
        overlap_ratio >= min_horizontal_overlap_ratio
        or left_delta_ratio <= max_left_delta_ratio
        or center_delta_ratio <= max_left_delta_ratio
    )
    if not aligned:
        return 0.0

    gap_score = max(0.0, 1.0 - (vertical_gap_ratio / max(max_vertical_gap_ratio, 1e-6)))
    alignment_score = max(
        overlap_ratio,
        max(0.0, 1.0 - (left_delta_ratio / max(max_left_delta_ratio, 1e-6))),
        max(0.0, 1.0 - (center_delta_ratio / max(max_left_delta_ratio, 1e-6))),
    )
    return 0.45 * gap_score + 0.35 * alignment_score + 0.20 * height_ratio


def build_chunks_from_lines_cluster(
    lines: list[OCRLine],
    config: InfographicChunkConfig = DEFAULT_INFOGRAPHIC_CHUNK_CONFIG,
) -> list[InfographicChunk]:
    ordered_lines = sorted(lines, key=lambda item: (item.page, item.bbox.top, item.bbox.left))
    if not ordered_lines:
        return []

    parent = list(range(len(ordered_lines)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left in range(len(ordered_lines)):
        source = ordered_lines[left]
        for right in range(left + 1, len(ordered_lines)):
            target = ordered_lines[right]
            if source.page != target.page:
                continue
            avg_height = max((source.bbox.height + target.bbox.height) / 2.0, 1.0)
            if (target.bbox.top - source.bbox.bottom) > (
                config.max_vertical_gap_ratio * avg_height * 1.5
            ):
                break
            score = _line_merge_score(
                source=source,
                target=target,
                max_vertical_gap_ratio=config.max_vertical_gap_ratio,
                max_horizontal_gap_ratio=config.max_horizontal_gap_ratio,
                max_left_delta_ratio=config.max_left_delta_ratio,
                min_horizontal_overlap_ratio=config.min_horizontal_overlap_ratio,
            )
            if score >= config.min_merge_score:
                union(left, right)

    groups: dict[int, list[OCRLine]] = {}
    for index, line in enumerate(ordered_lines):
        groups.setdefault(find(index), []).append(line)

    chunks: list[InfographicChunk] = []
    for chunk_index, chunk_lines in enumerate(
        sorted(
            groups.values(),
            key=lambda group: (
                group[0].page,
                min(item.bbox.top for item in group),
                min(item.bbox.left for item in group),
            ),
        )
    ):
        chunk_lines = sorted(chunk_lines, key=lambda item: (item.bbox.top, item.bbox.left))
        chunks.append(
            InfographicChunk(
                id=f"p{chunk_lines[0].page}_chunk_{chunk_index:03d}",
                page=chunk_lines[0].page,
                text="\n".join(line.text for line in chunk_lines if line.text).strip(),
                bbox=BoundingBox.merge(line.bbox for line in chunk_lines),
                lines=chunk_lines,
            )
        )
    return chunks


def build_infographic_chunks_from_ocr(
    image_path: str | Path,
    ocr_path: str | Path,
    page_number: int = 1,
    config: InfographicChunkConfig = DEFAULT_INFOGRAPHIC_CHUNK_CONFIG,
) -> tuple[list[OCRWord], list[OCRLine], list[InfographicChunk]]:
    words, lines = load_infographic_words_and_lines(
        image_path=image_path,
        ocr_path=ocr_path,
        page_number=page_number,
    )
    chunks = build_chunks_from_lines_cluster(lines, config=config)
    return words, lines, chunks
