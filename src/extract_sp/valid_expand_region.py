from __future__ import annotations

from dataclasses import dataclass

from src.type.base import BoundingBox, OCRLine

from .region_to_chunk import LayoutRegion


@dataclass(slots=True)
class ExpandedRegion:
    region_id: str
    label: str
    num_lines: int
    line_ids: list[str]
    expanded_bbox: BoundingBox
    original_bbox: BoundingBox
    min_line_left: float
    max_line_right: float


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


def line_region_overlap_ratio(line: OCRLine, region: LayoutRegion) -> float:
    line_area = _area(line.bbox)
    if line_area <= 0.0:
        return 0.0
    return _inter_area(line.bbox, region.bbox) / line_area


def line_belongs_to_region(
    line: OCRLine,
    region: LayoutRegion,
    min_line_overlap_ratio: float = 0.5,
) -> bool:
    return line_region_overlap_ratio(line, region) >= min_line_overlap_ratio


def select_lines_in_region(
    lines: list[OCRLine],
    region: LayoutRegion,
    min_line_overlap_ratio: float = 0.5,
) -> list[OCRLine]:
    return [
        line
        for line in lines
        if line_belongs_to_region(
            line=line,
            region=region,
            min_line_overlap_ratio=min_line_overlap_ratio,
        )
    ]


def expand_bbox_horizontally_to_line(
    current_box: BoundingBox,
    line_box: BoundingBox,
    horizontal_padding: float = 0.0,
    clamp_box: BoundingBox | None = None,
) -> BoundingBox:
    expanded_left = min(current_box.left, line_box.left) - horizontal_padding
    expanded_right = max(current_box.right, line_box.right) + horizontal_padding

    if clamp_box is not None:
        expanded_left = max(clamp_box.left, expanded_left)
        expanded_right = min(clamp_box.right, expanded_right)

    return BoundingBox(
        left=expanded_left,
        top=current_box.top,
        right=expanded_right,
        bottom=current_box.bottom,
    )


def expand_region_horizontally_to_lines(
    region_box: BoundingBox,
    line_boxes: list[BoundingBox],
    horizontal_padding: float = 0.0,
    clamp_box: BoundingBox | None = None,
) -> BoundingBox:
    if not line_boxes:
        return region_box

    min_line_left = min(box.left for box in line_boxes)
    max_line_right = max(box.right for box in line_boxes)
    expanded_left = min(region_box.left, min_line_left) - horizontal_padding
    expanded_right = max(region_box.right, max_line_right) + horizontal_padding

    if clamp_box is not None:
        expanded_left = max(clamp_box.left, expanded_left)
        expanded_right = min(clamp_box.right, expanded_right)

    return BoundingBox(
        left=expanded_left,
        top=region_box.top,
        right=expanded_right,
        bottom=region_box.bottom,
    )


def build_expanded_region(
    lines: list[OCRLine],
    region: LayoutRegion,
    min_line_overlap_ratio: float = 0.5,
    horizontal_padding: float = 0.0,
    clamp_to_region: bool = False,
) -> ExpandedRegion:
    selected_lines = select_lines_in_region(
        lines=lines,
        region=region,
        min_line_overlap_ratio=min_line_overlap_ratio,
    )
    line_boxes = [line.bbox for line in selected_lines]
    clamp_box = region.bbox if clamp_to_region else None
    expanded_bbox = expand_region_horizontally_to_lines(
        region_box=region.bbox,
        line_boxes=line_boxes,
        horizontal_padding=horizontal_padding,
        clamp_box=clamp_box,
    )

    if line_boxes:
        min_line_left = min(box.left for box in line_boxes)
        max_line_right = max(box.right for box in line_boxes)
    else:
        min_line_left = region.bbox.left
        max_line_right = region.bbox.right

    return ExpandedRegion(
        region_id=region.id,
        label=region.label,
        num_lines=len(selected_lines),
        line_ids=[line.id for line in selected_lines],
        expanded_bbox=expanded_bbox,
        original_bbox=region.bbox,
        min_line_left=min_line_left,
        max_line_right=max_line_right,
    )
