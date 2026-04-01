from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "dataset" / "infographic" / "infographicsvqa_images"
DEFAULT_OCR_DIR = PROJECT_ROOT / "dataset" / "infographic" / "infographicsvqa_ocr"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "visualizations"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot infographic OCR bounding boxes from PAGE/LINE/WORD JSON."
    )
    parser.add_argument(
        "image",
        type=Path,
        help="Path to the infographic image. If relative, it is resolved from the project root or infographic image folder.",
    )
    parser.add_argument(
        "--ocr",
        type=Path,
        default=None,
        help="Optional OCR JSON path. Defaults to the file with the same stem in infographicsvqa_ocr.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image path. Defaults to artifacts/visualizations/<stem>_infographic_ocr.png",
    )
    parser.add_argument(
        "--block-types",
        default="LINE",
        help="Comma-separated OCR block types to draw. Supported: PAGE,LINE,WORD. Default: LINE",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of blocks to draw per type.",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=2,
        help="Bounding box line width.",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=14,
        help="Label font size.",
    )
    parser.add_argument(
        "--hide-text",
        action="store_true",
        help="Draw only boxes without text labels.",
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


def load_payload(ocr_path: Path) -> dict[str, Any]:
    return json.loads(ocr_path.read_text(encoding="utf-8"))


def parse_block_types(raw: str) -> list[str]:
    valid = {"PAGE", "LINE", "WORD"}
    block_types = [item.strip().upper() for item in raw.split(",") if item.strip()]
    invalid = [item for item in block_types if item not in valid]
    if invalid:
        raise ValueError(f"Unsupported block types: {', '.join(invalid)}")
    return block_types or ["LINE"]


def get_font(font_size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        return ImageFont.load_default()


def bbox_to_xyxy(bbox: dict[str, float], width: int, height: int) -> tuple[int, int, int, int]:
    left = int(round(float(bbox["Left"]) * width))
    top = int(round(float(bbox["Top"]) * height))
    right = int(round((float(bbox["Left"]) + float(bbox["Width"])) * width))
    bottom = int(round((float(bbox["Top"]) + float(bbox["Height"])) * height))
    return left, top, right, bottom


def draw_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
) -> None:
    if not text:
        return
    x, y = xy
    left, top, right, bottom = draw.textbbox((x, y), text, font=font)
    draw.rectangle((left - 2, top - 1, right + 2, bottom + 1), fill="black")
    draw.text((x, y), text, fill=fill, font=font)


def plot_blocks(
    image: Image.Image,
    payload: dict[str, Any],
    block_types: list[str],
    limit: int | None,
    line_width: int,
    font_size: int,
    hide_text: bool,
) -> Image.Image:
    color_map = {
        "PAGE": "gold",
        "LINE": "deepskyblue",
        "WORD": "tomato",
    }
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = get_font(font_size)
    width, height = canvas.size

    for block_type in block_types:
        blocks = payload.get(block_type, [])
        if limit is not None:
            blocks = blocks[:limit]
        color = color_map[block_type]
        for idx, block in enumerate(blocks):
            geometry = block.get("Geometry", {})
            bbox = geometry.get("BoundingBox")
            if not bbox:
                continue
            x1, y1, x2, y2 = bbox_to_xyxy(bbox, width, height)
            draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
            if hide_text:
                continue
            text = block.get("Text", "")
            label = f"{block_type}:{idx}"
            if text:
                label = f"{label} {text}"
            draw_label(draw, (x1 + 2, max(0, y1 - font_size - 2)), label, font, color)
    return canvas


def resolve_output_path(image_path: Path, output: Path | None) -> Path:
    if output is not None:
        return output if output.is_absolute() else (PROJECT_ROOT / output).resolve()
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return (DEFAULT_OUTPUT_DIR / f"{image_path.stem}_infographic_ocr.png").resolve()


def main() -> None:
    args = parse_args()
    image_path = resolve_image_path(args.image)
    ocr_path = resolve_ocr_path(image_path, args.ocr)
    payload = load_payload(ocr_path)
    block_types = parse_block_types(args.block_types)

    image = Image.open(image_path).convert("RGB")
    overlay = plot_blocks(
        image=image,
        payload=payload,
        block_types=block_types,
        limit=args.limit,
        line_width=args.line_width,
        font_size=args.font_size,
        hide_text=args.hide_text,
    )

    output_path = resolve_output_path(image_path, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_path)

    print(f"Image: {image_path}")
    print(f"OCR: {ocr_path}")
    print(f"Block types: {', '.join(block_types)}")
    print(f"Saved overlay to: {output_path}")


if __name__ == "__main__":
    main()
