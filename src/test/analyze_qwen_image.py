from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.qwen_vl_region_analysis import analyze_region_with_qwen
from src.utils.config import DEFAULT_QWEN_VL_MODEL


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "qwen"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a pre-cropped image region with Qwen2.5-VL."
    )
    parser.add_argument(
        "image",
        type=Path,
        help="Path to a cropped table/figure/chart/image region.",
    )
    parser.add_argument(
        "--label",
        default="image",
        choices=["table", "figure", "chart", "image"],
        help="Semantic label for the cropped image region.",
    )
    parser.add_argument(
        "--qwen-model",
        default=DEFAULT_QWEN_VL_MODEL,
        help="Qwen2.5-VL model name.",
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
        help="Max generated tokens for Qwen analysis.",
    )
    parser.add_argument(
        "--qwen-temperature",
        type=float,
        default=0.1,
        help="Temperature for Qwen analysis.",
    )
    parser.add_argument(
        "--ocr-text",
        default="",
        help="Optional OCR text to provide as extra context for the cropped image.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to save the Qwen JSON analysis.",
    )
    parser.add_argument(
        "--text-out",
        type=Path,
        default=None,
        help="Optional path to save the merged graph text.",
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
    raise FileNotFoundError(f"Image not found: {image_path}")


def default_json_out(image_path: Path) -> Path:
    return DEFAULT_OUTPUT_DIR / f"{image_path.stem}.json"


def default_text_out(image_path: Path) -> Path:
    return DEFAULT_OUTPUT_DIR / f"{image_path.stem}.txt"


def main() -> None:
    args = parse_args()
    image_path = resolve_image_path(args.image)
    image = Image.open(image_path).convert("RGB")

    analysis = analyze_region_with_qwen(
        image=image,
        region_id=image_path.stem,
        label=args.label,
        ocr_text=args.ocr_text.strip(),
        model=args.qwen_model,
        device=args.qwen_device,
        dtype=args.qwen_dtype,
        max_new_tokens=args.qwen_max_new_tokens,
        temperature=args.qwen_temperature,
    )

    merged_text = analysis.to_graph_text(ocr_text=args.ocr_text.strip())

    json_out = args.json_out or default_json_out(image_path)
    text_out = args.text_out or default_text_out(image_path)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    text_out.parent.mkdir(parents=True, exist_ok=True)

    json_payload = {
        "image": str(image_path),
        "label": args.label,
        "analysis": analysis.to_payload(),
        "graph_text": merged_text,
    }
    json_out.write_text(
        json.dumps(json_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    text_out.write_text(merged_text, encoding="utf-8")

    print(f"Saved JSON: {json_out}")
    print(f"Saved text: {text_out}")
    print()
    print("Summary:")
    print(analysis.summary)
    if analysis.structured_content:
        print()
        print("Structured content:")
        print(analysis.structured_content)


if __name__ == "__main__":
    main()

