from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.qwen_vl_region_analysis import annotate_detections_with_qwen
from src.extract.doclayout_yolo import (
    attach_ocr_content,
    parse_ocr_lines,
    render_bboxes,
    run_doclayout_yolo,
)
from src.extract.document_pipeline import load_layout_detections
from src.utils.config import DEFAULT_QWEN_VL_MODEL

DEFAULT_OCR_DIR = PROJECT_ROOT / "dataset" / "spdocvqa" / "spdocvqa_ocr"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect document regions and analyze them with Qwen2.5-VL."
    )
    parser.add_argument(
        "image",
        type=Path,
        help="Path to the document image.",
    )
    parser.add_argument(
        "--ocr",
        type=Path,
        default=None,
        help="Optional OCR JSON path. Defaults to spdocvqa_ocr/<image_stem>.json.",
    )
    parser.add_argument(
        "--layout-json",
        type=Path,
        default=None,
        help="Optional existing layout JSON. If omitted, run DocLayout-YOLO.",
    )
    parser.add_argument(
        "--detect-layout",
        action="store_true",
        help="Run DocLayout-YOLO instead of reading layout JSON.",
    )
    parser.add_argument(
        "--layout-threshold",
        type=float,
        default=0.25,
        help="Detection threshold for DocLayout-YOLO.",
    )
    parser.add_argument(
        "--layout-labels",
        default="table,figure,chart,image",
        help="Comma-separated labels to keep when detecting layout.",
    )
    parser.add_argument(
        "--local-model-path",
        default="",
        help="Optional local DocLayout-YOLO model path.",
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
        "--crops-dir",
        type=Path,
        default=None,
        help="Optional directory to save cropped regions sent to Qwen.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional output path for enriched layout JSON.",
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=None,
        help="Optional output path for region bbox preview.",
    )
    return parser.parse_args()


def resolve_optional_ocr_path(image_path: Path, ocr_path: Path | None) -> Path | None:
    if ocr_path is not None:
        resolved = ocr_path if ocr_path.is_absolute() else PROJECT_ROOT / ocr_path
        if not resolved.exists():
            raise FileNotFoundError(f"OCR JSON not found: {resolved}")
        return resolved
    candidate = DEFAULT_OCR_DIR / f"{image_path.stem}.json"
    return candidate if candidate.exists() else None


def resolve_output_path(image_path: Path, output_path: Path | None, suffix: str) -> Path:
    if output_path is not None:
        return output_path
    return DEFAULT_OUTPUT_DIR / suffix / f"{image_path.stem}.json"


def main() -> None:
    args = parse_args()
    image_path = args.image if args.image.is_absolute() else PROJECT_ROOT / args.image
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    ocr_path = resolve_optional_ocr_path(image_path, args.ocr)
    if args.layout_json is not None:
        detections = load_layout_detections(args.layout_json)
    else:
        labels = [item.strip() for item in args.layout_labels.split(",") if item.strip()]
        result = run_doclayout_yolo(
            image_path=image_path,
            labels=labels,
            score_threshold=args.layout_threshold,
            local_model_path=args.local_model_path or None,
        )
        detections = result.get("detections", [])

    if ocr_path is not None:
        ocr_lines = parse_ocr_lines(ocr_path)
        attach_ocr_content(detections, ocr_lines)

    analyses = annotate_detections_with_qwen(
        image_path=image_path,
        detections=detections,
        model=args.qwen_model,
        device=args.qwen_device,
        dtype=args.qwen_dtype,
        max_new_tokens=args.qwen_max_new_tokens,
        temperature=args.qwen_temperature,
        save_crops_dir=args.crops_dir,
        merge_into_content=True,
    )

    output_path = resolve_output_path(
        image_path=image_path,
        output_path=args.json_out,
        suffix="layout",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "image": str(image_path),
        "ocr": str(ocr_path) if ocr_path is not None else "",
        "num_detections": len(detections),
        "num_analyses": len(analyses),
        "detections": detections,
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.plot_out is not None:
        render_bboxes(
            image_path=image_path,
            detections=detections,
            out_path=args.plot_out,
        )

    print(f"Saved: {output_path}")
    print(f"Detections: {len(detections)}")
    print(f"Qwen analyses: {len(analyses)}")
    for analysis in analyses:
        print(f"- {analysis.region_id} [{analysis.label}] -> {analysis.summary[:120]}")


if __name__ == "__main__":
    main()

