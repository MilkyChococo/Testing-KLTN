from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.index_offline_qwen import (
    DEFAULT_IMAGE_DIR,
    DEFAULT_OCR_DIR,
    DEFAULT_OUTPUT_ROOT,
    ensure_offline_store_qwen,
)
from src.api.qwen_vl_region_analysis import DEFAULT_QWEN_MODEL
from src.utils.config import DEFAULT_QWEN_EMBED_MODEL
from src.utils.io import load_json, save_json


DEFAULT_TEST_JSON = PROJECT_ROOT / "dataset" / "spdocvqa" / "spdocvqa_qas" / "test_v1.0.json"
DEFAULT_REPORT_JSON = PROJECT_ROOT / "artifacts" / "inference" / "qwen_store_build_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Qwen offline stores for all unique images referenced by an SP-DocVQA test JSON."
    )
    parser.add_argument(
        "test_json",
        type=Path,
        nargs="?",
        default=DEFAULT_TEST_JSON,
        help="Path to the SP-DocVQA test JSON file.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help="Directory containing source page images.",
    )
    parser.add_argument(
        "--ocr-dir",
        type=Path,
        default=DEFAULT_OCR_DIR,
        help="Directory containing OCR JSON files.",
    )
    parser.add_argument(
        "--store-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root folder where per-image Qwen stores are created.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=DEFAULT_REPORT_JSON,
        help="JSON report updated after each image.",
    )
    parser.add_argument("--force-reindex", action="store_true", help="Rebuild even if a complete store already exists.")
    parser.add_argument("--detect-layout", dest="detect_layout", action="store_true", help="Run DocLayout-YOLO.")
    parser.add_argument("--no-detect-layout", dest="detect_layout", action="store_false", help="Disable DocLayout-YOLO.")
    parser.add_argument("--layout-threshold", type=float, default=0.6, help="Layout detection threshold.")
    parser.add_argument("--layout-labels", default="table,figure,chart,image", help="Comma-separated layout labels to keep.")
    parser.add_argument("--local-model-path", type=Path, default=None, help="Optional local DocLayout-YOLO checkpoint.")
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL, help="Qwen model for region analysis.")
    parser.add_argument("--qwen-device", default="auto", help="Qwen device for region analysis.")
    parser.add_argument("--qwen-dtype", default="auto", help="Qwen dtype for region analysis.")
    parser.add_argument("--qwen-max-new-tokens", type=int, default=1024, help="Max output tokens for region analysis.")
    parser.add_argument("--qwen-temperature", type=float, default=0.1, help="Temperature for region analysis.")
    parser.add_argument("--qwen-crops-dir", type=Path, default=None, help="Optional directory to save Qwen crops.")
    parser.add_argument("--qwen-batch-size", type=int, default=1, help="Batch size for Qwen region analysis.")
    parser.add_argument("--embed-model", default=DEFAULT_QWEN_EMBED_MODEL, help="Embedding model.")
    parser.add_argument("--embed-device", default="auto", help="Embedding device.")
    parser.add_argument("--embed-dtype", default="auto", help="Embedding dtype.")
    parser.add_argument("--embed-batch-size", type=int, default=4, help="Embedding batch size.")
    parser.add_argument("--embed-max-length", type=int, default=8192, help="Embedding max length.")
    parser.add_argument("--embed-node-types", default="line,chunk,region,fine", help="Comma-separated node types to embed.")
    parser.add_argument(
        "--document-instruction",
        default="Represent this document graph node for retrieval in document question answering.",
        help="Instruction prepended before embedding node context.",
    )
    parser.add_argument("--lambda-hub", type=float, default=0.1, help="Hub penalty lambda.")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on number of unique images for debugging.")
    parser.set_defaults(detect_layout=True)
    return parser.parse_args()


def _resolve_dataset_image_path(image_ref: str, image_dir: Path) -> Path:
    return image_dir / Path(image_ref).name


def _resolve_dataset_ocr_path(image_ref: str, ocr_dir: Path) -> Path:
    return ocr_dir / f"{Path(image_ref).stem}.json"


def _build_unique_entries(entries: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique: list[dict] = []
    for entry in entries:
        image_ref = str(entry.get("image", "")).strip()
        if not image_ref or image_ref in seen:
            continue
        seen.add(image_ref)
        unique.append(entry)
    return unique


def main() -> None:
    args = parse_args()
    test_json_path = args.test_json if args.test_json.is_absolute() else (PROJECT_ROOT / args.test_json)
    image_dir = args.image_dir if args.image_dir.is_absolute() else (PROJECT_ROOT / args.image_dir)
    ocr_dir = args.ocr_dir if args.ocr_dir.is_absolute() else (PROJECT_ROOT / args.ocr_dir)
    store_root = args.store_root if args.store_root.is_absolute() else (PROJECT_ROOT / args.store_root)
    report_output = args.report_output if args.report_output.is_absolute() else (PROJECT_ROOT / args.report_output)

    payload = load_json(test_json_path)
    entries = _build_unique_entries(list(payload.get("data", [])))
    if args.limit > 0:
        entries = entries[: args.limit]

    report: list[dict[str, str | int]] = []
    progress = tqdm(entries, desc="Qwen offline index", unit="image")

    for entry in progress:
        image_ref = str(entry["image"]).strip()
        question_id = int(entry.get("questionId", 0))
        image_path = _resolve_dataset_image_path(image_ref, image_dir=image_dir)
        ocr_path = _resolve_dataset_ocr_path(image_ref, ocr_dir=ocr_dir)

        try:
            store_dir = ensure_offline_store_qwen(
                image_path=image_path,
                ocr_path=ocr_path,
                output_root=store_root,
                detect_layout=args.detect_layout,
                layout_threshold=args.layout_threshold,
                layout_labels=tuple(label.strip() for label in args.layout_labels.split(",") if label.strip()),
                local_model_path=args.local_model_path,
                qwen_model=args.qwen_model,
                qwen_device=args.qwen_device,
                qwen_dtype=args.qwen_dtype,
                qwen_max_new_tokens=args.qwen_max_new_tokens,
                qwen_temperature=args.qwen_temperature,
                qwen_crops_dir=args.qwen_crops_dir,
                qwen_batch_size=args.qwen_batch_size,
                embed_model=args.embed_model,
                embed_device=args.embed_device,
                embed_dtype=args.embed_dtype,
                embed_batch_size=args.embed_batch_size,
                embed_max_length=args.embed_max_length,
                embed_node_types=tuple(label.strip() for label in args.embed_node_types.split(",") if label.strip()),
                document_instruction=args.document_instruction,
                lambda_hub=args.lambda_hub,
                force_reindex=args.force_reindex,
            )
            report.append(
                {
                    "questionId": question_id,
                    "image": image_ref,
                    "store_dir": str(store_dir),
                    "status": "ok",
                }
            )
            progress.set_postfix_str(Path(image_ref).stem)
        except Exception as exc:
            report.append(
                {
                    "questionId": question_id,
                    "image": image_ref,
                    "store_dir": "",
                    "status": f"error: {exc}",
                }
            )
            progress.write(f"[ERROR] image={image_ref}: {exc}")
            progress.set_postfix_str(f"error {Path(image_ref).stem}")

        save_json(report, report_output)

    print(report_output)


if __name__ == "__main__":
    main()
