from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algo import enrich_graph_for_cse, save_enriched_graph
from src.api import DEFAULT_GEMINI_MODEL, annotate_detections_with_gemini
from src.database import (
    Qwen3VLEmbeddingConfig,
    Qwen3VLNodeEmbedder,
    build_graph_payload,
    build_node_embedding_records,
    save_embedding_store,
    save_graph_payload,
)
from src.extract.document_pipeline import expand_regions_with_lines, run_document_pipeline
from src.extract.region_to_chunk import build_layout_regions
from src.utils.config import DEFAULT_QWEN_EMBED_MODEL
from src.utils.io import resolve_existing_file, resolve_existing_path, save_json


DEFAULT_OCR_DIR = PROJECT_ROOT / "dataset" / "spdocvqa" / "spdocvqa_ocr"
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "dataset" / "spdocvqa" / "spdocvqa_images"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "node_stores_inference_gemini"


def resolve_image_path(image_path: str | Path, image_dir: str | Path = DEFAULT_IMAGE_DIR) -> Path:
    candidate = Path(image_path)
    if candidate.is_absolute():
        return resolve_existing_file(candidate)
    try:
        return resolve_existing_file(candidate, base_dir=PROJECT_ROOT)
    except FileNotFoundError:
        return resolve_existing_file(Path(image_dir) / candidate.name)


def resolve_ocr_path(
    image_path: Path,
    ocr_path: str | Path | None = None,
    ocr_dir: str | Path = DEFAULT_OCR_DIR,
) -> Path:
    if ocr_path is not None:
        return resolve_existing_file(ocr_path, base_dir=PROJECT_ROOT)
    inferred = Path(ocr_dir) / f"{image_path.stem}.json"
    return resolve_existing_file(inferred)


def resolve_optional_output_dir(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def build_node_to_row(records: list) -> dict[str, int]:
    return {record.node_id: record.row for record in records}


def get_store_dir(
    image_path: Path,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    return Path(output_root) / f"{image_path.stem}_gemini"


def has_complete_store(store_dir: str | Path) -> bool:
    store_dir = Path(store_dir)
    required = (
        store_dir / "graph.json",
        store_dir / "embeddings.npy",
        store_dir / "embedding_meta.json",
        store_dir / "graph_enriched.json",
    )
    return all(path.is_file() for path in required)


def ensure_offline_store_gemini(
    image_path: str | Path,
    ocr_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    page: int = 1,
    layout_json: str | Path | None = None,
    detect_layout: bool = True,
    layout_threshold: float = 0.6,
    layout_labels: tuple[str, ...] = ("table", "figure", "chart", "image"),
    local_model_path: str | Path | None = None,
    gemini_model: str = DEFAULT_GEMINI_MODEL,
    gemini_api_key: str | None = None,
    gemini_max_output_tokens: int = 1024,
    gemini_temperature: float = 0.1,
    gemini_crops_dir: str | Path | None = None,
    embed_model: str = DEFAULT_QWEN_EMBED_MODEL,
    embed_device: str = "auto",
    embed_dtype: str = "auto",
    embed_batch_size: int = 4,
    embed_max_length: int = 8192,
    embed_node_types: tuple[str, ...] = ("line", "chunk", "region", "fine"),
    document_instruction: str = "Represent this document graph node for retrieval in document question answering.",
    lambda_hub: float = 0.1,
    force_reindex: bool = False,
) -> Path:
    resolved_image_path = resolve_image_path(image_path)
    resolved_ocr_path = resolve_ocr_path(resolved_image_path, ocr_path=ocr_path)
    resolved_layout_path = (
        resolve_existing_path(layout_json, base_dir=PROJECT_ROOT)
        if layout_json is not None
        else None
    )
    resolved_local_model_path = (
        resolve_existing_path(local_model_path, base_dir=PROJECT_ROOT)
        if local_model_path is not None
        else None
    )
    resolved_gemini_crops_dir = resolve_optional_output_dir(gemini_crops_dir)

    if output_dir is not None:
        store_dir = Path(output_dir)
        if not store_dir.is_absolute():
            store_dir = PROJECT_ROOT / store_dir
    else:
        store_dir = get_store_dir(resolved_image_path, output_root=output_root)

    if has_complete_store(store_dir) and not force_reindex:
        return store_dir

    pipeline = run_document_pipeline(
        image_path=resolved_image_path,
        ocr_path=resolved_ocr_path,
        page_number=page,
        layout_path=resolved_layout_path,
        detect_layout=detect_layout,
        layout_labels=list(layout_labels),
        layout_threshold=layout_threshold,
        local_model_path=resolved_local_model_path,
        include_visual_fine_nodes=False,
        analyze_regions_with_qwen=False,
    )

    if pipeline.detections:
        annotate_detections_with_gemini(
            image_path=resolved_image_path,
            detections=pipeline.detections,
            labels=tuple(layout_labels),
            model=gemini_model,
            api_key=gemini_api_key,
            temperature=gemini_temperature,
            max_output_tokens=gemini_max_output_tokens,
            save_crops_dir=resolved_gemini_crops_dir,
            merge_into_content=True,
        )
        pipeline.original_regions = build_layout_regions(
            detections=pipeline.detections,
            page_number=page,
        )
        pipeline.regions = expand_regions_with_lines(
            regions=pipeline.original_regions,
            lines=pipeline.lines,
        )

    graph_payload = build_graph_payload(pipeline)
    records = build_node_embedding_records(
        graph_payload=graph_payload,
        target_node_types=tuple(item.strip().lower() for item in embed_node_types if item.strip()),
    )

    embed_config = Qwen3VLEmbeddingConfig(
        model=embed_model,
        device=embed_device,
        dtype=embed_dtype,
        batch_size=embed_batch_size,
        max_length=embed_max_length,
        document_instruction=document_instruction,
        target_node_types=tuple(item.strip().lower() for item in embed_node_types if item.strip()),
    )

    store_dir.mkdir(parents=True, exist_ok=True)
    save_graph_payload(graph_payload, store_dir / "graph.json")
    save_json([record.to_payload() for record in records], store_dir / "embedding_meta.json")
    initial_enriched_graph = enrich_graph_for_cse(
        graph=graph_payload,
        embeddings=np.zeros((0, 0), dtype=np.float32),
        node_to_row=build_node_to_row(records),
        lambda_hub=lambda_hub,
    )
    save_enriched_graph(initial_enriched_graph, store_dir / "graph_enriched.json")

    embedder = Qwen3VLNodeEmbedder(config=embed_config)
    embeddings = embedder.embed_texts(
        texts=[record.context_text for record in records],
        instruction=embed_config.document_instruction,
    )

    save_embedding_store(
        output_dir=store_dir,
        embeddings=embeddings,
        records=records,
        graph_payload=graph_payload,
    )

    enriched_graph = enrich_graph_for_cse(
        graph=graph_payload,
        embeddings=embeddings,
        node_to_row=build_node_to_row(records),
        lambda_hub=lambda_hub,
    )
    save_enriched_graph(enriched_graph, store_dir / "graph_enriched.json")
    return store_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or refresh one Gemini offline store for a document image."
    )
    parser.add_argument("image", type=Path, help="Input image path.")
    parser.add_argument("--ocr", type=Path, default=None, help="Optional OCR JSON path.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional exact output directory.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Root directory for generated stores.")
    parser.add_argument("--page", type=int, default=1, help="OCR page number.")
    parser.add_argument("--layout-json", type=Path, default=None, help="Optional layout JSON path.")
    parser.add_argument("--detect-layout", dest="detect_layout", action="store_true", help="Run DocLayout-YOLO.")
    parser.add_argument("--no-detect-layout", dest="detect_layout", action="store_false", help="Disable DocLayout-YOLO.")
    parser.add_argument("--layout-threshold", type=float, default=0.6, help="Layout detection threshold.")
    parser.add_argument("--layout-labels", default="table,figure,chart,image", help="Comma-separated layout labels to keep.")
    parser.add_argument("--local-model-path", type=Path, default=None, help="Optional local DocLayout-YOLO checkpoint.")
    parser.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL, help="Gemini model for region analysis.")
    parser.add_argument("--gemini-api-key", default=None, help="Optional Gemini API key.")
    parser.add_argument("--gemini-max-output-tokens", type=int, default=1024, help="Max region-analysis output tokens.")
    parser.add_argument("--gemini-temperature", type=float, default=0.1, help="Gemini temperature for region analysis.")
    parser.add_argument("--gemini-crops-dir", type=Path, default=None, help="Optional directory to save Gemini crops.")
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
    parser.add_argument("--lambda-hub", type=float, default=0.1, help="Hub penalty stored in graph_enriched.json.")
    parser.add_argument("--force-reindex", action="store_true", help="Rebuild even if a complete store already exists.")
    parser.set_defaults(detect_layout=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store_dir = ensure_offline_store_gemini(
        image_path=args.image,
        ocr_path=args.ocr,
        output_dir=args.output_dir,
        output_root=args.output_root,
        page=args.page,
        layout_json=args.layout_json,
        detect_layout=args.detect_layout,
        layout_threshold=args.layout_threshold,
        layout_labels=tuple(label.strip() for label in args.layout_labels.split(",") if label.strip()),
        local_model_path=args.local_model_path,
        gemini_model=args.gemini_model,
        gemini_api_key=args.gemini_api_key,
        gemini_max_output_tokens=args.gemini_max_output_tokens,
        gemini_temperature=args.gemini_temperature,
        gemini_crops_dir=args.gemini_crops_dir,
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
    print(store_dir)


if __name__ == "__main__":
    main()

