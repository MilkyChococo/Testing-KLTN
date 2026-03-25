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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the offline indexing workflow with Gemini region analysis: OCR graph -> Gemini regions -> node embeddings -> CSE-enriched graph."
    )
    parser.add_argument("image", type=Path, help="Path to the input image.")
    parser.add_argument("--ocr", type=Path, default=None, help="Optional OCR JSON path. Defaults to spdocvqa_ocr/<image_stem>.json.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory. Defaults to artifacts/node_stores/<image_stem>_gemini/.")
    parser.add_argument("--page", type=int, default=1, help="OCR page number to load.")
    parser.add_argument("--layout-json", type=Path, default=None, help="Optional layout detection JSON.")
    parser.add_argument("--detect-layout", action="store_true", help="Run DocLayout-YOLO instead of loading layout JSON.")
    parser.add_argument("--layout-threshold", type=float, default=0.6, help="Detection score threshold for DocLayout-YOLO.")
    parser.add_argument("--layout-labels", default="table,figure,chart,image", help="Comma-separated labels to keep when running layout detection.")
    parser.add_argument("--local-model-path", type=Path, default=None, help="Optional local DocLayout-YOLO model path.")
    parser.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL, help="Gemini model name used for region analysis.")
    parser.add_argument("--gemini-api-key", default=None, help="Optional Gemini API key. Falls back to GOOGLE_API_KEY or GEMINI_API_KEY.")
    parser.add_argument("--gemini-max-output-tokens", type=int, default=1024, help="Max output tokens for Gemini region analysis.")
    parser.add_argument("--gemini-temperature", type=float, default=0.1, help="Temperature for Gemini region analysis.")
    parser.add_argument("--gemini-crops-dir", type=Path, default=None, help="Optional directory to save region crops sent to Gemini.")
    parser.add_argument("--embed-model", default=DEFAULT_QWEN_EMBED_MODEL, help="Embedding model used to encode graph node contexts.")
    parser.add_argument("--embed-device", default="auto", help="Embedding device: auto, cuda, or cpu.")
    parser.add_argument("--embed-dtype", default="auto", help="Embedding dtype: auto, float16, bfloat16, or float32.")
    parser.add_argument("--embed-batch-size", type=int, default=4, help="Batch size for node embedding.")
    parser.add_argument("--embed-max-length", type=int, default=8192, help="Maximum token length for node embedding.")
    parser.add_argument("--embed-node-types", default="line,chunk,region,fine", help="Comma-separated node types to embed.")
    parser.add_argument(
        "--document-instruction",
        default="Represent this document graph node for retrieval in document question answering.",
        help="Instruction prepended to node context before embedding.",
    )
    parser.add_argument("--lambda-hub", type=float, default=0.1, help="Hub-penalty lambda stored in the enriched graph.")
    return parser.parse_args()


def resolve_image_path(image_path: Path) -> Path:
    try:
        return resolve_existing_file(image_path, base_dir=PROJECT_ROOT)
    except FileNotFoundError:
        return resolve_existing_file(DEFAULT_IMAGE_DIR / image_path.name)


def resolve_ocr_path(image_path: Path, ocr_path: Path | None) -> Path:
    if ocr_path is not None:
        return resolve_existing_file(ocr_path, base_dir=PROJECT_ROOT)
    inferred = DEFAULT_OCR_DIR / f"{image_path.stem}.json"
    return resolve_existing_file(inferred)


def build_node_to_row(records: list) -> dict[str, int]:
    return {record.node_id: record.row for record in records}


def resolve_optional_output_dir(path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def main() -> None:
    args = parse_args()
    image_path = resolve_image_path(args.image)
    ocr_path = resolve_ocr_path(image_path, args.ocr)
    layout_path = resolve_existing_path(args.layout_json, base_dir=PROJECT_ROOT) if args.layout_json else None
    local_model_path = resolve_existing_path(args.local_model_path, base_dir=PROJECT_ROOT) if args.local_model_path else None
    gemini_crops_dir = resolve_optional_output_dir(args.gemini_crops_dir)
    layout_labels = [label.strip() for label in args.layout_labels.split(",") if label.strip()]

    pipeline = run_document_pipeline(
        image_path=image_path,
        ocr_path=ocr_path,
        page_number=args.page,
        layout_path=layout_path,
        detect_layout=args.detect_layout,
        layout_labels=layout_labels,
        layout_threshold=args.layout_threshold,
        local_model_path=local_model_path,
        include_visual_fine_nodes=False,
        analyze_regions_with_qwen=False,
    )

    if pipeline.detections:
        annotate_detections_with_gemini(
            image_path=image_path,
            detections=pipeline.detections,
            labels=tuple(layout_labels),
            model=args.gemini_model,
            api_key=args.gemini_api_key,
            temperature=args.gemini_temperature,
            max_output_tokens=args.gemini_max_output_tokens,
            save_crops_dir=gemini_crops_dir,
            merge_into_content=True,
        )
        pipeline.original_regions = build_layout_regions(
            detections=pipeline.detections,
            page_number=args.page,
        )
        pipeline.regions = expand_regions_with_lines(
            regions=pipeline.original_regions,
            lines=pipeline.lines,
        )

    graph_payload = build_graph_payload(pipeline)
    target_node_types = tuple(label.strip().lower() for label in args.embed_node_types.split(",") if label.strip())
    records = build_node_embedding_records(
        graph_payload=graph_payload,
        target_node_types=target_node_types,
    )

    embed_config = Qwen3VLEmbeddingConfig(
        model=args.embed_model,
        device=args.embed_device,
        dtype=args.embed_dtype,
        batch_size=args.embed_batch_size,
        max_length=args.embed_max_length,
        document_instruction=args.document_instruction,
        target_node_types=target_node_types,
    )

    if args.output_dir is not None:
        output_dir = args.output_dir
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
    else:
        output_dir = embed_config.output_root / f"{image_path.stem}_gemini"
    output_dir.mkdir(parents=True, exist_ok=True)

    initial_graph_path = save_graph_payload(graph_payload, output_dir / "graph.json")
    initial_meta_path = output_dir / "embedding_meta.json"
    save_json([record.to_payload() for record in records], initial_meta_path)
    initial_enriched_graph = enrich_graph_for_cse(
        graph=graph_payload,
        embeddings=np.zeros((0, 0), dtype=np.float32),
        node_to_row=build_node_to_row(records),
        lambda_hub=args.lambda_hub,
    )
    initial_enriched_path = save_enriched_graph(
        initial_enriched_graph,
        output_dir / "graph_enriched.json",
    )
    print(f"Output directory: {output_dir}")
    print(f"Initial graph JSON saved to: {initial_graph_path}")
    print(f"Initial embedding metadata saved to: {initial_meta_path}")
    print(f"Initial enriched graph saved to: {initial_enriched_path}")
    print(f"Detections: {len(pipeline.detections)}")
    print(f"Original regions: {len(pipeline.original_regions)}")
    print(f"Expanded regions: {len(pipeline.regions)}")

    embedder = Qwen3VLNodeEmbedder(config=embed_config)
    embeddings = embedder.embed_texts(
        texts=[record.context_text for record in records],
        instruction=embed_config.document_instruction,
    )

    saved_paths = save_embedding_store(
        output_dir=output_dir,
        embeddings=embeddings,
        records=records,
        graph_payload=graph_payload,
    )

    enriched_graph = enrich_graph_for_cse(
        graph=graph_payload,
        embeddings=embeddings,
        node_to_row=build_node_to_row(records),
        lambda_hub=args.lambda_hub,
    )
    enriched_path = save_enriched_graph(enriched_graph, output_dir / "graph_enriched.json")

    print(f"Words: {len(pipeline.words)}")
    print(f"Lines: {len(pipeline.lines)}")
    print(f"Chunks: {len(pipeline.chunks)}")
    print(f"Regions: {len(pipeline.regions)}")
    print(f"Fine nodes: {len(pipeline.fine_nodes)}")
    print(f"Embedded nodes: {len(records)}")
    print(f"Embedding shape: {tuple(embeddings.shape)}")
    print(f"Saved graph JSON to: {saved_paths['graph_path']}")
    print(f"Saved embeddings to: {saved_paths['embedding_path']}")
    print(f"Saved embedding metadata to: {saved_paths['meta_path']}")
    print(f"Saved enriched graph to: {enriched_path}")


if __name__ == "__main__":
    main()

