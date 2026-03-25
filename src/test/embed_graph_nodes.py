from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database import (
    Qwen3VLEmbeddingConfig,
    Qwen3VLNodeEmbedder,
    build_node_embedding_records,
    load_graph_payload,
    save_embedding_store,
)
from src.utils.config import DEFAULT_QWEN_EMBED_MODEL
from src.utils.io import resolve_existing_file, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read an existing graph.json and save node embeddings as .npy."
    )
    parser.add_argument(
        "graph_json",
        type=Path,
        help="Path to an existing graph.json exported from the graph pipeline.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to <graph_json_stem>_store beside the graph file.",
    )
    parser.add_argument(
        "--embed-model",
        default=DEFAULT_QWEN_EMBED_MODEL,
        help="Embedding model used to encode graph node contexts.",
    )
    parser.add_argument("--embed-device", default="auto", help="Embedding device: auto, cuda, or cpu.")
    parser.add_argument("--embed-dtype", default="auto", help="Embedding dtype: auto, float16, bfloat16, or float32.")
    parser.add_argument("--embed-batch-size", type=int, default=8, help="Batch size for node embedding.")
    parser.add_argument("--embed-max-length", type=int, default=8192, help="Maximum token length for node embedding.")
    parser.add_argument(
        "--embed-node-types",
        default="line,chunk,region,fine",
        help="Comma-separated node types to embed.",
    )
    parser.add_argument(
        "--document-instruction",
        default="Represent this document graph node for retrieval in document question answering.",
        help="Instruction prepended to node context before embedding.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph_path = resolve_existing_file(args.graph_json, base_dir=PROJECT_ROOT)
    payload = load_graph_payload(graph_path)
    target_node_types = tuple(
        label.strip().lower() for label in args.embed_node_types.split(",") if label.strip()
    )
    records = build_node_embedding_records(
        graph_payload=payload,
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
    embedder = Qwen3VLNodeEmbedder(config=embed_config)
    embeddings = embedder.embed_texts(
        texts=[record.context_text for record in records],
        instruction=embed_config.document_instruction,
    )

    if args.output_dir is not None:
        output_dir = args.output_dir
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
    else:
        output_dir = graph_path.parent / f"{graph_path.stem}_store"

    saved = save_embedding_store(
        output_dir=output_dir,
        embeddings=embeddings,
        records=records,
        graph_payload=None,
    )

    output_graph_copy = output_dir / "graph.json"
    save_json(payload, output_graph_copy)

    print(f"Embedded nodes: {len(records)}")
    print(f"Embedding shape: {tuple(embeddings.shape)}")
    print(f"Source graph JSON: {graph_path}")
    print(f"Copied graph JSON to: {output_graph_copy}")
    print(f"Saved embeddings to: {saved['embedding_path']}")
    print(f"Saved embedding metadata to: {saved['meta_path']}")


if __name__ == "__main__":
    main()

