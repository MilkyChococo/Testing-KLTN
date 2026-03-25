from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algo.cse_query import run_multi_subgraph_cse_from_store, save_cse_subgraph
from src.api import DEFAULT_GEMINI_MODEL, answer_subgraph_with_gemini
from src.utils.config import DEFAULT_QWEN_EMBED_MODEL
from src.utils.fallback import backfill_document_from_sibling_graph
from src.utils.io import resolve_existing_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run top-k seed retrieval -> per-seed CSE -> top-k subgraph QA with Gemini."
    )
    parser.add_argument("store_dir", type=Path, help="Directory containing graph_enriched.json, embeddings.npy, and embedding_meta.json.")
    parser.add_argument("query", help="User query used to retrieve seed nodes, expand subgraphs, and answer.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of seed nodes and reranked subgraphs to keep.")
    parser.add_argument("--hops", type=int, default=5, help="Number of expansion hops per seed subgraph.")
    parser.add_argument("--top-m", type=int, default=5, help="Per-frontier top-m neighbors kept after scoring.")
    parser.add_argument("--threshold", type=float, default=0.2, help="Minimum edge score to keep during expansion.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for offline edge confidence.")
    parser.add_argument("--lambda-hub", type=float, default=0.05, help="Hub penalty lambda.")
    parser.add_argument("--max-nodes", type=int, default=100, help="Maximum number of nodes in each expanded subgraph.")
    parser.add_argument("--max-edges", type=int, default=200, help="Maximum number of edges in each expanded subgraph.")
    parser.add_argument("--seed-node-types", default="line,chunk,region,fine", help="Comma-separated node types allowed as initial seeds.")
    parser.add_argument("--allowed-relations", default="", help="Optional comma-separated whitelist of edge relations for expansion.")
    parser.add_argument("--embed-model", default=DEFAULT_QWEN_EMBED_MODEL, help="Embedding model used for the query.")
    parser.add_argument("--embed-device", default="auto", help="Embedding device: auto, cuda, or cpu.")
    parser.add_argument("--embed-dtype", default="auto", help="Embedding dtype: auto, float16, bfloat16, or float32.")
    parser.add_argument("--embed-max-length", type=int, default=8192, help="Maximum token length for query embedding.")
    parser.add_argument("--embed-batch-size", type=int, default=8, help="Batch size setting passed to the embedder.")
    parser.add_argument("--answer-model", default=DEFAULT_GEMINI_MODEL, help="Gemini model used for final answering.")
    parser.add_argument("--gemini-api-key", default=None, help="Optional Gemini API key. Falls back to GOOGLE_API_KEY or GEMINI_API_KEY.")
    parser.add_argument("--answer-max-output-tokens", type=int, default=512, help="Max output tokens for the final answer.")
    parser.add_argument("--answer-temperature", type=float, default=0.1, help="Generation temperature.")
    parser.add_argument("--max-context-nodes", type=int, default=20, help="Maximum number of nodes taken from each expanded subgraph for the text prompt.")
    parser.add_argument("--max-images", type=int, default=4, help="Maximum number of region crops passed into Gemini for each expanded subgraph.")
    parser.add_argument("--subgraph-output", type=Path, default=None, help="Optional path to save the reranked top-k subgraphs JSON payload.")
    parser.add_argument("--answer-output", type=Path, default=None, help="Optional path to save full answer JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store_dir = resolve_existing_dir(args.store_dir, base_dir=PROJECT_ROOT)
    seed_node_types = tuple(item.strip().lower() for item in args.seed_node_types.split(",") if item.strip())
    allowed_relations = tuple(item.strip() for item in args.allowed_relations.split(",") if item.strip()) or None

    subgraph_payload = run_multi_subgraph_cse_from_store(
        store_dir=store_dir,
        query=args.query,
        top_k=args.top_k,
        hops=args.hops,
        top_m=args.top_m,
        threshold=args.threshold,
        alpha=args.alpha,
        lambda_hub=args.lambda_hub,
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
        allowed_seed_node_types=seed_node_types,
        allowed_relations=allowed_relations,
        embed_model=args.embed_model,
        embed_device=args.embed_device,
        embed_dtype=args.embed_dtype,
        embed_max_length=args.embed_max_length,
        embed_batch_size=args.embed_batch_size,
    )
    subgraph_payload = backfill_document_from_sibling_graph(
        subgraph_payload,
        store_dir / "graph_enriched.json",
    )

    if args.subgraph_output is not None:
        subgraph_output = args.subgraph_output
        if not subgraph_output.is_absolute():
            subgraph_output = PROJECT_ROOT / subgraph_output
        save_cse_subgraph(subgraph_payload, subgraph_output)

    answer = answer_subgraph_with_gemini(
        subgraph_payload=subgraph_payload,
        model=args.answer_model,
        api_key=args.gemini_api_key,
        temperature=args.answer_temperature,
        max_output_tokens=args.answer_max_output_tokens,
        max_context_nodes=args.max_context_nodes,
        max_images=args.max_images,
    )

    if args.answer_output is not None:
        answer_output = args.answer_output
        if not answer_output.is_absolute():
            answer_output = PROJECT_ROOT / answer_output
        save_json(answer.to_payload(), answer_output)

    print(answer.answer)


if __name__ == "__main__":
    main()

