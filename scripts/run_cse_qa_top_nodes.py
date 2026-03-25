from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algo.cse_query import run_multi_subgraph_cse_from_store, save_cse_subgraph
from src.api.qwen_vl_answering import answer_subgraph_with_qwen
from src.api.qwen_vl_region_analysis import DEFAULT_QWEN_MODEL
from src.utils.config import DEFAULT_QWEN_EMBED_MODEL
from src.utils.fallback import backfill_document_from_sibling_graph
from src.utils.io import resolve_existing_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test variant: expand top-k seed subgraphs, then keep top-n nodes per expanded subgraph before Qwen2.5-VL answering."
    )
    parser.add_argument("store_dir", type=Path, help="Directory containing graph_enriched.json, embeddings.npy, and embedding_meta.json.")
    parser.add_argument("query", help="User query.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of seed nodes / expanded subgraphs.")
    parser.add_argument("--hops", type=int, default=5, help="Number of expansion hops per seed subgraph.")
    parser.add_argument("--top-m", type=int, default=5, help="Per-frontier top-m neighbors kept after scoring.")
    parser.add_argument("--threshold", type=float, default=0.3, help="Minimum edge score to keep during expansion.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for offline edge confidence.")
    parser.add_argument("--lambda-hub", type=float, default=0.05, help="Hub penalty lambda.")
    parser.add_argument("--max-nodes", type=int, default=100, help="Maximum number of nodes in each expanded subgraph.")
    parser.add_argument("--max-edges", type=int, default=200, help="Maximum number of edges in each expanded subgraph.")
    parser.add_argument("--top-nodes-per-subgraph", type=int, default=5, help="Number of highest-scoring nodes kept in each expanded subgraph.")
    parser.add_argument(
        "--seed-node-types",
        default="line,chunk,region,fine",
        help="Comma-separated node types allowed as initial seeds.",
    )
    parser.add_argument(
        "--allowed-relations",
        default="",
        help="Optional comma-separated whitelist of edge relations for expansion.",
    )
    parser.add_argument("--embed-model", default=DEFAULT_QWEN_EMBED_MODEL, help="Embedding model used for the query.")
    parser.add_argument("--embed-device", default="auto", help="Embedding device: auto, cuda, or cpu.")
    parser.add_argument("--embed-dtype", default="auto", help="Embedding dtype: auto, float16, bfloat16, or float32.")
    parser.add_argument("--embed-max-length", type=int, default=8192, help="Maximum token length for query embedding.")
    parser.add_argument("--embed-batch-size", type=int, default=8, help="Batch size setting passed to the embedder.")
    parser.add_argument("--answer-model", default=DEFAULT_QWEN_MODEL, help="Qwen2.5-VL model used for final answering.")
    parser.add_argument("--answer-device", default="auto", help="Qwen answer device: auto, cuda, or cpu.")
    parser.add_argument("--answer-dtype", default="auto", help="Qwen answer dtype: auto, float16, bfloat16, or float32.")
    parser.add_argument("--answer-max-new-tokens", type=int, default=512, help="Max generated tokens for the final answer.")
    parser.add_argument("--answer-temperature", type=float, default=0.1, help="Generation temperature.")
    parser.add_argument("--max-context-nodes", type=int, default=20, help="Maximum number of nodes taken from each pruned subgraph for the text prompt.")
    parser.add_argument("--max-images", type=int, default=4, help="Maximum number of region crops passed into Qwen for each pruned subgraph.")
    parser.add_argument("--subgraph-output", type=Path, default=None, help="Optional path to save the pruned subgraph payload.")
    parser.add_argument("--answer-output", type=Path, default=None, help="Optional path to save full answer JSON.")
    return parser.parse_args()


def _compute_node_final_scores(subgraph: dict[str, Any]) -> dict[str, float]:
    node_scores: dict[str, float] = {}
    seed_node = dict(subgraph.get("seed_node", {}))
    seed_node_id = str(seed_node.get("node_id", "")).strip()
    seed_rel = float(seed_node.get("rel", 0.0) or 0.0)
    if seed_node_id:
        node_scores[seed_node_id] = seed_rel

    for node in subgraph.get("nodes", []):
        node_id = str(node.get("id", "")).strip()
        if not node_id:
            continue
        node_scores[node_id] = max(node_scores.get(node_id, 0.0), float(node.get("rel", 0.0) or 0.0))

    for edge in subgraph.get("edges", []):
        target_id = str(edge.get("target_id", "")).strip()
        if not target_id:
            continue
        node_scores[target_id] = max(node_scores.get(target_id, 0.0), float(edge.get("score", 0.0) or 0.0))

    return node_scores


def _prune_single_subgraph(subgraph: dict[str, Any], top_nodes: int) -> dict[str, Any]:
    nodes = [dict(node) for node in subgraph.get("nodes", [])]
    if not nodes or top_nodes <= 0:
        pruned = dict(subgraph)
        pruned["nodes"] = []
        pruned["edges"] = []
        pruned["stats"] = {
            **dict(subgraph.get("stats", {})),
            "num_selected_nodes": 0,
            "num_selected_edges": 0,
            "num_pruned_nodes": len(nodes),
        }
        return pruned

    node_scores = _compute_node_final_scores(subgraph)
    seed_node = dict(subgraph.get("seed_node", {}))
    seed_node_id = str(seed_node.get("node_id", "")).strip()

    for node in nodes:
        node_id = str(node.get("id", "")).strip()
        final_score = float(node_scores.get(node_id, float(node.get("rel", 0.0) or 0.0)))
        node["final_score"] = round(final_score, 6)
        node["original_rel"] = float(node.get("rel", 0.0) or 0.0)
        node["rel"] = round(final_score, 6)

    nodes.sort(
        key=lambda item: (
            -float(item.get("final_score", 0.0)),
            0 if str(item.get("id", "")).strip() == seed_node_id else 1,
        )
    )

    kept_nodes: list[dict[str, Any]] = []
    kept_ids: set[str] = set()
    if seed_node_id:
        for node in nodes:
            if str(node.get("id", "")).strip() == seed_node_id:
                kept_nodes.append(node)
                kept_ids.add(seed_node_id)
                break

    for node in nodes:
        node_id = str(node.get("id", "")).strip()
        if node_id in kept_ids:
            continue
        if len(kept_nodes) >= top_nodes:
            break
        kept_nodes.append(node)
        kept_ids.add(node_id)

    kept_edges = [
        dict(edge)
        for edge in subgraph.get("edges", [])
        if str(edge.get("source_id", "")).strip() in kept_ids
        and str(edge.get("target_id", "")).strip() in kept_ids
    ]

    pruned = dict(subgraph)
    pruned["nodes"] = kept_nodes
    pruned["edges"] = kept_edges
    pruned["stats"] = {
        **dict(subgraph.get("stats", {})),
        "num_selected_nodes": len(kept_nodes),
        "num_selected_edges": len(kept_edges),
        "num_pruned_nodes": max(0, len(nodes) - len(kept_nodes)),
    }
    return pruned


def prune_multi_subgraph_payload(payload: dict[str, Any], top_nodes_per_subgraph: int) -> dict[str, Any]:
    subgraphs = list(payload.get("subgraphs", []))
    pruned_subgraphs = [
        _prune_single_subgraph(subgraph, top_nodes=top_nodes_per_subgraph)
        for subgraph in subgraphs
    ]
    result = dict(payload)
    result["subgraphs"] = pruned_subgraphs
    result["stats"] = {
        **dict(payload.get("stats", {})),
        "top_nodes_per_subgraph": top_nodes_per_subgraph,
        "num_total_nodes_after_prune": sum(len(item.get("nodes", [])) for item in pruned_subgraphs),
        "num_total_edges_after_prune": sum(len(item.get("edges", [])) for item in pruned_subgraphs),
    }
    return result


def main() -> None:
    args = parse_args()
    store_dir = resolve_existing_dir(args.store_dir, base_dir=PROJECT_ROOT)
    seed_node_types = tuple(item.strip().lower() for item in args.seed_node_types.split(",") if item.strip())
    allowed_relations = tuple(item.strip() for item in args.allowed_relations.split(",") if item.strip()) or None

    expanded_payload = run_multi_subgraph_cse_from_store(
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
    expanded_payload = backfill_document_from_sibling_graph(
        expanded_payload,
        store_dir / "graph_enriched.json",
    )
    pruned_payload = prune_multi_subgraph_payload(
        expanded_payload,
        top_nodes_per_subgraph=args.top_nodes_per_subgraph,
    )

    if args.subgraph_output is not None:
        subgraph_output = args.subgraph_output
        if not subgraph_output.is_absolute():
            subgraph_output = PROJECT_ROOT / subgraph_output
        save_cse_subgraph(pruned_payload, subgraph_output)

    answer = answer_subgraph_with_qwen(
        subgraph_payload=pruned_payload,
        model=args.answer_model,
        device=args.answer_device,
        dtype=args.answer_dtype,
        max_new_tokens=args.answer_max_new_tokens,
        temperature=args.answer_temperature,
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

