from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.index_offline_gemini import (
    DEFAULT_IMAGE_DIR,
    DEFAULT_OCR_DIR,
    DEFAULT_OUTPUT_ROOT,
    ensure_offline_store_gemini,
)
from src.algo.cse_query import run_multi_subgraph_cse_from_store
from src.api import DEFAULT_GEMINI_MODEL, answer_subgraph_with_gemini
from src.utils.config import DEFAULT_QWEN_EMBED_MODEL
from src.utils.fallback import backfill_document_from_sibling_graph
from src.utils.io import load_json, save_json


DEFAULT_TEST_JSON = PROJECT_ROOT / "dataset" / "spdocvqa" / "spdocvqa_qas" / "test_v1.0.json"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "artifacts" / "inference" / "spdocvqa_test_answers_gemini.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Gemini offline indexing + CSE QA sequentially on an SP-DocVQA test JSON file."
    )
    parser.add_argument(
        "test_json",
        type=Path,
        nargs="?",
        default=DEFAULT_TEST_JSON,
        help="Path to the SP-DocVQA test JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Output JSON file updated after each answered question.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help="Directory containing the source page images.",
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
        help="Root folder where per-image Gemini stores are created.",
    )
    parser.add_argument("--force-reindex", action="store_true", help="Rebuild existing offline stores.")
    parser.add_argument("--overwrite-existing", action="store_true", help="Re-answer questionIds already present in the output JSON.")
    parser.add_argument("--detect-layout", dest="detect_layout", action="store_true", help="Run DocLayout-YOLO when building offline stores.")
    parser.add_argument("--no-detect-layout", dest="detect_layout", action="store_false", help="Disable DocLayout-YOLO when building offline stores.")
    parser.add_argument("--layout-threshold", type=float, default=0.6, help="Layout detection threshold.")
    parser.add_argument("--layout-labels", default="table,figure,chart,image", help="Comma-separated layout labels to keep.")
    parser.add_argument("--local-model-path", type=Path, default=None, help="Optional local DocLayout-YOLO checkpoint.")
    parser.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL, help="Gemini model for both region analysis and final QA.")
    parser.add_argument("--gemini-api-key", default=None, help="Optional Gemini API key.")
    parser.add_argument("--gemini-max-output-tokens", type=int, default=1024, help="Max output tokens for region analysis.")
    parser.add_argument("--gemini-temperature", type=float, default=0.1, help="Temperature for region analysis.")
    parser.add_argument("--gemini-crops-dir", type=Path, default=None, help="Optional directory to save Gemini region crops.")
    parser.add_argument("--embed-model", default=DEFAULT_QWEN_EMBED_MODEL, help="Embedding model for nodes and queries.")
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
    parser.add_argument("--top-k", type=int, default=10, help="Number of seed nodes and reranked subgraphs to keep.")
    parser.add_argument("--hops", type=int, default=5, help="Number of expansion hops per seed.")
    parser.add_argument("--top-m", type=int, default=5, help="Per-frontier top-m neighbors.")
    parser.add_argument("--threshold", type=float, default=0.2, help="Minimum CSE edge score.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for offline edge confidence.")
    parser.add_argument("--max-nodes", type=int, default=100, help="Maximum nodes in each subgraph.")
    parser.add_argument("--max-edges", type=int, default=200, help="Maximum edges in each subgraph.")
    parser.add_argument("--seed-node-types", default="line,chunk,region,fine", help="Comma-separated seed node types.")
    parser.add_argument("--allowed-relations", default="", help="Optional comma-separated relation whitelist.")
    parser.add_argument("--answer-max-output-tokens", type=int, default=512, help="Max output tokens for final Gemini answer.")
    parser.add_argument("--answer-temperature", type=float, default=0.1, help="Temperature for final Gemini answer.")
    parser.add_argument("--max-context-nodes", type=int, default=20, help="Maximum nodes per subgraph sent as text context.")
    parser.add_argument("--max-images", type=int, default=4, help="Maximum region crops per subgraph.")
    parser.add_argument("--fail-answer", default="NOT_FOUND", help="Answer string written when one sample fails.")
    parser.set_defaults(detect_layout=True)
    return parser.parse_args()


def _resolve_dataset_image_path(image_ref: str, image_dir: Path) -> Path:
    return image_dir / Path(image_ref).name


def _resolve_dataset_ocr_path(image_ref: str, ocr_dir: Path) -> Path:
    return ocr_dir / f"{Path(image_ref).stem}.json"


def _load_existing_answers(output_path: Path) -> dict[int, str]:
    if not output_path.is_file():
        return {}
    payload = load_json(output_path)
    if not isinstance(payload, list):
        return {}
    answers: dict[int, str] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        question_id = item.get("questionId")
        answer = item.get("answer")
        if isinstance(question_id, int) and isinstance(answer, str):
            answers[question_id] = answer
    return answers


def _build_output_payload(entries: list[dict[str, Any]], answers_by_id: dict[int, str]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for entry in entries:
        question_id = int(entry["questionId"])
        if question_id not in answers_by_id:
            continue
        payload.append(
            {
                "questionId": question_id,
                "answer": answers_by_id[question_id],
            }
        )
    return payload


def _answer_one_question(
    store_dir: Path,
    query: str,
    args: argparse.Namespace,
) -> str:
    seed_node_types = tuple(item.strip().lower() for item in args.seed_node_types.split(",") if item.strip())
    allowed_relations = tuple(item.strip() for item in args.allowed_relations.split(",") if item.strip()) or None

    subgraph_payload = run_multi_subgraph_cse_from_store(
        store_dir=store_dir,
        query=query,
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
    answer = answer_subgraph_with_gemini(
        subgraph_payload=subgraph_payload,
        model=args.gemini_model,
        api_key=args.gemini_api_key,
        temperature=args.answer_temperature,
        max_output_tokens=args.answer_max_output_tokens,
        max_context_nodes=args.max_context_nodes,
        max_images=args.max_images,
    )
    return answer.answer.strip()


def main() -> None:
    args = parse_args()
    test_json_path = args.test_json if args.test_json.is_absolute() else (PROJECT_ROOT / args.test_json)
    output_path = args.output if args.output.is_absolute() else (PROJECT_ROOT / args.output)
    image_dir = args.image_dir if args.image_dir.is_absolute() else (PROJECT_ROOT / args.image_dir)
    ocr_dir = args.ocr_dir if args.ocr_dir.is_absolute() else (PROJECT_ROOT / args.ocr_dir)
    store_root = args.store_root if args.store_root.is_absolute() else (PROJECT_ROOT / args.store_root)

    test_payload = load_json(test_json_path)
    entries = list(test_payload.get("data", []))
    answers_by_id = _load_existing_answers(output_path)

    progress = tqdm(entries, desc="Gemini inference", unit="question")
    for entry in progress:
        question_id = int(entry["questionId"])
        question = str(entry["question"]).strip()

        if question_id in answers_by_id and not args.overwrite_existing:
            progress.set_postfix_str(f"skip qid={question_id}")
            continue

        image_path = _resolve_dataset_image_path(str(entry["image"]), image_dir=image_dir)
        ocr_path = _resolve_dataset_ocr_path(str(entry["image"]), ocr_dir=ocr_dir)

        try:
            store_dir = ensure_offline_store_gemini(
                image_path=image_path,
                ocr_path=ocr_path,
                output_root=store_root,
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
            answer = _answer_one_question(
                store_dir=store_dir,
                query=question,
                args=args,
            )
            answers_by_id[question_id] = answer or args.fail_answer
            progress.set_postfix_str(f"qid={question_id}")
        except Exception as exc:
            answers_by_id[question_id] = args.fail_answer
            progress.write(f"[ERROR] questionId={question_id}: {exc}")
            progress.set_postfix_str(f"error qid={question_id}")

        save_json(_build_output_payload(entries, answers_by_id), output_path)

    print(output_path)


if __name__ == "__main__":
    main()

