from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config_model_embed import Qwen3VLEmbeddingConfig
from .graph_store import save_graph_payload


_EMBED_MODEL_CACHE: dict[tuple[str, str, str], tuple[Any, Any, str, Any]] = {}


@dataclass(slots=True)
class NodeEmbeddingRecord:
    row: int
    node_id: str
    node_type: str
    page: int | None
    context_text: str
    label: str = ""
    modality: str = ""

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


class Qwen3VLNodeEmbedder:
    def __init__(self, config: Qwen3VLEmbeddingConfig | None = None) -> None:
        self.config = config or Qwen3VLEmbeddingConfig()
        self._processor: Any | None = None
        self._model: Any | None = None
        self._device: str | None = None
        self._dtype: Any | None = None

    @staticmethod
    def _torch() -> Any:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Torch is required for Qwen3-VL node embedding. Install torch in the active environment first."
            ) from exc
        return torch

    def _resolve_device(self) -> str:
        torch = self._torch()
        if self.config.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config.device

    def _resolve_dtype(self, device: str) -> Any:
        torch = self._torch()
        if self.config.dtype == "auto":
            if device == "cuda":
                if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                    return torch.bfloat16
                return torch.float16
            return torch.float32

        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        try:
            return mapping[self.config.dtype.lower()]
        except KeyError as exc:
            raise ValueError(f"Unsupported dtype: {self.config.dtype}") from exc

    def _load(self) -> tuple[Any, Any, str]:
        if self._processor is not None and self._model is not None and self._device is not None:
            return self._processor, self._model, self._device

        torch = self._torch()
        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3-VL-Embedding requires transformers with Qwen3VL support. "
                "Use the official requirement transformers>=4.57.0."
            ) from exc

        device = self._resolve_device()
        dtype = self._resolve_dtype(device)
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "Qwen3-VL embedding was requested on CUDA, but torch.cuda.is_available() is False."
            )
        dtype_name = str(dtype).replace("torch.", "")
        cache_key = (self.config.model, device, dtype_name)
        cached = _EMBED_MODEL_CACHE.get(cache_key)
        if cached is not None:
            processor, model, cached_device, cached_dtype = cached
            print(
                f"[Embed] Reusing cached model: {self.config.model} on {cached_device} ({str(cached_dtype).replace('torch.', '')})",
                flush=True,
            )
            self._processor = processor
            self._model = model
            self._device = cached_device
            self._dtype = cached_dtype
            return processor, model, cached_device

        print(f"[Embed] Loading processor: {self.config.model}", flush=True)
        processor = AutoProcessor.from_pretrained(
            self.config.model,
            trust_remote_code=True,
        )

        print(
            f"[Embed] Loading model: {self.config.model} on {device} ({str(dtype).replace('torch.', '')})",
            flush=True,
        )
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.config.model,
            dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model = model.to(device)
        model.eval()

        self._processor = processor
        self._model = model
        self._device = device
        self._dtype = dtype
        _EMBED_MODEL_CACHE[cache_key] = (processor, model, device, dtype)
        return processor, model, device

    def _format_text(self, text: str, instruction: str | None = None) -> str:
        cleaned = text.strip() or "[EMPTY NODE CONTEXT]"
        if instruction:
            cleaned = f"Instruct: {instruction}\nInput: {cleaned}"

        processor, _, _ = self._load()
        eos_token = getattr(processor.tokenizer, "eos_token", None)
        if eos_token and not cleaned.endswith(eos_token):
            cleaned = f"{cleaned}{eos_token}"
        return cleaned

    def embed_texts(
        self,
        texts: list[str],
        instruction: str | None = None,
        batch_size: int | None = None,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        torch = self._torch()
        processor, model, device = self._load()
        batch_size = batch_size or self.config.batch_size
        formatted_texts = [self._format_text(text, instruction=instruction) for text in texts]

        batches: list[np.ndarray] = []
        for start in range(0, len(formatted_texts), batch_size):
            batch_texts = formatted_texts[start : start + batch_size]
            print(
                f"[Embed] Encoding batch {start // batch_size + 1}/{math.ceil(len(formatted_texts) / batch_size)} ",
                flush=True,
            )
            inputs = processor(
                text=batch_texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )
            inputs = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }

            with torch.inference_mode():
                outputs = model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )

            last_hidden_state = outputs.hidden_states[-1]
            attention_mask = inputs["attention_mask"]
            last_token_index = attention_mask.sum(dim=1) - 1
            pooled = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                last_token_index,
            ]
            if self.config.normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            batches.append(pooled.detach().cpu().to(torch.float32).numpy())

        return np.concatenate(batches, axis=0)


def build_node_context(node: dict[str, Any]) -> str:
    node_type = str(node.get("node_type", "")).strip().lower()
    page = node.get("page")
    text = str(node.get("text", "") or "").strip()

    prefix_lines: list[str] = [f"Node type: {node_type or 'unknown'}"]
    if page is not None:
        prefix_lines.append(f"Page: {page}")

    if node_type == "region":
        label = str(node.get("label", "") or "").strip()
        if label:
            prefix_lines.append(f"Region label: {label}")
    if node_type == "fine":
        modality = str(node.get("modality", "") or "").strip()
        if modality:
            prefix_lines.append(f"Fine modality: {modality}")

    content = text or "[EMPTY NODE CONTEXT]"
    return "\n".join(prefix_lines) + f"\nContent:\n{content}"


def build_node_embedding_records(
    graph_payload: dict[str, Any],
    target_node_types: tuple[str, ...] = ("line", "chunk", "region", "fine"),
) -> list[NodeEmbeddingRecord]:
    allowed = {item.strip().lower() for item in target_node_types}
    records: list[NodeEmbeddingRecord] = []

    for node in graph_payload.get("nodes", []):
        node_type = str(node.get("node_type", "")).strip().lower()
        if allowed and node_type not in allowed:
            continue

        records.append(
            NodeEmbeddingRecord(
                row=len(records),
                node_id=str(node.get("id", "")),
                node_type=node_type,
                page=node.get("page"),
                context_text=build_node_context(node),
                label=str(node.get("label", "") or ""),
                modality=str(node.get("modality", "") or ""),
            )
        )

    return records


def save_embedding_store(
    output_dir: str | Path,
    embeddings: np.ndarray,
    records: list[NodeEmbeddingRecord],
    graph_payload: dict[str, Any] | None = None,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embedding_path = output_dir / "embeddings.npy"
    meta_path = output_dir / "embedding_meta.json"

    graph_path: Path | None = None
    if graph_payload is not None:
        graph_path = save_graph_payload(graph_payload, output_dir / "graph.json")

    np.save(embedding_path, embeddings)
    meta_path.write_text(
        json.dumps([record.to_payload() for record in records], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    saved: dict[str, Path] = {
        "embedding_path": embedding_path,
        "meta_path": meta_path,
    }
    if graph_path is not None:
        saved["graph_path"] = graph_path
    return saved
