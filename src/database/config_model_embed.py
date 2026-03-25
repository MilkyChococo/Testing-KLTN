from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.utils.config import DEFAULT_QWEN_EMBED_MODEL, PROJECT_ROOT


@dataclass(slots=True)
class Qwen3VLEmbeddingConfig:
    model: str = DEFAULT_QWEN_EMBED_MODEL
    device: str = "auto"
    dtype: str = "auto"
    batch_size: int = 8
    max_length: int = 8192
    normalize: bool = True
    document_instruction: str = (
        "Represent this document graph node for retrieval in document question answering."
    )
    query_instruction: str = (
        "Represent the user's document question to retrieve relevant graph nodes."
    )
    target_node_types: tuple[str, ...] = ("line", "chunk", "region", "fine")
    output_root: Path = PROJECT_ROOT / "artifacts" / "node_stores"
