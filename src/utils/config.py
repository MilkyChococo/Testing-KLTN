from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(slots=True, frozen=True)
class ModelSelectionConfig:
    qwen_vl_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    qwen_embedding_model: str = "Qwen/Qwen3-VL-Embedding-2B"
    gemini_model: str = "gemini-2.5-flash"


DEFAULT_MODEL_CONFIG = ModelSelectionConfig()
DEFAULT_QWEN_VL_MODEL = DEFAULT_MODEL_CONFIG.qwen_vl_model
DEFAULT_QWEN_EMBED_MODEL = DEFAULT_MODEL_CONFIG.qwen_embedding_model
DEFAULT_GEMINI_MODEL = DEFAULT_MODEL_CONFIG.gemini_model


@dataclass(slots=True)
class QueryRequestConfig:
    topk: int = 10
    alpha: float = 0.5
    beta: float = 0.65
    gamma: float = 0.35
    lambda_cse: float = 0.1
    max_nodes: int = 100
    max_edges: int = 200


@dataclass(slots=True)
class DocLayoutConfig:
    model_repo: str = "anyformat/doclayout-yolo-docstructbench"
    model_file: str = "model.onnx"
    weights_path: Path = PROJECT_ROOT / "weights" / "doclayout_yolo.onnx"
    device: str = "cpu"
    conf_threshold: float = 0.75
    iou_threshold: float = 1
    image_size: int = 1024
    target_labels: tuple[str, ...] = ("table", "figure", "chart", "image")
    save_visualizations: bool = False
    save_crops: bool = False


@dataclass(slots=True)
class EmbeddingConfig:
    text_embedding_model: str = "google/embeddinggemma-300m"
    image_embedding_model: str = "clip-ViT-B-32"
    device: str = "cpu"
    text_dim: int = 768
    image_dim: int = 768


@dataclass(slots=True)
class QwenVLConfig:
    model: str = DEFAULT_QWEN_VL_MODEL
    device: str = "auto"
    dtype: str = "auto"
    temperature: float = 0.1
    max_new_tokens: int = 1024
    target_labels: tuple[str, ...] = ("table", "figure", "chart", "image")
