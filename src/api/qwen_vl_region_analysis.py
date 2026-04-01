from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from src.utils.config import DEFAULT_QWEN_VL_MODEL
from src.utils.prompt import get_qwen_region_prompt


DEFAULT_QWEN_MODEL = DEFAULT_QWEN_VL_MODEL
DEFAULT_SUPPORTED_LABELS = ("table", "figure", "chart", "image")

_MODEL_CACHE: dict[tuple[str, str, str], tuple[Any, Any, str]] = {}


@dataclass(slots=True)
class QwenRegionAnalysis:
    region_id: str
    label: str
    model: str
    region_type: str
    title_or_topic: str
    summary: str
    structured_content: str
    key_points: list[str] = field(default_factory=list)
    visible_text: list[str] = field(default_factory=list)
    raw_response_text: str = ""
    raw_payload: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    def to_graph_text(self, ocr_text: str = "") -> str:
        parts: list[str] = []
        if self.title_or_topic:
            parts.append(f"Topic: {self.title_or_topic}")
        if self.summary:
            parts.append(f"Summary: {self.summary}")
        if self.structured_content:
            parts.append(f"Structured content:\n{self.structured_content}")
        if self.key_points:
            parts.append("Key points: " + "; ".join(item for item in self.key_points if item))
        if self.visible_text:
            parts.append("Visible text: " + "; ".join(item for item in self.visible_text if item))
        if ocr_text.strip():
            parts.append(f"OCR text:\n{ocr_text.strip()}")
        return "\n\n".join(part for part in parts if part).strip()


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Torch is required for Qwen region analysis. Install torch in the active environment first."
        ) from exc
    return torch


def _norm_label(label: str) -> str:
    return label.strip().lower().replace("_", " ").replace("-", " ")


def _canonical_label(label: str) -> str:
    norm = _norm_label(label)
    if norm in {"bordered", "borderless", "table"}:
        return "table"
    if any(token in norm for token in ("chart", "graph", "plot")):
        return "chart"
    if any(token in norm for token in ("figure", "diagram", "illustration")):
        return "figure"
    if any(token in norm for token in ("image", "photo", "picture")):
        return "image"
    return norm


def _image_bbox(detection: dict[str, Any]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(float(value)) for value in detection.get("bbox_xyxy", [0, 0, 0, 0])]
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox in detection: {detection.get('bbox_xyxy')}")
    return x1, y1, x2, y2


def crop_detection_region(
    image_path: str | Path,
    detection: dict[str, Any],
    crop_path: str | Path | None = None,
) -> Image.Image:
    image_path = Path(image_path)
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    x1, y1, x2, y2 = _image_bbox(detection)
    x1, x2 = sorted((max(0, min(width - 1, x1)), max(0, min(width - 1, x2))))
    y1, y2 = sorted((max(0, min(height - 1, y1)), max(0, min(height - 1, y2))))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Detection bbox is outside image bounds or does not match this image's coordinate space: "
            f"{detection.get('bbox_xyxy')}"
        )

    crop = image.crop((x1, y1, x2, y2))
    if crop_path is not None:
        out_path = Path(crop_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(out_path)
        detection["crop_path"] = str(out_path)
    return crop


def _resolve_device(device: str) -> str:
    torch = _torch()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _resolve_dtype(device: str, dtype: str) -> Any:
    torch = _torch()
    if dtype == "auto":
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
        return mapping[dtype.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {dtype}") from exc


def _dtype_name(dtype: torch.dtype) -> str:
    torch = _torch()
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    return "float32"


def _resolve_qwen_model_class(model_id: str) -> Any:
    try:
        import transformers
    except ImportError as exc:
        raise RuntimeError(
            "Qwen region analysis requires transformers with Qwen-VL support."
        ) from exc

    model_name = model_id.strip().lower()
    qwen25_cls = getattr(transformers, "Qwen2_5_VLForConditionalGeneration", None)
    qwen3_cls = getattr(transformers, "Qwen3VLForConditionalGeneration", None)

    if "qwen3-vl" in model_name:
        if qwen3_cls is None:
            raise RuntimeError(
                "This environment does not provide Qwen3VLForConditionalGeneration. "
                "Install a transformers version with Qwen3-VL support."
            )
        return qwen3_cls

    if "qwen2.5-vl" in model_name:
        if qwen25_cls is None:
            raise RuntimeError(
                "This environment does not provide Qwen2_5_VLForConditionalGeneration. "
                "Install a transformers version with Qwen2.5-VL support."
            )
        return qwen25_cls

    if qwen3_cls is not None:
        return qwen3_cls
    if qwen25_cls is not None:
        return qwen25_cls

    raise RuntimeError(
        "No supported Qwen-VL conditional generation class was found in transformers."
    )


def _load_model_and_processor(
    model_id: str = DEFAULT_QWEN_MODEL,
    device: str = "auto",
    dtype: str = "auto",
) -> tuple[Any, Any, str]:
    torch = _torch()
    resolved_device = _resolve_device(device)
    resolved_dtype = _resolve_dtype(resolved_device, dtype)
    cache_key = (model_id, resolved_device, _dtype_name(resolved_dtype))
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        print(
            f"[Qwen] Reusing cached model {model_id} on {resolved_device} ({_dtype_name(resolved_dtype)}).",
            flush=True,
        )
        return cached

    try:
        from transformers import AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "Qwen region analysis requires transformers with Qwen-VL support."
        ) from exc

    model_class = _resolve_qwen_model_class(model_id)

    print(f"[Qwen] Loading processor: {model_id}", flush=True)
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=True,
    )

    print(
        f"[Qwen] Loading model: {model_id} on {resolved_device} ({_dtype_name(resolved_dtype)})",
        flush=True,
    )
    try:
        model = model_class.from_pretrained(
            model_id,
            torch_dtype=resolved_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    except OSError as exc:
        message = str(exc)
        if "safetensors" in message and model_id in message:
            raise RuntimeError(
                f"Qwen model cache appears incomplete for '{model_id}'. Delete the local cache folder "
                f"'C:\\Users\\GIGABYTE\\.cache\\huggingface\\hub\\models--{model_id.replace('/', '--')}' "
                "and run again to re-download all shards."
            ) from exc
        raise

    print(f"[Qwen] Moving model to device: {resolved_device}", flush=True)
    model = model.to(resolved_device)
    model.eval()
    print("[Qwen] Model ready.", flush=True)

    cached = (processor, model, resolved_device)
    _MODEL_CACHE[cache_key] = cached
    return cached


def _build_model_inputs(processor: Any, image: Image.Image, prompt: str) -> dict[str, Any]:
    return _build_model_inputs_batch(
        processor=processor,
        images=[image],
        prompts=[prompt],
    )


def _build_model_inputs_batch(
    processor: Any,
    images: list[Image.Image],
    prompts: list[str],
) -> dict[str, Any]:
    if len(images) != len(prompts):
        raise ValueError("The number of images and prompts must match for batch Qwen analysis.")

    chat_texts: list[str] = []
    for prompt in prompts:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        chat_texts.append(
            processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    return processor(
        text=chat_texts,
        images=images,
        padding=True,
        return_tensors="pt",
    )


def _build_generate_kwargs(max_new_tokens: int, temperature: float) -> dict[str, Any]:
    do_sample = temperature > 0.0
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature
    return generate_kwargs


def _decode_generated_texts(
    processor: Any,
    generated_ids: Any,
    inputs: dict[str, Any],
) -> list[str]:
    prompt_length = int(inputs["input_ids"].shape[1])
    generated_trimmed = generated_ids[:, prompt_length:]
    return [
        item.strip()
        for item in processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    ]


def analyze_regions_batch_with_qwen(
    images: list[Image.Image],
    region_ids: list[str],
    labels: list[str],
    ocr_texts: list[str] | None = None,
    model: str = DEFAULT_QWEN_MODEL,
    device: str = "auto",
    dtype: str = "auto",
    max_new_tokens: int = 1024,
    temperature: float = 0.1,
) -> list[QwenRegionAnalysis]:
    if not images:
        return []
    if not (len(images) == len(region_ids) == len(labels)):
        raise ValueError("images, region_ids, and labels must have the same length.")

    ocr_texts = ocr_texts or [""] * len(images)
    if len(ocr_texts) != len(images):
        raise ValueError("ocr_texts must have the same length as images when provided.")

    torch = _torch()
    processor, qwen_model, resolved_device = _load_model_and_processor(
        model_id=model,
        device=device,
        dtype=dtype,
    )
    prompts = [
        get_qwen_region_prompt(label=label, ocr_text=ocr_text)
        for label, ocr_text in zip(labels, ocr_texts)
    ]

    joined_region_ids = ", ".join(region_ids)
    print(f"[Qwen] Building batch inputs for regions: {joined_region_ids}.", flush=True)
    inputs = _build_model_inputs_batch(
        processor=processor,
        images=images,
        prompts=prompts,
    )
    inputs = {
        key: value.to(resolved_device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    print(f"[Qwen] Generating batch output for regions: {joined_region_ids}.", flush=True)
    with torch.inference_mode():
        generated_ids = qwen_model.generate(
            **inputs,
            **_build_generate_kwargs(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            ),
        )

    raw_texts = _decode_generated_texts(
        processor=processor,
        generated_ids=generated_ids,
        inputs=inputs,
    )

    analyses: list[QwenRegionAnalysis] = []
    for region_id, label, ocr_text, raw_text in zip(region_ids, labels, ocr_texts, raw_texts):
        payload = _parse_payload_from_text(raw_text=raw_text, label=label)
        analyses.append(
            QwenRegionAnalysis(
                region_id=region_id,
                label=label,
                model=model,
                region_type=str(payload.get("region_type", _canonical_label(label))).strip(),
                title_or_topic=str(payload.get("title_or_topic", "")).strip(),
                summary=str(payload.get("summary", "")).strip(),
                structured_content=str(payload.get("structured_content", "")).strip(),
                key_points=[str(item).strip() for item in payload.get("key_points", []) if str(item).strip()],
                visible_text=[str(item).strip() for item in payload.get("visible_text", []) if str(item).strip()],
                raw_response_text=raw_text,
                raw_payload=payload,
            )
        )

    print(f"[Qwen] Finished batch for regions: {joined_region_ids}.", flush=True)
    return analyses


def _extract_json_payload(raw_text: str) -> dict[str, Any]:
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    candidates = [cleaned]
    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidates.insert(0, cleaned[first : last + 1])

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _parse_payload_from_text(raw_text: str, label: str) -> dict[str, Any]:
    payload = _extract_json_payload(raw_text)
    if payload:
        return payload

    return {
        "region_type": _canonical_label(label),
        "title_or_topic": "",
        "summary": raw_text.strip(),
        "structured_content": "",
        "key_points": [],
        "visible_text": [],
    }


def analyze_region_with_qwen(
    image: Image.Image,
    region_id: str,
    label: str,
    ocr_text: str = "",
    model: str = DEFAULT_QWEN_MODEL,
    device: str = "auto",
    dtype: str = "auto",
    max_new_tokens: int = 1024,
    temperature: float = 0.1,
) -> QwenRegionAnalysis:
    print(f"[Qwen] Preparing analysis for region '{region_id}' [{label}].", flush=True)
    return analyze_regions_batch_with_qwen(
        images=[image],
        region_ids=[region_id],
        labels=[label],
        ocr_texts=[ocr_text],
        model=model,
        device=device,
        dtype=dtype,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )[0]


def annotate_detections_with_qwen(
    image_path: str | Path,
    detections: list[dict[str, Any]],
    labels: tuple[str, ...] = DEFAULT_SUPPORTED_LABELS,
    model: str = DEFAULT_QWEN_MODEL,
    device: str = "auto",
    dtype: str = "auto",
    max_new_tokens: int = 1024,
    temperature: float = 0.1,
    save_crops_dir: str | Path | None = None,
    merge_into_content: bool = True,
    batch_size: int = 1,
) -> list[QwenRegionAnalysis]:
    image_path = Path(image_path)
    supported = {_canonical_label(label) for label in labels}
    analyses: list[QwenRegionAnalysis] = []
    prepared_items: list[tuple[int, dict[str, Any], Image.Image, str, str]] = []

    for index, detection in enumerate(detections):
        raw_label = str(detection.get("label", "region"))
        norm_label = _norm_label(raw_label)
        canonical_label = _canonical_label(raw_label)
        if supported and canonical_label not in supported:
            continue

        print(
            f"[Qwen] Region {index + 1}/{len(detections)}: "
            f"{detection.get('id', f'region_{index:03d}')} [{raw_label}]",
            flush=True,
        )

        crop_path: Path | None = None
        if save_crops_dir is not None:
            crop_path = Path(save_crops_dir) / f"{image_path.stem}_{index:03d}_{norm_label.replace(' ', '_')}.png"

        crop = crop_detection_region(
            image_path=image_path,
            detection=detection,
            crop_path=crop_path,
        )
        ocr_text = str(detection.get("content", "")).strip()
        region_id = str(detection.get("id") or f"region_{index:03d}")
        prepared_items.append((index, detection, crop, raw_label, ocr_text))

    effective_batch_size = max(1, int(batch_size))
    for start in range(0, len(prepared_items), effective_batch_size):
        batch_items = prepared_items[start : start + effective_batch_size]
        batch_analyses = analyze_regions_batch_with_qwen(
            images=[item[2] for item in batch_items],
            region_ids=[str(item[1].get("id") or f"region_{item[0]:03d}") for item in batch_items],
            labels=[item[3] for item in batch_items],
            ocr_texts=[item[4] for item in batch_items],
            model=model,
            device=device,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        analyses.extend(batch_analyses)

        for (_, detection, _, _, ocr_text), analysis in zip(batch_items, batch_analyses):
            detection["ocr_content"] = ocr_text
            detection["qwen_analysis"] = analysis.to_payload()
            detection["qwen_model"] = model
            detection["qwen_text"] = analysis.to_graph_text(ocr_text=ocr_text)
            if merge_into_content:
                detection["content"] = detection["qwen_text"]

    return analyses
