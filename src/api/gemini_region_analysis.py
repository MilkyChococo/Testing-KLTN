from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from src.api.qwen_vl_region_analysis import crop_detection_region
from src.utils.config import DEFAULT_GEMINI_MODEL as DEFAULT_GEMINI_MODEL_NAME, PROJECT_ROOT
from src.utils.prompt import get_qwen_region_prompt


DEFAULT_GEMINI_MODEL = DEFAULT_GEMINI_MODEL_NAME
DEFAULT_SUPPORTED_LABELS = ("table", "figure", "chart", "image")


def _load_dotenv_if_present() -> None:
    candidates = [PROJECT_ROOT / ".env", PROJECT_ROOT / "src" / ".env"]
    for path in candidates:
        if not path.is_file():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value and key not in os.environ:
                os.environ[key] = value


def _resolve_api_key(api_key: str | None = None) -> str:
    if api_key:
        return api_key.strip()
    _load_dotenv_if_present()
    for key in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    raise RuntimeError("Gemini API key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY.")


def _load_gemini_backend(api_key: str) -> tuple[str, Any, Any | None]:
    try:
        from google import genai
        from google.genai import types

        return "google.genai", genai.Client(api_key=api_key), types
    except ImportError:
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Gemini integration requires either 'google-genai' or 'google-generativeai'."
            ) from exc
        genai.configure(api_key=api_key)
        return "google.generativeai", genai, None


def _response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    candidates: list[str] = []
    for attr in ("output_text",):
        value = getattr(response, attr, None)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())

    if candidates:
        return "\n".join(candidates).strip()

    return str(response).strip()


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


@dataclass(slots=True)
class GeminiRegionAnalysis:
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


def analyze_region_with_gemini(
    image: Image.Image,
    region_id: str,
    label: str,
    ocr_text: str = "",
    model: str = DEFAULT_GEMINI_MODEL,
    api_key: str | None = None,
    temperature: float = 0.1,
    max_output_tokens: int = 1024,
) -> GeminiRegionAnalysis:
    resolved_api_key = _resolve_api_key(api_key)
    backend_name, backend, types_module = _load_gemini_backend(resolved_api_key)
    prompt = get_qwen_region_prompt(label=label, ocr_text=ocr_text)

    print(f"[Gemini] Preparing analysis for region '{region_id}' [{label}] via {backend_name}.", flush=True)
    if backend_name == "google.genai":
        config = None
        if types_module is not None:
            config = types_module.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
        response = backend.models.generate_content(
            model=model,
            contents=[prompt, image],
            config=config,
        )
    else:
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        gemini_model = backend.GenerativeModel(model_name=model)
        response = gemini_model.generate_content(
            [prompt, image],
            generation_config=generation_config,
        )

    raw_text = _response_text(response)
    payload = _parse_payload_from_text(raw_text=raw_text, label=label)
    return GeminiRegionAnalysis(
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


def annotate_detections_with_gemini(
    image_path: str | Path,
    detections: list[dict[str, Any]],
    labels: tuple[str, ...] = DEFAULT_SUPPORTED_LABELS,
    model: str = DEFAULT_GEMINI_MODEL,
    api_key: str | None = None,
    temperature: float = 0.1,
    max_output_tokens: int = 1024,
    save_crops_dir: str | Path | None = None,
    merge_into_content: bool = True,
) -> list[GeminiRegionAnalysis]:
    image_path = Path(image_path)
    supported = {_canonical_label(label) for label in labels}
    analyses: list[GeminiRegionAnalysis] = []

    for index, detection in enumerate(detections):
        raw_label = str(detection.get("label", "region"))
        canonical_label = _canonical_label(raw_label)
        if supported and canonical_label not in supported:
            continue

        print(
            f"[Gemini] Region {index + 1}/{len(detections)}: "
            f"{detection.get('id', f'region_{index:03d}')} [{raw_label}]",
            flush=True,
        )

        crop_path: Path | None = None
        if save_crops_dir is not None:
            norm_label = _norm_label(raw_label).replace(" ", "_")
            crop_path = Path(save_crops_dir) / f"{image_path.stem}_{index:03d}_{norm_label}.png"

        crop = crop_detection_region(
            image_path=image_path,
            detection=detection,
            crop_path=crop_path,
        )
        ocr_text = str(detection.get("content", "")).strip()
        region_id = str(detection.get("id") or f"region_{index:03d}")
        analysis = analyze_region_with_gemini(
            image=crop,
            region_id=region_id,
            label=raw_label,
            ocr_text=ocr_text,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        analyses.append(analysis)

        detection["ocr_content"] = ocr_text
        detection["gemini_analysis"] = analysis.to_payload()
        detection["gemini_model"] = model
        detection["gemini_text"] = analysis.to_graph_text(ocr_text=ocr_text)
        if merge_into_content:
            detection["content"] = detection["gemini_text"]

    return analyses
