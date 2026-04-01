from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image, ImageDraw


DEFAULT_MODEL_REPO = "anyformat/doclayout-yolo-docstructbench"
DEFAULT_MODEL_FILE = "model.onnx"
DEFAULT_IMAGE_SIZE = 1024


def _norm_label(name: str) -> str:
    return name.strip().lower().replace("_", " ").replace("-", " ")


def _match_query_label(model_label: str, query_labels: list[str]) -> bool:
    if not query_labels:
        return True

    model_label_norm = _norm_label(model_label)
    query_set = {_norm_label(label) for label in query_labels}
    if model_label_norm in query_set:
        return True

    synonyms = {
        "table": {"table", "tabular"},
        "figure": {"figure", "graphic", "diagram", "illustration"},
        "image": {"image", "picture", "photo", "graphic", "figure"},
        "chart": {"chart", "graph", "plot", "diagram"},
    }
    for query in query_set:
        if query in synonyms and (
            model_label_norm in synonyms[query]
            or any(token in model_label_norm for token in synonyms[query])
        ):
            return True
        if query in model_label_norm:
            return True
    return False


def _box_area(box: list[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _inter_area(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def parse_ocr_lines(ocr_path: str | Path) -> list[dict[str, Any]]:
    path = Path(ocr_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []

    for page in payload.get("recognitionResults", []):
        for line in page.get("lines", []):
            polygon = line.get("boundingBox", [])
            if len(polygon) != 8:
                continue
            xs = [float(polygon[index]) for index in range(0, 8, 2)]
            ys = [float(polygon[index]) for index in range(1, 8, 2)]
            out.append(
                {
                    "text": str(line.get("text", "")).strip(),
                    "bbox_xyxy": [min(xs), min(ys), max(xs), max(ys)],
                }
            )

    return out


def attach_ocr_content(
    detections: list[dict[str, Any]],
    ocr_lines: list[dict[str, Any]],
    overlap_threshold: float = 0.35,
) -> None:
    for detection in detections:
        detection_box = detection["bbox_xyxy"]
        captured: list[str] = []
        for line in ocr_lines:
            line_box = line["bbox_xyxy"]
            intersection = _inter_area(detection_box, line_box)
            line_area = _box_area(line_box)
            if line_area <= 0.0:
                continue
            if intersection / line_area >= overlap_threshold and line.get("text"):
                captured.append(str(line["text"]))
        detection["content_lines"] = captured
        detection["content"] = "\n".join(captured).strip()


def _load_yolo_backend():
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(
            "DocLayout-YOLO requires the 'ultralytics' package. "
            "Install it before running layout detection."
        ) from exc
    return YOLO


def resolve_model_path(
    model_repo: str = DEFAULT_MODEL_REPO,
    model_file: str = DEFAULT_MODEL_FILE,
    local_model_path: str | Path | None = None,
) -> tuple[str, str]:
    if local_model_path is not None:
        local_path = Path(local_model_path)
        if local_path.exists():
            return str(local_path), local_path.name

    try:
        downloaded = hf_hub_download(repo_id=model_repo, filename=model_file)
        return downloaded, model_file
    except Exception:
        repo_files = list_repo_files(model_repo)
        candidate_files = [file for file in repo_files if file.lower().endswith(".pt")]
        if not candidate_files:
            raise RuntimeError(
                f"Could not find '{model_file}' or any .pt weight file in repo '{model_repo}'."
            )

        ranked = sorted(
            candidate_files,
            key=lambda file: (
                0 if "doclayout" in file.lower() else 1,
                0 if "yolo" in file.lower() else 1,
                0 if "best" in file.lower() else 1,
                len(file),
            ),
        )
        fallback_file = ranked[0]
        downloaded = hf_hub_download(repo_id=model_repo, filename=fallback_file)
        return downloaded, fallback_file


def run_doclayout_yolo(
    image_path: str | Path,
    labels: list[str] | None = None,
    score_threshold: float = 0.25,
    model_repo: str = DEFAULT_MODEL_REPO,
    model_file: str = DEFAULT_MODEL_FILE,
    local_model_path: str | Path | None = None,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> dict[str, Any]:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    YOLO = _load_yolo_backend()
    model_path, actual_model_file = resolve_model_path(
        model_repo=model_repo,
        model_file=model_file,
        local_model_path=local_model_path,
    )
    model = YOLO(model_path, task="detect")
    prediction = model.predict(
        source=str(image_path),
        conf=score_threshold,
        imgsz=image_size,
        verbose=False,
    )[0]

    label_filter = labels or []
    names = prediction.names if isinstance(prediction.names, dict) else {}
    detections: list[dict[str, Any]] = []

    for box in prediction.boxes:
        cls_idx = int(box.cls.item())
        score = float(box.conf.item())
        label = names.get(cls_idx, f"class_{cls_idx}")
        if label_filter and not _match_query_label(label, label_filter):
            continue

        x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
        detections.append(
            {
                "label": label,
                "score": score,
                "bbox_xyxy": [x1, y1, x2, y2],
            }
        )

    detections.sort(key=lambda item: item["score"], reverse=True)
    return {
        "image": str(image_path),
        "backend": "doclayout-yolo",
        "model_repo": model_repo,
        "model_file": actual_model_file,
        "queries": label_filter,
        "score_threshold": score_threshold,
        "num_detections": len(detections),
        "detections": detections,
    }


def render_bboxes(
    image_path: str | Path,
    detections: list[dict[str, Any]],
    out_path: str | Path,
) -> None:
    image_path = Path(image_path)
    out_path = Path(out_path)
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    palette = {
        "table": (34, 197, 94),
        "chart": (249, 115, 22),
        "figure": (168, 85, 247),
        "image": (59, 130, 246),
    }

    for detection in detections:
        label = str(detection.get("label", "obj"))
        score = float(detection.get("score", 0.0))
        x1, y1, x2, y2 = [int(value) for value in detection.get("bbox_xyxy", [0, 0, 0, 0])]
        x1, x2 = sorted((max(0, min(width - 1, x1)), max(0, min(width - 1, x2))))
        y1, y2 = sorted((max(0, min(height - 1, y1)), max(0, min(height - 1, y2))))
        if x2 <= x1 or y2 <= y1:
            continue

        color = palette.get(_norm_label(label), (99, 102, 241))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        tag = f"{label} {score:.2f}"
        tag_x1 = x1
        tag_y1 = max(0, y1 - 18)
        tag_x2 = min(width - 1, x1 + min(260, len(tag) * 8 + 10))
        tag_y2 = max(0, y1)
        draw.rectangle([tag_x1, tag_y1, tag_x2, tag_y2], fill=color)
        draw.text((tag_x1 + 4, tag_y1 + 2), tag, fill=(255, 255, 255))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def save_region_crops(
    image_path: str | Path,
    detections: list[dict[str, Any]],
    out_dir: str | Path,
) -> list[dict[str, Any]]:
    image_path = Path(image_path)
    out_dir = Path(out_dir)
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[dict[str, Any]] = []

    for index, detection in enumerate(detections, start=1):
        x1, y1, x2, y2 = [int(value) for value in detection.get("bbox_xyxy", [0, 0, 0, 0])]
        x1, x2 = sorted((max(0, min(width - 1, x1)), max(0, min(width - 1, x2))))
        y1, y2 = sorted((max(0, min(height - 1, y1)), max(0, min(height - 1, y2))))
        if x2 <= x1 or y2 <= y1:
            continue

        crop = image.crop((x1, y1, x2, y2))
        label = _norm_label(str(detection.get("label", "region"))).replace(" ", "_")
        crop_path = out_dir / f"{image_path.stem}_{index:03d}_{label}.png"
        crop.save(crop_path)
        detection["crop_path"] = str(crop_path)
        saved.append(
            {
                "index": index,
                "label": detection.get("label", ""),
                "crop_path": str(crop_path),
            }
        )

    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect table/chart/figure regions using DocLayout-YOLO."
    )
    parser.add_argument("--image", required=True, help="Path to document image file.")
    parser.add_argument(
        "--labels",
        default="table,chart,figure",
        help="Comma-separated query labels to keep.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.25,
        help="Detection confidence threshold.",
    )
    parser.add_argument(
        "--ocr-json",
        default="",
        help="Optional OCR JSON path to attach text content to each region.",
    )
    parser.add_argument("--out", default="", help="Optional output JSON path.")
    parser.add_argument(
        "--plot-out",
        default="",
        help="Optional output image path with rendered bounding boxes.",
    )
    parser.add_argument(
        "--save-crops-dir",
        default="",
        help="Optional directory to save each detected region crop.",
    )
    parser.add_argument(
        "--model-repo",
        default=DEFAULT_MODEL_REPO,
        help="Hugging Face repo for the DocLayout-YOLO checkpoint.",
    )
    parser.add_argument(
        "--model-file",
        default=DEFAULT_MODEL_FILE,
        help="Weight filename inside the Hugging Face repo.",
    )
    parser.add_argument(
        "--local-model-path",
        default="",
        help="Optional local ONNX/PT model path. If provided, skip download.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Prediction image size passed to YOLO.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = [item.strip() for item in args.labels.split(",") if item.strip()]
    result = run_doclayout_yolo(
        image_path=args.image,
        labels=labels,
        score_threshold=args.threshold,
        model_repo=args.model_repo,
        model_file=args.model_file,
        local_model_path=args.local_model_path or None,
        image_size=args.image_size,
    )

    if args.ocr_json:
        ocr_path = Path(args.ocr_json)
        if ocr_path.exists():
            ocr_lines = parse_ocr_lines(ocr_path)
            attach_ocr_content(result["detections"], ocr_lines)

    if args.save_crops_dir:
        result["saved_crops"] = save_region_crops(
            image_path=args.image,
            detections=result["detections"],
            out_dir=args.save_crops_dir,
        )

    if args.plot_out:
        render_bboxes(
            image_path=args.image,
            detections=result["detections"],
            out_path=args.plot_out,
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Saved: {out_path}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
