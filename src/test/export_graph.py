from __future__ import annotations

import argparse
import html
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.graph_store import build_graph_payload, save_graph_payload
from src.extract.document_pipeline import run_document_pipeline


DEFAULT_OCR_DIR = PROJECT_ROOT / "dataset" / "spdocvqa" / "spdocvqa_ocr"
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "dataset" / "spdocvqa" / "spdocvqa_images"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "graphs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the document graph built from OCR, layout, and region analysis to JSON."
    )
    parser.add_argument(
        "image",
        type=Path,
        help="Path to the input image. If relative, it is resolved from the project root.",
    )
    parser.add_argument(
        "--ocr",
        type=Path,
        default=None,
        help="Optional OCR JSON path. Defaults to spdocvqa_ocr/<image_stem>.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON path. Defaults to artifacts/graphs/<image_stem>_graph.json.",
    )
    parser.add_argument(
        "--html-output",
        type=Path,
        default=None,
        help="Optional output HTML path. Defaults to the JSON path with .html suffix.",
    )
    parser.add_argument(
        "--page",
        type=int,
        default=1,
        help="OCR page number to load. Default is 1.",
    )
    parser.add_argument(
        "--layout-json",
        type=Path,
        default=None,
        help="Optional layout detection JSON.",
    )
    parser.add_argument(
        "--detect-layout",
        action="store_true",
        help="Run DocLayout-YOLO instead of loading a layout JSON file.",
    )
    parser.add_argument(
        "--layout-threshold",
        type=float,
        default=0.7,
        help="Detection score threshold for DocLayout-YOLO.",
    )
    parser.add_argument(
        "--layout-labels",
        default="table,figure,chart,image",
        help="Comma-separated labels to keep when running layout detection.",
    )
    parser.add_argument(
        "--local-model-path",
        type=Path,
        default=None,
        help="Optional local DocLayout-YOLO model path.",
    )
    parser.add_argument(
        "--analyze-regions-with-qwen",
        action="store_true",
        help="Run Qwen2.5-VL on detected regions and inject structured context into region nodes.",
    )
    parser.add_argument(
        "--qwen-model",
        default=DEFAULT_QWEN_VL_MODEL,
        help="Qwen2.5-VL model name used for region analysis.",
    )
    parser.add_argument(
        "--qwen-device",
        default="auto",
        help="Qwen inference device: auto, cuda, or cpu.",
    )
    parser.add_argument(
        "--qwen-dtype",
        default="auto",
        help="Qwen inference dtype: auto, float16, bfloat16, or float32.",
    )
    parser.add_argument(
        "--qwen-max-new-tokens",
        type=int,
        default=1024,
        help="Max generated tokens for Qwen region analysis.",
    )
    parser.add_argument(
        "--qwen-temperature",
        type=float,
        default=0.1,
        help="Temperature for Qwen region analysis.",
    )
    parser.add_argument(
        "--qwen-crops-dir",
        type=Path,
        default=None,
        help="Optional directory to save region crops sent to Qwen.",
    )
    return parser.parse_args()


def resolve_image_path(image_path: Path) -> Path:
    if image_path.is_absolute():
        return image_path
    if image_path.exists():
        return image_path.resolve()
    candidate = PROJECT_ROOT / image_path
    if candidate.exists():
        return candidate.resolve()
    candidate = DEFAULT_IMAGE_DIR / image_path.name
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f"Image not found: {image_path}")


def resolve_ocr_path(image_path: Path, ocr_path: Path | None) -> Path:
    if ocr_path is not None:
        if ocr_path.is_absolute():
            return ocr_path
        if ocr_path.exists():
            return ocr_path.resolve()
        candidate = PROJECT_ROOT / ocr_path
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(f"OCR file not found: {ocr_path}")

    candidate = DEFAULT_OCR_DIR / f"{image_path.stem}.json"
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(
        f"Could not infer OCR JSON for image stem '{image_path.stem}' in {DEFAULT_OCR_DIR}"
    )


def resolve_optional_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    candidate = PROJECT_ROOT / path
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f"File not found: {path}")


def build_html_report(payload: dict[str, Any]) -> str:
    stats = payload.get("stats", {})
    nodes = payload.get("nodes", [])
    edges = payload.get("edges", [])
    node_type_counts = Counter(str(node.get("node_type", "unknown")) for node in nodes)
    relation_counts = Counter(str(edge.get("relation", "unknown")) for edge in edges)

    stats_cards = "\n".join(
        f"""
        <div class="card">
          <div class="label">{html.escape(str(key))}</div>
          <div class="value">{html.escape(str(value))}</div>
        </div>
        """.strip()
        for key, value in stats.items()
    )

    node_type_rows = "\n".join(
        f"<tr><td>{html.escape(node_type)}</td><td>{count}</td></tr>"
        for node_type, count in sorted(node_type_counts.items())
    )
    relation_rows = "\n".join(
        f"<tr><td>{html.escape(relation)}</td><td>{count}</td></tr>"
        for relation, count in sorted(relation_counts.items())
    )

    data_json = json.dumps(payload, ensure_ascii=False)
    image_path = html.escape(str(payload.get("document", {}).get("image_path", "")))
    ocr_path = html.escape(str(payload.get("document", {}).get("ocr_path", "")))

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Graph Report</title>
  <style>
    :root {{
      --bg: #f7f4eb;
      --panel: #fffdf7;
      --ink: #1f1d1a;
      --muted: #6d655d;
      --line: #d7cfc0;
      --accent: #005f73;
      --accent-2: #bb3e03;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", system-ui, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, #fff6d7 0, transparent 26%),
        linear-gradient(135deg, #f7f4eb 0%, #efe6d7 100%);
    }}
    .wrap {{
      max-width: 1360px;
      margin: 0 auto;
      padding: 24px;
    }}
    .hero {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 22px 24px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.06);
      margin-bottom: 18px;
    }}
    h1, h2 {{ margin: 0 0 12px; }}
    .muted {{ color: var(--muted); }}
    .path {{
      font-family: Consolas, monospace;
      font-size: 13px;
      overflow-wrap: anywhere;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 12px;
      margin: 18px 0;
    }}
    .card {{
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
    }}
    .label {{
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: .04em;
    }}
    .value {{
      margin-top: 6px;
      font-size: 24px;
      font-weight: 700;
      color: var(--accent);
    }}
    .grid {{
      display: grid;
      grid-template-columns: 320px 1fr;
      gap: 18px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 8px 20px rgba(0,0,0,0.05);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    th, td {{
      padding: 10px 8px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      font-size: 13px;
    }}
    th {{
      color: var(--muted);
      font-weight: 600;
      position: sticky;
      top: 0;
      background: var(--panel);
    }}
    .controls {{
      display: flex;
      gap: 10px;
      margin-bottom: 12px;
      flex-wrap: wrap;
    }}
    input, select {{
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px 12px;
      background: #fff;
      color: var(--ink);
    }}
    .scroll {{
      max-height: 70vh;
      overflow: auto;
    }}
    .pill {{
      display: inline-block;
      padding: 3px 8px;
      border-radius: 999px;
      background: #e9f3f5;
      color: var(--accent);
      font-size: 12px;
      font-weight: 600;
    }}
    .text {{
      max-width: 480px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }}
    .meta {{
      max-width: 360px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-family: Consolas, monospace;
      font-size: 12px;
    }}
    @media (max-width: 980px) {{
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>Document Graph Report</h1>
      <div class="muted">Image</div>
      <div class="path">{image_path}</div>
      <div class="muted" style="margin-top:10px;">OCR</div>
      <div class="path">{ocr_path}</div>
      <div class="cards">{stats_cards}</div>
    </section>

    <section class="grid">
      <div class="panel">
        <h2>Node Types</h2>
        <table>
          <thead><tr><th>Type</th><th>Count</th></tr></thead>
          <tbody>{node_type_rows}</tbody>
        </table>
        <h2 style="margin-top:18px;">Relations</h2>
        <table>
          <thead><tr><th>Relation</th><th>Count</th></tr></thead>
          <tbody>{relation_rows}</tbody>
        </table>
      </div>

      <div class="panel">
        <div class="controls">
          <input id="nodeSearch" type="text" placeholder="Search node text or id">
          <select id="nodeTypeFilter">
            <option value="">All node types</option>
          </select>
        </div>
        <div class="scroll">
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Type</th>
                <th>Page</th>
                <th>Text</th>
                <th>Metadata</th>
              </tr>
            </thead>
            <tbody id="nodesTable"></tbody>
          </table>
        </div>
      </div>
    </section>

    <section class="panel" style="margin-top:18px;">
      <div class="controls">
        <input id="edgeSearch" type="text" placeholder="Search source / target / relation">
        <select id="edgeTypeFilter">
          <option value="">All relations</option>
        </select>
      </div>
      <div class="scroll">
        <table>
          <thead>
            <tr>
              <th>Source</th>
              <th>Target</th>
              <th>Relation</th>
              <th>Score</th>
            </tr>
          </thead>
          <tbody id="edgesTable"></tbody>
        </table>
      </div>
    </section>
  </div>

  <script>
    const payload = {data_json};
    const nodes = payload.nodes || [];
    const edges = payload.edges || [];

    const nodeSearch = document.getElementById('nodeSearch');
    const nodeTypeFilter = document.getElementById('nodeTypeFilter');
    const edgeSearch = document.getElementById('edgeSearch');
    const edgeTypeFilter = document.getElementById('edgeTypeFilter');
    const nodesTable = document.getElementById('nodesTable');
    const edgesTable = document.getElementById('edgesTable');

    function uniqueSorted(values) {{
      return [...new Set(values)].filter(Boolean).sort();
    }}

    function populateSelect(select, values) {{
      for (const value of uniqueSorted(values)) {{
        const option = document.createElement('option');
        option.value = value;
        option.textContent = value;
        select.appendChild(option);
      }}
    }}

    function escapeHtml(value) {{
      return String(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;');
    }}

    function renderNodes() {{
      const query = nodeSearch.value.trim().toLowerCase();
      const type = nodeTypeFilter.value;
      const filtered = nodes.filter((node) => {{
        const hay = JSON.stringify([node.id, node.node_type, node.text, node.metadata]).toLowerCase();
        return (!type || node.node_type === type) && (!query || hay.includes(query));
      }});

      nodesTable.innerHTML = filtered.map((node) => `
        <tr>
          <td><span class="pill">${{escapeHtml(node.id)}}</span></td>
          <td>${{escapeHtml(node.node_type)}}</td>
          <td>${{escapeHtml(node.page ?? '')}}</td>
          <td><div class="text">${{escapeHtml(node.text ?? '')}}</div></td>
          <td><div class="meta">${{escapeHtml(JSON.stringify(node.metadata ?? {{}}, null, 2))}}</div></td>
        </tr>
      `).join('');
    }}

    function renderEdges() {{
      const query = edgeSearch.value.trim().toLowerCase();
      const relation = edgeTypeFilter.value;
      const filtered = edges.filter((edge) => {{
        const hay = JSON.stringify([edge.source_id, edge.target_id, edge.relation]).toLowerCase();
        return (!relation || edge.relation === relation) && (!query || hay.includes(query));
      }});

      edgesTable.innerHTML = filtered.map((edge) => `
        <tr>
          <td>${{escapeHtml(edge.source_id)}}</td>
          <td>${{escapeHtml(edge.target_id)}}</td>
          <td>${{escapeHtml(edge.relation)}}</td>
          <td>${{Number(edge.score ?? 1).toFixed(4)}}</td>
        </tr>
      `).join('');
    }}

    populateSelect(nodeTypeFilter, nodes.map((node) => node.node_type));
    populateSelect(edgeTypeFilter, edges.map((edge) => edge.relation));

    nodeSearch.addEventListener('input', renderNodes);
    nodeTypeFilter.addEventListener('change', renderNodes);
    edgeSearch.addEventListener('input', renderEdges);
    edgeTypeFilter.addEventListener('change', renderEdges);

    renderNodes();
    renderEdges();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    image_path = resolve_image_path(args.image)
    ocr_path = resolve_ocr_path(image_path, args.ocr)
    layout_path = resolve_optional_path(args.layout_json)

    pipeline = run_document_pipeline(
        image_path=image_path,
        ocr_path=ocr_path,
        page_number=args.page,
        layout_path=layout_path,
        detect_layout=args.detect_layout,
        layout_labels=[
            label.strip()
            for label in args.layout_labels.split(",")
            if label.strip()
        ],
        layout_threshold=args.layout_threshold,
        local_model_path=resolve_optional_path(args.local_model_path),
        include_visual_fine_nodes=False,
        analyze_regions_with_qwen=args.analyze_regions_with_qwen,
        qwen_labels=[
            label.strip()
            for label in args.layout_labels.split(",")
            if label.strip()
        ],
        qwen_model=args.qwen_model,
        qwen_device=args.qwen_device,
        qwen_dtype=args.qwen_dtype,
        qwen_max_new_tokens=args.qwen_max_new_tokens,
        qwen_temperature=args.qwen_temperature,
        qwen_crops_dir=resolve_optional_path(args.qwen_crops_dir),
    )

    payload = build_graph_payload(pipeline)

    if args.output is not None:
        output_path = args.output
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
    else:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = DEFAULT_OUTPUT_DIR / f"{image_path.stem}_graph.json"

    output_path = save_graph_payload(payload, output_path)
    if args.html_output is not None:
        html_output_path = args.html_output
        if not html_output_path.is_absolute():
            html_output_path = PROJECT_ROOT / html_output_path
    else:
        html_output_path = output_path.with_suffix(".html")

    html_output_path.parent.mkdir(parents=True, exist_ok=True)
    html_output_path.write_text(
        build_html_report(payload),
        encoding="utf-8",
    )
    print(f"Saved graph JSON to: {output_path}")
    print(f"Saved graph HTML to: {html_output_path}")


if __name__ == "__main__":
    main()

