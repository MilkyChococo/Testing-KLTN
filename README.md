# VQA KLTN

Pipeline DocVQA / InfographicVQA theo hướng graph:
- OCR `word -> line -> chunk`
- layout / region analysis cho tài liệu thường
- node embedding
- Controlled Subgraph Expansion (CSE)
- final answer bằng VLM/LLM

## Cấu trúc chính

```text
scripts/
  run_offline_index.py
  run_cse_qa.py
  run_offline_index_gemini.py
  run_cse_qa_gemini.py

inference/
  index_offline_qwen.py
  run_cse_qwen.py
  index_offline_gemini.py
  run_cse_gemini.py

src/
  extract_sp/       # pipeline cho SP-DocVQA / document thường
  extract_infor/    # parser + chunk + relation cho infographic
  api/              # Qwen / Gemini region analysis + answering
  database/         # graph store + node embedding
  algo/             # CSE indexing + online CSE
  utils/            # config, prompt, io, fallback

test_info/
  plot_infographic_ocr.py
  plot_infographic_chunks.py
```

## Nhánh document thường

Workflow chính:
1. OCR words -> lines
2. lines -> chunks
3. detect layout bằng DocLayout-YOLO
4. phân tích region bằng Qwen hoặc Gemini
5. build `graph.json`
6. embed node context
7. build `graph_enriched.json`
8. query -> CSE -> answer

Offline:

```bash
python scripts/run_offline_index.py dataset/spdocvqa/spdocvqa_images/fggh0224_12.png --detect-layout --analyze-regions-with-qwen --embed-device cpu --embed-dtype float32
```

Online:

```bash
python scripts/run_cse_qa.py artifacts/node_stores/fggh0224_12 "what is the effective date?" --embed-device cpu --embed-dtype float32 --answer-device cpu --answer-dtype float32
```

## Nhánh infographic

Infographic OCR không dùng schema cũ của SP-DocVQA. Nó đang ở dạng block:
- `PAGE`
- `LINE`
- `WORD`

Bounding box là normalized coordinates, nên có parser riêng ở:
- [src/extract_infor/ocr_parser.py](src/extract_infor/ocr_parser.py)

Chunk cho infographic đang build theo heuristic clustering hình học ở:
- [src/extract_infor/line_to_chunk.py](src/extract_infor/line_to_chunk.py)

Relation text-only cho infographic đang thử nghiệm ở:
- [src/extract_infor/relations.py](src/extract_infor/relations.py)

Hiện tại nhánh infographic mới tập trung vào:
- `line`
- `chunk`
- `line-line`
- `line-chunk`
- `chunk-chunk`

Chưa nối:
- `image`
- `chart`
- `figure`

## Plot / debug infographic

Vẽ OCR block:

```bash
python test_info/plot_infographic_ocr.py dataset/infographic/infographicsvqa_images/10002.jpeg --block-types LINE,WORD
```

Vẽ line + chunk + relation:

```bash
python test_info/plot_infographic_chunks.py dataset/infographic/infographicsvqa_images/10002.jpeg --no-show
```

Output mặc định:

```text
artifacts/visualizations/<image_stem>_infographic_chunks.png
```

## Dữ liệu graph

Một document store hiện thường có:

```text
graph.json
embeddings.npy
embedding_meta.json
graph_enriched.json
```

`graph_enriched.json` dùng cho CSE, có thêm:
- `embedding_row`
- `deg_in`
- `deg_out`
- `deg`
- `hub`
- `neighbors`
- `conf_off`

## Công thức CSE hiện tại

Offline:

```text
conf_off(u, v) = (1 + cos(e(u), e(v))) / 2
hub(v) = log(1 + deg(v))
```

Online:

```text
rel(v | Q) = (1 + cos(q, e(v))) / 2
score(u, v) = alpha * conf_off(u, v) + rel(v | Q) - lambda * hub(v)
```

Baseline:
- `alpha = 0.5`
- `lambda = 0.1`

## Model mặc định

Default model được gom ở:
- [src/utils/config.py](src/utils/config.py)

Các nhóm model chính:
- layout detection
- region analysis / answer model
- embedding model
- Gemini fallback model

Chỉ cần đổi trong `config.py` để đổi default toàn repo.

## Lưu ý

- `dataset/` và `artifacts/` đang được ignore, không push lên repo.
- Nếu offline index dừng giữa chừng thì có thể mới chỉ thấy `graph.json`.
- Với model lớn, nên tách:
  - build offline store trước
  - answer sau
