# Document Q&A Agent

An end-to-end retrieval-augmented question answering project for PDF collections, built with **PyTorch, Hugging Face Transformers, LangChain, LangGraph, FAISS, and IBM watsonx.ai**.

Topics: `rag` · `langchain` · `faiss` · `pytorch` · `watsonx` · `genai` · `nlp`

## Architecture

```text
 PDF files
    │
    ▼
 pypdf page extraction ──► text cleanup + page metadata
                                     │
                                     ▼
                         token-aware recursive chunks
                                     │
                                     ▼
 Hugging Face AutoTokenizer/AutoModel (batched PyTorch, mean pool, L2 norm)
                                     │
                                     ▼
                      FAISS IndexFlatIP + JSON metadata
                                     │
 question ──► LangGraph agent ──► search_documents / get_document_list / summarize_document
                    │                                │
                    └──── ChatWatsonx (or offline FakeLLM) ◄──┘
                                     │
                                     ▼
                    answer + [file, page] citations + latency
```

## Requirements and setup

- Python 3.10 or newer
- Windows, macOS, or Linux
- Internet access for the first install and to download the Hugging Face embedding model
- IBM watsonx.ai credentials for generated answers, tool calling, question drafting, and LLM-as-judge metrics. Without them, retrieval and deterministic offline answers still run.

From the repository root, create an environment and install the pinned dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS/Linux, use `python3 -m venv .venv` and `source .venv/bin/activate` instead. PyTorch selects CPU by default unless a CUDA-enabled installation is configured. The model downloads on first use and is cached under `data/cache/models/`.

## Watsonx setup

Copy `.env.example` to `.env` and replace the placeholders with your IBM watsonx.ai API key, service URL, and project ID. The model defaults to `ibm/granite-3-8b-instruct`; set `WATSONX_MODEL_ID` in `.env` to select another tool-calling capable chat model available to your project.

```powershell
Copy-Item .env.example .env
```

Keep `.env` private. It is excluded by `.gitignore`; only `.env.example` with placeholder values belongs in Git. When any required credential is absent, the pipeline uses an extractive `FakeLLM` that selects a query-relevant retrieved sentence. This fallback makes tests and retrieval evaluation work offline, but it does not represent watsonx answer quality or a real LLM judge.

## Download sample PDFs

The URL list contains 25 public arXiv PDF links. The downloader skips failed or invalid responses, keeps PDFs local and ignored by Git, and reports the actual downloaded file and page counts:

```powershell
python scripts/download_sample_pdfs.py
```

You can also put your own PDFs (with extractable text) in `data/pdfs/`. Scanned image-only PDFs need OCR before ingestion. Individual download failures are logged and do not stop the rest of the sample set.

## Build the index and ask questions

```powershell
python scripts/build_index.py
python scripts/ask.py "What does the report say about its main method?"
```

The answer command prints JSON with `answer`, `sources`, `latency_s`, and `tool_calls`. Source page numbers are 1-based. The FAISS index is saved under `data/index/` in a cache directory keyed by chunk size, overlap, and embedding model; PDF content and index files are not committed. `--force` rebuilds the configured index.

Configuration is in `config.yaml`: chunk size and overlap (measured in `cl100k_base` tokens), top-k, embedding model, batch size, model token limit, generation limit, and data paths. The package defaults `OMP_NUM_THREADS` to `4` before importing FAISS/PyTorch to avoid oversized native thread pools; set it in the environment before starting Python to tune this. On Windows it also enables `KMP_DUPLICATE_LIB_OK` because the PyTorch and FAISS wheels bundle different OpenMP runtimes; this project runs their native CPU operations sequentially. Embedding text longer than the model token limit is encoded in windows and pooled back to one vector so no chunk tail is silently dropped. Supported local embeddings:

- `sentence-transformers/all-MiniLM-L6-v2`
- `BAAI/bge-small-en-v1.5`
- `sentence-transformers/all-mpnet-base-v2`

The first run downloads model files and may take longer than later queries. The under-three-second average is a target to measure on your hardware and IBM region, not a guaranteed result.

## Optional Streamlit interface

```powershell
streamlit run app.py
```

Add PDFs and build/load the index from the sidebar, then ask questions in the chat.

## Benchmark workflow

`data/benchmark/questions.json` is the **verified benchmark input**. Each item must have exactly these core fields:

```json
{
  "id": "q001",
  "question": "What approach does the paper describe?",
  "ground_truth_answer": "A concise answer supported by the source page.",
  "source_file": "paper.pdf",
  "source_page": 3
}
```

Draft a target of 50 source-linked Q&A pairs with Watsonx:

```powershell
python scripts/generate_benchmark_draft.py --count 50
```

Every draft is labeled `manual_review_required`. Check the question and answer against the cited PDF page, correct errors, remove the review flag, and copy only verified entries into `data/benchmark/questions.json`. The experiment runner refuses to report metrics when that final file is empty. This avoids presenting generated draft answers as validated ground truth.

## Experiments and metrics

Run the baseline, one-variable-at-a-time comparisons, and then the combined grid:

```powershell
python scripts/run_experiments.py
```

The baseline is `chunk_size=1024`, `chunk_overlap=0`, `top_k=2`, `sentence-transformers/all-MiniLM-L6-v2`. The controlled values are chunk size `256, 512, 768, 1024`; overlap `0, 50, 100`; top-k `2, 3, 5, 8`; and the three embedding models listed above. A 144-point combined grid follows the controlled runs. `--no-combined-grid` runs only the baseline and one-factor set.

The runner records mean and standard deviation of correctness, mean/p50/p95 latency, and per-question outcomes. It reports two accuracy measures separately:

1. **LLM-as-judge**: Watsonx compares the response with the verified answer and returns a strict correct/incorrect verdict. This field is blank when Watsonx is unavailable or a verdict cannot be parsed.
2. **Retrieval heuristic**: the labeled file and page must appear in retrieved sources and at least 35% of the answer's ground-truth terms must overlap. This is a retrieval-oriented proxy, not a semantic judge.

Outputs are written from actual runs to `results/experiments.csv`, `results/per_question.csv`, `results/summary.md`, and three PNG plots (`accuracy_vs_chunk_size.png`, `accuracy_vs_top_k.png`, and `accuracy_vs_embedding_model.png`). Indexes are cached so changing only top-k does not re-embed the corpus.

## Results

Real smoke run: the downloader fetched **25 PDFs (908 pages)**, and the index build produced **1,258 chunks**. The sample question `What does the Longformer paper propose for processing long documents?` returned a cited extract from `longformer.pdf`, page 10. Warm retrieval plus the offline extractive `FakeLLM` took **0.090 seconds** for that single query on this machine.

This is one retrieval/fallback smoke query, not a 50-question benchmark or a Watsonx generation measurement. No accuracy, p95, or Watsonx latency result is recorded: Watsonx settings are present in the local `.env`, but the Python TLS connection failed during the IAM token request, before the credentials could be validated. The verified benchmark file is still empty. Run the benchmark workflow above after resolving the TLS connection to produce those results; scripts do not insert sample or estimated metrics.

## Tests

Tests use tiny local stubs for embeddings and do not need model downloads or Watsonx credentials:

```powershell
pytest -q
```

