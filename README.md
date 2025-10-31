# TicketBuddy RAG Pipeline

An end-to-end Retrieval-Augmented Generation (RAG) system that understands IT issue queries, retrieves relevant context from past ticket data and knowledge base text files, and generates structured, evidence-backed JSON responses.

## Data Sources
- `Assignment_docs/IT_tickets_data.xlsx` — columns: `ticket_id`, `title`, `description`, `root_cause`, `resolution`
- `Assignment_docs/KB_docs/*.txt` — 10 text files with common IT problems and solutions
- `Assignment_docs/user_queries.xlsx` — column: `user_query`

## Features
- Hybrid Retrieval: BM25 (sparse) + Dense (Sentence-Transformers) with score fusion
- Re-ranking: Cross-encoder (bge-reranker) to pick the best contexts
- Context Optimization: MMR diversity + redundancy filtering
- JSON Output: `root_causes`, `resolution_steps`, `sources`
- Edge Cases: Multi-intent query splitting, fallback synthesis when LLM is unavailable

## Quickstart

1) Create and activate a virtual environment (optional but recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies:

```powershell
pip install -r requirements.txt
```

3) (Optional) Use OpenAI for generation. Set your key:

```powershell
$env:OPENAI_API_KEY = "sk-..."
```

4) Run the pipeline:

```powershell
python ticketbuddy_rag_pipeline.py
```

This will create `results.json` at the project root with structured outputs for all user queries.

## Configuration
Environment variables to tweak models:
- `TICKETBUDDY_EMBEDDINGS` (default: `BAAI/bge-small-en-v1.5`)
- `TICKETBUDDY_RERANKER` (default: `BAAI/bge-reranker-base`)
- `TICKETBUDDY_LLM` (default: `gpt-4o-mini`)

To enable PCA dimensionality reduction (e.g., 768 -> 384), set:
- `TICKETBUDDY_USE_PCA_DIM` (not wired as env directly; edit `config.py` or extend as needed)

## Architecture
- `ticketbuddy/config.py` — paths and knobs
- `ticketbuddy/data_loader.py` — reads Excel and KB text files
- `ticketbuddy/preprocess.py` — cleans and chunks documents (RecursiveCharacterTextSplitter fallback)
- `ticketbuddy/embeddings.py` — sentence-transformers encoder + optional PCA
- `ticketbuddy/vectorstore.py` — FAISS HNSW index with numpy fallback
- `ticketbuddy/retriever.py` — BM25 + dense fusion
- `ticketbuddy/reranker.py` — cross-encoder reranking
- `ticketbuddy/compressor.py` — MMR selection and redundancy filter
- `ticketbuddy/generator.py` — OpenAI (if available) or deterministic fallback
- `ticketbuddy/pipeline.py` — orchestration end-to-end
- `ticketbuddy_rag_pipeline.py` — CLI entry point

## Notes
- Models like `bge-small-en-v1.5` and `bge-reranker-base` are smaller and faster; you can switch to larger variants for higher quality at higher cost.
- If FAISS is not available, the system falls back to a numpy-based exact search.
- If OpenAI is not configured, the generator uses a heuristic fallback that still returns valid JSON using evidence extracted from tickets/KB.

## Troubleshooting
- Excel reading errors: ensure files exist at `Assignment_docs/` and `openpyxl` is installed.
- CUDA/torch warnings: CPU inference is fine; defaults don’t require GPU.
- Memory: If RAM is limited, switch to smaller models and reduce `chunk_size` in `config.py`.
