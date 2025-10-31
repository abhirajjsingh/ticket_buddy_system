from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Assignment_docs"
KB_DIR = DATA_DIR / "KB_docs"
IT_TICKETS_XLSX = DATA_DIR / "IT_tickets_data.xlsx"
USER_QUERIES_XLSX = DATA_DIR / "user_queries.xlsx"
RESULTS_JSON = PROJECT_ROOT / "results.json"


@dataclass
class ModelConfig:
    embedding_model_name: str = os.environ.get("TICKETBUDDY_EMBEDDINGS", "BAAI/bge-small-en-v1.5")
    reranker_model_name: str = os.environ.get("TICKETBUDDY_RERANKER", "BAAI/bge-reranker-base")
    use_pca_dim: Optional[int] = int(os.environ.get("TICKETBUDDY_USE_PCA_DIM", "0") or "0")  # 0/None disables PCA
    openai_model: str = os.environ.get("TICKETBUDDY_LLM", "gpt-4o-mini")
    temperature: float = float(os.environ.get("TICKETBUDDY_TEMPERATURE", "0.3"))


@dataclass
class RetrievalConfig:
    top_k_dense: int = 12
    top_k_bm25: int = 12
    fuse_alpha: float = 0.6  # weight for dense vs. sparse when fusing
    final_k: int = 8         # after fusion
    mmr_k: int = 6
    mmr_lambda: float = 0.5
    rerank_top_k: int = 3
    similarity_threshold: float = 0.88  # redundancy filter


@dataclass
class ChunkingConfig:
    chunk_size: int = 800
    chunk_overlap: int = 120


@dataclass
class Paths:
    project_root: Path = PROJECT_ROOT
    data_dir: Path = DATA_DIR
    kb_dir: Path = KB_DIR
    it_tickets_xlsx: Path = IT_TICKETS_XLSX
    user_queries_xlsx: Path = USER_QUERIES_XLSX
    results_json: Path = RESULTS_JSON


model_config = ModelConfig()
retrieval_config = RetrievalConfig()
chunking_config = ChunkingConfig()
paths = Paths()
