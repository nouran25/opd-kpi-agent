"""Configuration management for OPD KPI Agent"""

import os
from dataclasses import dataclass
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class Config:
    """Agent configuration"""

    # Model settings
    llm_model: str = os.getenv("LLM_MODEL", "qwen2.5:7b")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    temperature: float = float(os.getenv("TEMPERATURE", "0.1"))
    llm_num_ctx: int = int(os.getenv("LLM_NUM_CTX", "2048"))
    llm_num_predict: int = int(os.getenv("LLM_NUM_PREDICT", "512"))
    llm_num_thread: int = int(os.getenv("LLM_NUM_THREAD", str(os.cpu_count() or 4)))
    llm_num_gpu: int = int(os.getenv("LLM_NUM_GPU", "0"))
    llm_keep_alive: str = os.getenv("LLM_KEEP_ALIVE", "30m")

    # Ollama settings
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Data paths
    dataset_path: Path = PROJECT_ROOT / "data" / "OPD dataset.xlsx"
    knowledge_path: Path = PROJECT_ROOT / "data" / "Knowledge base.xlsx"

    # Database paths
    db_path: Path = PROJECT_ROOT / "data" / "opd_analytics.db"
    vector_store_path: Path = PROJECT_ROOT / "data" / "chroma_db"

    # Analysis thresholds
    revenue_leakage_threshold: float = 0.10
    no_show_threshold: float = 0.25
    retention_threshold: float = 0.60

    # Web server
    server_host: str = "0.0.0.0"
    server_port: int = 7860

    def ensure_directories(self):
        """Create necessary directories"""
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_store_path.mkdir(parents=True, exist_ok=True)


# Singleton config instance
config = Config()
