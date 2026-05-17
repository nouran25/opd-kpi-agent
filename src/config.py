"""Configuration management for OPD KPI Agent"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class Config:
    """Agent configuration"""

    # Model settings
    llm_model: str = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    temperature: float = float(os.getenv("TEMPERATURE", "0.0"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))
    llm_timeout: int = int(os.getenv("LLM_TIMEOUT", "60"))
    llm_max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "2"))
    llm_reasoning_effort: str = os.getenv("LLM_REASONING_EFFORT", "medium")

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
    server_host: str = os.getenv("SERVER_HOST", "127.0.0.1")
    server_port: int = 7860

    def ensure_directories(self):
        """Create necessary directories"""
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_store_path.mkdir(parents=True, exist_ok=True)


# Singleton config instance
config = Config()
