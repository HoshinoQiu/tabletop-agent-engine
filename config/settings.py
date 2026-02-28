from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    """Application settings with validation."""

    # RAG Configuration
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    # 模型选择：
    # - 中文文档: "shibing624/text2vec-base-chinese"
    # - 英文文档: "sentence-transformers/all-MiniLM-L6-v2"
    # - 中英混合: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" (推荐)
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # Embedding backend:
    # - local: load sentence-transformers locally (higher memory usage)
    # - api: call cloud embedding API (Render free recommended)
    EMBEDDING_BACKEND: str = "local"  # local | api
    # Embedding provider when EMBEDDING_BACKEND=api
    EMBEDDING_PROVIDER: str = "zhipuai"  # zhipuai | openai
    EMBEDDING_DIMENSION: int = 0  # 0 = auto detect via probe
    EMBEDDING_API_BATCH_SIZE: int = 16
    VECTOR_STORE_PATH: str = "data/vector_store"
    FAISS_INDEX_TYPE: str = "flat"  # flat | ivf | hnsw
    FAISS_UPGRADE_MIN_VECTORS: int = 2000
    FAISS_ADD_BATCH_SIZE: int = 2048
    FAISS_IVF_NLIST: int = 128
    FAISS_IVF_NPROBE: int = 16
    FAISS_HNSW_M: int = 32
    FAISS_HNSW_EF_SEARCH: int = 64
    FAISS_HNSW_EF_CONSTRUCTION: int = 80

    # Agent Configuration
    MAX_REACT_ITERATIONS: int = 3  # Keep low to avoid looping
    TOP_K_RESULTS: int = 5
    MIN_SIMILARITY_SCORE: float = 0.3  # Below this = "not found"

    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = True

    # LLM Configuration
    # LLM_PROVIDER: zhipuai | openai
    LLM_PROVIDER: str = "zhipuai"
    # API密钥应通过环境变量设置
    ZHIPU_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    LLM_MODEL: str = "glm-4-flash"
    LLM_TEMPERATURE: float = 0.7

    # Logging
    LOG_LEVEL: str = "INFO"

    # Hybrid Search (BM25 + Vector)
    HYBRID_SEARCH_ENABLED: bool = True
    VECTOR_WEIGHT: float = 0.65
    BM25_WEIGHT: float = 0.35

    # Reranker (cross-encoder second-stage ranking)
    RERANKER_ENABLED: bool = True
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    RERANKER_CANDIDATES: int = 8

    # Chunking (simple strategy - structure_aware causes issues with PDF text)
    CHUNKING_STRATEGY: str = "simple"
    CHILD_CHUNK_SIZE: int = 200
    PARENT_CHUNK_SIZE: int = 1000
    ENABLE_PARENT_CHILD: bool = False

    # Session
    SESSION_TTL_SECONDS: int = 3600
    MAX_HISTORY_TURNS: int = 10

    # File Upload
    MAX_UPLOAD_SIZE_MB: int = 50
    ALLOWED_FILE_TYPES: list = [".pdf", ".txt", ".md", ".markdown"]

    # Cache (disabled - can return stale/wrong results)
    CACHE_ENABLED: bool = False
    CACHE_MAX_SIZE: int = 200
    CACHE_SIMILARITY_THRESHOLD: float = 0.95

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


settings = Settings()
