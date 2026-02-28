"""
Embedding Service: local sentence-transformers or cloud API embeddings.
"""

from typing import List, Any

from config.settings import settings
from loguru import logger


class EmbeddingService:
    """Service for generating text embeddings."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.backend = (settings.EMBEDDING_BACKEND or "local").strip().lower()
        self.provider = (
            settings.EMBEDDING_PROVIDER
            or settings.LLM_PROVIDER
            or "zhipuai"
        ).strip().lower()
        self.model_name = model_name
        self.model = None
        self.client = None

        if self.backend == "api":
            self._init_api_client()
            self.dimension = self._resolve_api_dimension()
            logger.info(
                f"Embedding API ready: provider={self.provider}, "
                f"model={self.model_name}, dimension={self.dimension}"
            )
        else:
            self._init_local_model()

    def _init_local_model(self):
        # Delayed import to avoid loading torch/transformers in API backend mode.
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading local embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Local embedding model loaded (dimension: {self.dimension})")

    def _init_api_client(self):
        if self.provider == "openai":
            from openai import OpenAI

            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not configured for embedding API")
            self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
            return

        # Default provider: zhipuai
        from zhipuai import ZhipuAI

        self.provider = "zhipuai"
        if not settings.ZHIPU_API_KEY:
            raise ValueError("ZHIPU_API_KEY not configured for embedding API")
        self.client = ZhipuAI(api_key=settings.ZHIPU_API_KEY)

    def _resolve_api_dimension(self) -> int:
        configured = int(settings.EMBEDDING_DIMENSION or 0)
        if configured > 0:
            return configured

        # Known dimensions to avoid startup-time probe requests.
        known_dims = {
            ("openai", "text-embedding-3-small"): 1536,
            ("openai", "text-embedding-3-large"): 3072,
            ("openai", "text-embedding-ada-002"): 1536,
            ("zhipuai", "embedding-2"): 1024,
            ("zhipuai", "embedding-3"): 1024,
        }
        key = (self.provider, (self.model_name or "").strip().lower())
        if key in known_dims:
            return known_dims[key]

        # Last-resort probe for unknown models.
        probe_vector = self._embed_texts_api(["dimension probe"])
        if not probe_vector or not probe_vector[0]:
            raise ValueError("Failed to probe embedding dimension from API")
        return len(probe_vector[0])

    @staticmethod
    def _extract_embedding_data(response: Any) -> List[List[float]]:
        data = getattr(response, "data", None)
        if data is None and isinstance(response, dict):
            data = response.get("data")
        if not data:
            return []

        vectors: List[List[float]] = []
        for item in data:
            vec = getattr(item, "embedding", None)
            if vec is None and isinstance(item, dict):
                vec = item.get("embedding")
            if vec:
                vectors.append(list(vec))
        return vectors

    def _embed_texts_api(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts if len(texts) > 1 else texts[0],
        )
        vectors = self._extract_embedding_data(response)
        if len(vectors) != len(texts):
            raise ValueError(
                f"Embedding API returned {len(vectors)} vectors for {len(texts)} inputs"
            )
        return vectors

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query."""
        logger.debug(f"Embedding query: {query[:100]}...")
        if self.backend == "api":
            return self._embed_texts_api([query])[0]

        embedding = self.model.encode(query, convert_to_numpy=True, show_progress_bar=False)
        return embedding.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple documents."""
        if not texts:
            return []

        logger.debug(f"Embedding {len(texts)} documents")
        if self.backend == "api":
            batch_size = max(1, int(settings.EMBEDDING_API_BATCH_SIZE))
            vectors: List[List[float]] = []
            for i in range(0, len(texts), batch_size):
                vectors.extend(self._embed_texts_api(texts[i:i + batch_size]))
            return vectors

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 10,
        )
        return embeddings.tolist()
