"""
Vector Store: FAISS-based vector database wrapper with BM25 hybrid search.
"""

import os
import pickle
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np

import faiss
from config.settings import settings


from loguru import logger


def _tokenize(text: str) -> List[str]:
    """Tokenize text using jieba for Chinese and whitespace for English."""
    try:
        import jieba
        tokens = list(jieba.cut(text))
    except ImportError:
        tokens = re.split(r'\s+', text)
    return [t.strip().lower() for t in tokens if t.strip() and len(t.strip()) > 1]


class VectorStore:
    """FAISS-based vector store with BM25 hybrid search support."""

    def __init__(self, embedding_dimension: int, store_path: str = "data/vector_store"):
        self.embedding_dimension = embedding_dimension
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)

        self.target_index_type = self._normalize_index_type(settings.FAISS_INDEX_TYPE)
        self.index = self._create_index("flat", vector_count=0)

        # Metadata storage
        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.vectors = np.empty((0, self.embedding_dimension), dtype="float32")

        # BM25 index
        self.bm25 = None
        self.bm25_tokenized: List[List[str]] = []

        # Load existing index if available
        self._load()
        self._apply_index_runtime_params()
        self._maybe_upgrade_index()

    @staticmethod
    def _normalize_index_type(index_type: str) -> str:
        idx = (index_type or "flat").strip().lower()
        return idx if idx in {"flat", "ivf", "hnsw"} else "flat"

    def _current_index_type(self) -> str:
        name = type(self.index).__name__.lower()
        if "ivf" in name:
            return "ivf"
        if "hnsw" in name:
            return "hnsw"
        return "flat"

    def _create_index(self, index_type: str, vector_count: int) -> faiss.Index:
        idx = self._normalize_index_type(index_type)
        if idx == "ivf":
            nlist = max(1, min(int(settings.FAISS_IVF_NLIST), max(1, vector_count)))
            quantizer = faiss.IndexFlatIP(self.embedding_dimension)
            return faiss.IndexIVFFlat(
                quantizer,
                self.embedding_dimension,
                nlist,
                faiss.METRIC_INNER_PRODUCT,
            )
        if idx == "hnsw":
            m = max(4, int(settings.FAISS_HNSW_M))
            return faiss.IndexHNSWFlat(self.embedding_dimension, m, faiss.METRIC_INNER_PRODUCT)
        return faiss.IndexFlatIP(self.embedding_dimension)

    def _apply_index_runtime_params(self):
        idx_type = self._current_index_type()
        if idx_type == "ivf":
            nprobe = max(1, int(settings.FAISS_IVF_NPROBE))
            if hasattr(self.index, "nlist"):
                nprobe = min(nprobe, int(self.index.nlist))
            self.index.nprobe = nprobe
        elif idx_type == "hnsw":
            if hasattr(self.index, "hnsw"):
                self.index.hnsw.efSearch = max(8, int(settings.FAISS_HNSW_EF_SEARCH))
                self.index.hnsw.efConstruction = max(16, int(settings.FAISS_HNSW_EF_CONSTRUCTION))

    def _add_vectors_to_index(self, vectors: np.ndarray):
        if vectors.size == 0:
            return
        batch_size = max(1, int(settings.FAISS_ADD_BATCH_SIZE))
        for start in range(0, vectors.shape[0], batch_size):
            end = start + batch_size
            self.index.add(vectors[start:end])

    def _rebuild_index_from_vectors(self, vectors: np.ndarray, index_type: Optional[str] = None):
        target = self._normalize_index_type(index_type or self._current_index_type())
        rebuilt = self._create_index(target, vector_count=vectors.shape[0])
        try:
            if target == "ivf" and vectors.shape[0] > 0:
                rebuilt.train(vectors)
            self.index = rebuilt
            self._apply_index_runtime_params()
            self._add_vectors_to_index(vectors)
        except Exception as e:
            if target != "flat":
                logger.warning(f"Failed to build {target} index ({e}), falling back to flat")
                self.index = self._create_index("flat", vector_count=vectors.shape[0])
                self._add_vectors_to_index(vectors)
            else:
                raise

    def _maybe_upgrade_index(self):
        if self.target_index_type == "flat":
            return
        if self._current_index_type() == self.target_index_type:
            return
        min_vectors = max(1, int(settings.FAISS_UPGRADE_MIN_VECTORS))
        if self.vectors.shape[0] < min_vectors:
            return
        logger.info(
            f"Upgrading FAISS index: {self._current_index_type()} -> "
            f"{self.target_index_type} ({self.vectors.shape[0]} vectors)"
        )
        self._rebuild_index_from_vectors(self.vectors, index_type=self.target_index_type)

    def _rebuild_bm25(self):
        """Rebuild BM25 index from tokenized documents."""
        if not self.bm25_tokenized:
            self.bm25 = None
            return
        try:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi(self.bm25_tokenized)
            logger.info(f"BM25 index rebuilt with {len(self.bm25_tokenized)} documents")
        except ImportError:
            logger.warning("rank_bm25 not installed. BM25 search disabled.")
            self.bm25 = None

    def add_embeddings(
        self,
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]]
    ):
        if not embeddings:
            logger.warning("No embeddings to add")
            return

        if len(embeddings) != len(documents) or len(documents) != len(metadatas):
            raise ValueError("embeddings, documents, metadatas must have the same length")

        logger.info(f"Adding {len(embeddings)} embeddings to vector store")

        embeddings_array = np.array(embeddings, dtype="float32", copy=False)
        if embeddings_array.ndim != 2 or embeddings_array.shape[1] != self.embedding_dimension:
            raise ValueError(
                f"Invalid embedding shape {embeddings_array.shape}, "
                f"expected (*, {self.embedding_dimension})"
            )

        faiss.normalize_L2(embeddings_array)
        new_vectors = (
            embeddings_array
            if self.vectors.size == 0
            else np.vstack([self.vectors, embeddings_array])
        )

        try:
            self._add_vectors_to_index(embeddings_array)
            self.vectors = new_vectors
        except Exception as e:
            logger.warning(f"Incremental add failed ({e}), rebuilding index")
            self.vectors = new_vectors
            self._rebuild_index_from_vectors(self.vectors)

        self.documents.extend(documents)
        self.metadatas.extend(metadatas)

        # Update BM25 tokenized corpus
        self.bm25_tokenized.extend(_tokenize(doc) for doc in documents)
        self._rebuild_bm25()
        self._maybe_upgrade_index()

        logger.info(f"Total embeddings in store: {self.index.ntotal}")

    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar documents using FAISS vector search."""
        if self.index.ntotal == 0:
            logger.warning("Vector store is empty")
            return []

        query_array = np.array([query_embedding], dtype="float32")
        faiss.normalize_L2(query_array)
        scores, indices = self.index.search(query_array, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append({
                    "score": float(score),
                    "document": self.documents[idx],
                    "metadata": self.metadatas[idx]
                })
        return results

    def search_bm25(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search using BM25 text matching."""
        if not self.bm25 or not self.documents:
            return []

        tokenized_query = _tokenize(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "score": float(scores[idx]),
                    "document": self.documents[idx],
                    "metadata": self.metadatas[idx]
                })
        return results

    def search_hybrid(self, query: str, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Hybrid search combining vector similarity and BM25 text matching.
        Scores are min-max normalized then weighted.
        """
        vector_weight = settings.VECTOR_WEIGHT
        bm25_weight = settings.BM25_WEIGHT
        candidates = max(top_k * 3, settings.RERANKER_CANDIDATES)

        # Vector search
        vector_results = self.search(query_embedding, top_k=candidates)

        # BM25 search
        bm25_results = self.search_bm25(query, top_k=candidates)

        if not vector_results and not bm25_results:
            return []

        # Build score maps keyed by stable document identity.
        def doc_key(result: Dict[str, Any]) -> str:
            meta = result.get("metadata", {}) or {}
            chunk_id = meta.get("chunk_id")
            if chunk_id:
                return f"{meta.get('source', '')}|cid:{chunk_id}"
            return "|".join([
                str(meta.get("source", "")),
                str(meta.get("chunk_index", "")),
                str(meta.get("page_number", "")),
                result.get("document", "")[:80],
            ])

        vector_scores = {}
        for r in vector_results:
            k = doc_key(r)
            vector_scores[k] = r["score"]

        bm25_scores = {}
        for r in bm25_results:
            k = doc_key(r)
            bm25_scores[k] = r["score"]

        # Min-max normalize
        def normalize(scores_dict):
            if not scores_dict:
                return scores_dict
            vals = list(scores_dict.values())
            mn, mx = min(vals), max(vals)
            if mx - mn < 1e-9:
                return {k: 1.0 for k in scores_dict}
            return {k: (v - mn) / (mx - mn) for k, v in scores_dict.items()}

        vector_norm = normalize(vector_scores)
        bm25_norm = normalize(bm25_scores)

        # Merge all candidates
        all_docs = {}
        for r in vector_results + bm25_results:
            k = doc_key(r)
            if k not in all_docs:
                all_docs[k] = r

        # Compute fused scores
        fused = []
        for k, r in all_docs.items():
            vs = vector_norm.get(k, 0.0) * vector_weight
            bs = bm25_norm.get(k, 0.0) * bm25_weight
            r_copy = r.copy()
            r_copy["score"] = vs + bs
            fused.append(r_copy)

        fused.sort(key=lambda x: x["score"], reverse=True)
        logger.info(f"Hybrid search: {len(fused)} candidates, returning top {top_k}")
        return fused[:top_k]

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get a chunk by its chunk_id from metadata."""
        for i, meta in enumerate(self.metadatas):
            if meta.get("chunk_id") == chunk_id:
                return {
                    "document": self.documents[i],
                    "metadata": meta
                }
        return None

    def search_with_parents(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search child chunks and return parent context when available."""
        results = self.search(query_embedding, top_k=top_k)
        enriched = []
        seen_parents = set()

        for r in results:
            parent_id = r["metadata"].get("parent_id")
            if parent_id and parent_id not in seen_parents:
                parent = self.get_chunk_by_id(parent_id)
                if parent:
                    seen_parents.add(parent_id)
                    r["parent_document"] = parent["document"]
                    r["parent_metadata"] = parent["metadata"]
            enriched.append(r)
        return enriched

    def remove_by_source(self, source: str):
        """Remove all documents matching a source, rebuilding the FAISS index."""
        keep_indices = [
            i for i, m in enumerate(self.metadatas)
            if m.get("source") != source
        ]

        if len(keep_indices) == len(self.documents):
            logger.info(f"No documents found with source: {source}")
            return

        removed = len(self.documents) - len(keep_indices)
        logger.info(f"Removing {removed} documents with source: {source}")

        current_index_type = self._current_index_type()
        new_docs = [self.documents[i] for i in keep_indices]
        new_metas = [self.metadatas[i] for i in keep_indices]
        new_tokens = [self.bm25_tokenized[i] for i in keep_indices] if self.bm25_tokenized else []
        new_vectors = (
            self.vectors[keep_indices]
            if keep_indices and self.vectors.shape[0] >= len(self.documents)
            else np.empty((0, self.embedding_dimension), dtype="float32")
        )

        self._rebuild_index_from_vectors(new_vectors, index_type=current_index_type)

        self.documents = new_docs
        self.metadatas = new_metas
        self.bm25_tokenized = new_tokens
        self.vectors = new_vectors
        self._rebuild_bm25()
        self._maybe_upgrade_index()

        logger.info(f"Rebuild complete. {self.index.ntotal} documents remaining.")

    @staticmethod
    def _atomic_pickle_dump(path: Path, data: Any):
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(data, f)
        os.replace(tmp, path)

    @staticmethod
    def _atomic_npy_dump(path: Path, data: np.ndarray):
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:
            np.save(f, data)
        os.replace(tmp, path)

    def save(self):
        """Persist the vector store to disk."""
        logger.info(f"Saving vector store to {self.store_path}")

        index_path = self.store_path / "index.faiss"
        index_tmp = index_path.with_name(index_path.name + ".tmp")
        faiss.write_index(self.index, str(index_tmp))
        os.replace(index_tmp, index_path)

        self._atomic_pickle_dump(self.store_path / "documents.pkl", self.documents)
        self._atomic_pickle_dump(self.store_path / "metadatas.pkl", self.metadatas)
        self._atomic_pickle_dump(self.store_path / "bm25_tokenized.pkl", self.bm25_tokenized)
        self._atomic_npy_dump(self.store_path / "vectors.npy", self.vectors)

        logger.info("Vector store saved successfully")

    def _load(self):
        """Load vector store from disk if available."""
        index_path = self.store_path / "index.faiss"
        docs_path = self.store_path / "documents.pkl"
        meta_path = self.store_path / "metadatas.pkl"
        bm25_path = self.store_path / "bm25_tokenized.pkl"
        vectors_path = self.store_path / "vectors.npy"

        if index_path.exists() and docs_path.exists() and meta_path.exists():
            logger.info("Loading existing vector store")
            self.index = faiss.read_index(str(index_path))

            with open(docs_path, "rb") as f:
                self.documents = pickle.load(f)

            with open(meta_path, "rb") as f:
                self.metadatas = pickle.load(f)

            # Load BM25 tokenized corpus
            if bm25_path.exists():
                with open(bm25_path, "rb") as f:
                    self.bm25_tokenized = pickle.load(f)
            else:
                # Rebuild from documents
                self.bm25_tokenized = [_tokenize(doc) for doc in self.documents]

            # Load vectors cache for faster rebuild/updates
            if vectors_path.exists():
                self.vectors = np.load(vectors_path).astype("float32", copy=False)
            elif self.index.ntotal > 0:
                # Backward compatibility for existing stores without vectors.npy
                self.vectors = np.array(
                    [self.index.reconstruct(i) for i in range(self.index.ntotal)],
                    dtype="float32",
                )
            else:
                self.vectors = np.empty((0, self.embedding_dimension), dtype="float32")

            counts = [
                len(self.documents),
                len(self.metadatas),
                len(self.bm25_tokenized),
                self.vectors.shape[0],
                self.index.ntotal,
            ]
            if len(set(counts)) != 1:
                min_count = min(counts)
                logger.warning(
                    "Vector store metadata/vector count mismatch "
                    f"{counts}, trimming to {min_count} and rebuilding index"
                )
                self.documents = self.documents[:min_count]
                self.metadatas = self.metadatas[:min_count]
                self.bm25_tokenized = self.bm25_tokenized[:min_count]
                self.vectors = self.vectors[:min_count]
                self._rebuild_index_from_vectors(self.vectors, index_type=self._current_index_type())

            self._rebuild_bm25()
            self._apply_index_runtime_params()
            logger.info(f"Loaded vector store with {self.index.ntotal} embeddings")
        else:
            logger.info("No existing vector store found, starting fresh")
