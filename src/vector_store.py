"""
vector_store.py - FAISS vector store with sentence-transformer embeddings.
Stores chunks as vectors and retrieves top-k most similar on query.
"""

import numpy as np
from typing import Optional
from src.chunker import Chunk

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False


# Lightweight model — fast on CPU, good quality
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class VectorStore:
    """
    FAISS-backed vector store for semantic search over article chunks.
    Falls back to keyword search if FAISS/SentenceTransformers unavailable.
    """

    def __init__(self):
        self.chunks: list[Chunk] = []
        self.index: Optional[object] = None
        self.model: Optional[object] = None
        self._load_model()

    def _load_model(self):
        if ST_AVAILABLE:
            self.model = SentenceTransformer(EMBEDDING_MODEL)

    def _embed(self, texts: list[str]) -> np.ndarray:
        if self.model:
            embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.array(embeddings, dtype=np.float32)
        # Fallback: simple TF-IDF-style bag-of-words
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(max_features=512)
        return vec.fit_transform(texts).toarray().astype(np.float32)

    def build(self, chunks: list[Chunk]):
        """Build FAISS index from a list of chunks."""
        self.chunks = chunks
        texts = [c.text for c in chunks]
        embeddings = self._embed(texts)
        dim = embeddings.shape[1]

        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine on normalized)
            self.index.add(embeddings)
        else:
            # Store embeddings in-memory as numpy array
            self.index = embeddings

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """
        Semantic search: returns top_k chunks most relevant to query.
        Returns list of (chunk, score) tuples.
        """
        if not self.chunks:
            return []

        query_emb = self._embed([query])

        if FAISS_AVAILABLE and isinstance(self.index, faiss.Index):
            scores, indices = self.index.search(query_emb, min(top_k, len(self.chunks)))
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0:
                    results.append((self.chunks[idx], float(score)))
            return results
        else:
            # Numpy cosine similarity fallback
            scores = (self.index @ query_emb.T).squeeze()
            if scores.ndim == 0:
                scores = np.array([scores])
            top_indices = np.argsort(scores)[::-1][:top_k]
            return [(self.chunks[i], float(scores[i])) for i in top_indices]

    def is_ready(self) -> bool:
        return len(self.chunks) > 0 and self.index is not None

    def clear(self):
        self.chunks = []
        self.index = None
