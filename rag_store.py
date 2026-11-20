import os
from typing import List, Optional

import numpy as np

from models import Document

FAISS_AVAILABLE = False
try:
    import faiss  # type: ignore

    FAISS_AVAILABLE = True
except Exception:
    faiss = None

# all-MiniLM-L6-v2 was trained on English text and ranks Korean food names poorly.
DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    return vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)


class Embedder:
    """Loads the sentence-transformers model once and shares it across requests."""

    def __init__(self, model_name: Optional[str] = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODEL)
        self.model = SentenceTransformer(self.model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        return _normalize(self.model.encode(texts, show_progress_bar=False))


class RAGIndex:
    """Search index for a single request.

    Each /ask call builds its own index, so two requests running at the same
    time cannot overwrite each other's documents.
    """

    def __init__(self, embedder, docs: List[Document], use_faiss: bool = True):
        self.embedder = embedder
        self.docs = docs
        self.use_faiss = bool(use_faiss and FAISS_AVAILABLE and docs)
        self.index = None
        self.emb = embedder.encode([d.text for d in docs]) if docs else None

        if self.use_faiss:
            self.index = faiss.IndexFlatIP(self.emb.shape[1])
            self.index.add(self.emb)

    def search(self, query: str, top_k: int = 3) -> List[Document]:
        if self.emb is None or not self.docs:
            return []

        top_k = min(top_k, len(self.docs))
        q = self.embedder.encode([query])

        if self.index is not None:
            _, idx = self.index.search(q, top_k)
            order = idx[0]
        else:
            scores = self.emb @ q[0]
            order = np.argsort(-scores)[:top_k]

        return [self.docs[i] for i in order if 0 <= i < len(self.docs)]
