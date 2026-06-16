from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from models import Document

FAISS_AVAILABLE = False
try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except Exception:
    faiss = None

class RAGStore:
    def __init__(self, embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embed_model)
        self.docs: List[Document] = []
        self.use_faiss: bool = False

        self.emb: Optional[np.ndarray] = None  # (N,D) normalized
        self.index = None

    def build(self, docs: List[Document], use_faiss: bool = True) -> int:
        self.docs = docs
        texts = [d.text for d in docs]
        vec = self.model.encode(texts, show_progress_bar=False)
        vec = np.array(vec, dtype=np.float32)

        # 코사인용 정규화
        vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)
        self.emb = vec

        self.use_faiss = bool(use_faiss and FAISS_AVAILABLE)
        self.index = None

        if self.use_faiss:
            dim = vec.shape[1]
            self.index = faiss.IndexFlatIP(dim)  # 코사인(정규화) => inner product
            self.index.add(vec)

        return len(docs)

    def search(self, query: str, top_k: int = 3) -> List[Document]:
        if self.emb is None or not self.docs:
            return []

        q = self.model.encode([query])
        q = np.array(q, dtype=np.float32)
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)

        if self.use_faiss and self.index is not None:
            scores, idx = self.index.search(q, top_k)
            I = idx[0]
        else:
            scores = (self.emb @ q[0]).reshape(-1)
            I = np.argsort(-scores)[:top_k]

        return [self.docs[i] for i in I if 0 <= i < len(self.docs)]
