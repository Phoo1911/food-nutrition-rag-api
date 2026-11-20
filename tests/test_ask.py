import numpy as np
from fastapi.testclient import TestClient

import main
from rag_store import RAGIndex
from utils import food_item_to_document


class KeywordEmbedder:
    """Tiny stand-in for sentence-transformers: one dimension per vocabulary word."""

    vocab = ["신라면", "진라면", "두부", "나트륨"]

    def encode(self, texts):
        vectors = np.array([[text.count(word) for word in self.vocab] for text in texts], dtype=np.float32)
        return vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)


class StubLLM:
    api_key = ""

    def extract_keywords(self, question, max_k=5):
        return ["라면"]

    def answer_with_context(self, question, keywords, context):
        return "context:" + context


ITEMS = [
    {"foodNm": "신라면", "foodCd": "1", "enerc": "500", "nat": "1790"},
    {"foodNm": "진라면", "foodCd": "2", "enerc": "480", "nat": "1640"},
    {"foodNm": "신라면", "foodCd": "1", "enerc": "500", "nat": "1790"},
]


def make_client(monkeypatch, call_public_api):
    monkeypatch.setattr(main, "get_llm_client", lambda: StubLLM())
    monkeypatch.setattr(main, "get_embedder", lambda: KeywordEmbedder())
    monkeypatch.setattr(main, "call_public_api", call_public_api)
    return TestClient(main.app)


def test_ask_deduplicates_and_returns_best_match(monkeypatch):
    client = make_client(monkeypatch, lambda **kw: {"response": {"body": {"items": {"item": ITEMS}}}})

    response = client.post(
        "/ask",
        json={"question": "신라면 나트륨", "apis": ["food"], "top_k": 1, "use_faiss": False, "serviceKey": "k"},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["built_docs"] == 2
    assert body["retrieved_count"] == 1
    assert "식품명: 신라면" in body["results"][0]["text"]
    assert "1790" in body["answer"]
    assert body["errors"] == []


def test_ask_reports_public_api_failures(monkeypatch):
    def failing_api(**kw):
        raise TimeoutError("read timed out")

    client = make_client(monkeypatch, failing_api)

    body = client.post("/ask", json={"question": "신라면", "apis": ["food", "nutri"], "serviceKey": "k"}).json()

    assert body["retrieved_count"] == 0
    assert len(body["errors"]) == 2
    assert body["errors"][0].startswith("food(라면): TimeoutError")


def test_ask_requires_service_key(monkeypatch):
    monkeypatch.setattr(main, "SERVICE_KEY_ENV", "")
    client = make_client(monkeypatch, lambda **kw: {})

    response = client.post("/ask", json={"question": "신라면", "serviceKey": "string"})

    assert response.status_code == 400


def test_each_index_keeps_its_own_documents():
    embedder = KeywordEmbedder()
    first = RAGIndex(embedder, [food_item_to_document(ITEMS[0], 0)], use_faiss=False)
    second = RAGIndex(embedder, [food_item_to_document({"foodNm": "두부", "foodCd": "9"}, 0)], use_faiss=False)

    assert "신라면" in first.search("신라면", top_k=5)[0].text
    assert "두부" in second.search("신라면", top_k=5)[0].text
