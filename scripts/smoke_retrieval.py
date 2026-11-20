"""Load the real embedding model and check that Korean queries rank the right food first."""
from rag_store import FAISS_AVAILABLE, Embedder, RAGIndex
from utils import food_item_to_document

FOODS = ["신라면", "두부", "현미밥", "우유", "김치찌개"]
CASES = {
    "신라면 나트륨 알려줘": "신라면",
    "두부 단백질": "두부",
    "우유 칼슘 함량": "우유",
}

docs = [food_item_to_document({"foodNm": name, "foodCd": str(i)}, i) for i, name in enumerate(FOODS)]
embedder = Embedder()
print("model:", embedder.model_name, "| faiss:", FAISS_AVAILABLE)

for use_faiss in (True, False):
    index = RAGIndex(embedder, docs, use_faiss=use_faiss)
    for query, expected in CASES.items():
        top = index.search(query, top_k=1)[0].text.splitlines()[0]
        print(f"[{'faiss' if index.use_faiss else 'numpy'}] {query} -> {top}")
        assert expected in top, f"{query!r} ranked {top!r} first"

print("ok")
