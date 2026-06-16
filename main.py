from functools import lru_cache
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from api_client import call_public_api, extract_items
from config import API_DEFS, SERVICE_KEY_ENV
from llm_client import LLMClient
from models import AskRequest
from rag_store import FAISS_AVAILABLE, RAGStore
from utils import dedup_items, food_item_to_document, make_llm_context

load_dotenv()

APP_TITLE = "Food Nutrition RAG API"
APP_VERSION = "2.1.0"
APP_DESCRIPTION = (
    "Korean public food and nutrition data search API with retrieval-augmented "
    "generation support."
)

def get_service_key(provided_key: Optional[str]) -> str:
    key = (provided_key or "").strip()

    if key.lower() in {"", "string", "서비스키", "여기에_본인_서비스키"}:
        key = ""

    if not key:
        key = (SERVICE_KEY_ENV or "").strip()

    if not key:
        raise HTTPException(
            status_code=400,
            detail=(
                "서비스키가 없습니다. 환경변수 DATA_GO_KR_SERVICE_KEY를 설정하거나 "
                "요청 본문에 serviceKey 값을 넣어주세요."
            ),
        )
    return key


@lru_cache
def get_rag_store() -> RAGStore:
    return RAGStore()


@lru_cache
def get_llm_client() -> LLMClient:
    return LLMClient()


def create_app() -> FastAPI:
    app = FastAPI(
        title=APP_TITLE,
        version=APP_VERSION,
        description=APP_DESCRIPTION,
    )

    @app.get("/")
    def root() -> Dict[str, Any]:
        return {
            "name": APP_TITLE,
            "version": APP_VERSION,
            "faiss_available": FAISS_AVAILABLE,
            "supported_apis": list(API_DEFS.keys()),
            "docs_url": "/docs",
        }

    @app.get("/health")
    def health() -> Dict[str, Any]:
        llm = get_llm_client()
        return {
            "status": "ok",
            "faiss_available": FAISS_AVAILABLE,
            "configured_api_count": len(API_DEFS),
            "gemini_configured": bool(llm.api_key),
        }

    @app.post("/ask")
    def ask(req: AskRequest) -> Dict[str, Any]:
        rag_store = get_rag_store()
        llm = get_llm_client()
        service_key = get_service_key(req.serviceKey)

        if req.resp_type != "json":
            raise HTTPException(
                status_code=400,
                detail="현재 /ask 엔드포인트는 resp_type=json만 지원합니다.",
            )

        try:
            keywords = llm.extract_keywords(req.question, max_k=5) or []
        except Exception:
            keywords = []

        if not keywords:
            keywords = [req.question.split()[0]] if req.question.strip() else [req.question]

        all_items: List[Dict[str, Any]] = []
        errors: List[str] = []

        for keyword in keywords:
            for api_name in req.apis:
                try:
                    data = call_public_api(
                        api_name=api_name,
                        service_key=service_key,
                        keyword=keyword,
                        pageNo=1,
                        numOfRows=req.top_rows,
                        resp_type=req.resp_type,
                    )
                    items = extract_items(data, req.resp_type)
                    all_items.extend(items)
                except Exception as exc:
                    errors.append(f"{api_name}({keyword}): {type(exc).__name__} {str(exc)[:180]}")

        all_items = dedup_items(all_items)

        if not all_items:
            return {
                "question": req.question,
                "keywords": keywords,
                "built_docs": 0,
                "using_faiss": False,
                "retrieved_count": 0,
                "results": [],
                "answer": (
                    "검색 결과가 0개입니다. 서비스키, 네트워크 연결, "
                    "또는 공공데이터 API 서버 상태를 확인해주세요."
                ),
                "errors": errors[:15],
            }

        docs = [food_item_to_document(item, index) for index, item in enumerate(all_items)]
        rag_store.build(docs, use_faiss=req.use_faiss)

        retrieved = rag_store.search(req.question, top_k=req.top_k)
        context = make_llm_context(retrieved, max_chars=4500)
        answer = llm.answer_with_context(req.question, keywords, context)

        return {
            "question": req.question,
            "keywords": keywords,
            "built_docs": len(docs),
            "using_faiss": rag_store.use_faiss,
            "retrieved_count": len(retrieved),
            "results": [{"id": doc.id, "text": doc.text} for doc in retrieved],
            "answer": answer,
            "errors": errors[:15],
        }

    return app


app = create_app()
