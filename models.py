from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


@dataclass
class Document:
    id: str
    text: str
    metadata: Dict[str, Any]


class AskRequest(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "question": "신라면 영양 정보 알려줘",
                "apis": ["material", "process", "food", "nutri", "mfds_db"],
                "top_rows": 100,
                "top_k": 3,
                "resp_type": "json",
                "use_faiss": True,
                "serviceKey": "your_data_go_kr_service_key",
            }
        },
    )

    question: str = Field(..., description="사용자 질문")
    apis: List[Literal["material", "process", "food", "nutri", "mfds_db"]] = Field(
        default_factory=lambda: ["material", "process", "food", "nutri", "mfds_db"],
        description="검색할 공공 데이터 API 목록",
    )
    top_rows: int = Field(100, ge=1, le=1000, description="API에서 읽어올 최대 row 수")
    top_k: int = Field(3, ge=1, le=20, description="RAG 검색에 사용할 top_k")
    resp_type: Literal["json", "xml"] = Field("json", description="응답 타입")
    use_faiss: bool = Field(True, description="FAISS 사용 여부")
    serviceKey: Optional[str] = Field(
        None,
        alias="servicekey",
        description="공공데이터포털 서비스키. 없으면 DATA_GO_KR_SERVICE_KEY 환경변수 사용",
    )
