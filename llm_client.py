import json
import os
import re
import time
from typing import List

import requests

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _parse_keyword_list(output: str) -> List[str]:
    """Gemini often wraps JSON in ```json fences; strip them before parsing."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", output.strip())
    values = json.loads(text)
    if not isinstance(values, list):
        return []
    return [v.strip() for v in values if isinstance(v, str) and v.strip()]


class LLMClient:
    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.timeout = 30

    def _gemini_generate(self, prompt: str, retries: int = 3) -> str:
        if not self.api_key:
            return ""

        url = f"{self.base_url}/models/{self.model}:generateContent"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2},
        }

        for attempt in range(retries):
            try:
                response = requests.post(
                    url,
                    headers={"x-goog-api-key": self.api_key},
                    json=payload,
                    timeout=self.timeout,
                )
            except requests.RequestException:
                response = None

            if response is not None and response.status_code not in RETRYABLE_STATUS:
                # 400/403/404 (bad key, retired model) will not succeed on retry.
                if not response.ok:
                    return ""
                try:
                    data = response.json()
                except ValueError:
                    return ""
                parts = (data.get("candidates") or [{}])[0].get("content", {}).get("parts") or [{}]
                return (parts[0].get("text") or "").strip()

            if attempt < retries - 1:
                time.sleep(2**attempt)

        return ""

    def _fallback_keywords(self, question: str, max_k: int = 5) -> List[str]:
        normalized = re.sub(r"[^0-9A-Za-z가-힣\s]", " ", question)
        stop_words = {
            "영양",
            "알려줘",
            "알려줘요",
            "뭐야",
            "뭔가",
            "얼마",
            "몇",
            "개",
            "두",
            "한",
            "좀",
            "해주세요",
            "해줘",
            "분석",
            "요약",
        }

        tokens = []
        for token in normalized.split():
            token = token.strip()
            if len(token) >= 2 and token not in stop_words:
                tokens.append(token)

        deduplicated: List[str] = []
        seen = set()
        for token in tokens:
            if token not in seen:
                deduplicated.append(token)
                seen.add(token)

        return deduplicated[:max_k] if deduplicated else [question.strip()[:20]]

    def extract_keywords(self, question: str, max_k: int = 5) -> List[str]:
        prompt = f"""
너는 사용자의 질문에서 식품명, 제품명, 핵심 검색어를 뽑는 도우미다.

질문: {question}

규칙:
- 검색에 쓸 핵심 키워드 1~{max_k}개를 고른다.
- JSON 배열로만 출력한다. 예: ["라면", "신라면"]
- 설명 문장이나 부가 텍스트는 출력하지 않는다.
"""
        output = self._gemini_generate(prompt).strip()

        if not output:
            return self._fallback_keywords(question, max_k)

        try:
            cleaned = _parse_keyword_list(output)
        except ValueError:
            cleaned = []

        return cleaned[:max_k] if cleaned else self._fallback_keywords(question, max_k)

    def answer_with_context(self, question: str, keywords: List[str], context: str) -> str:
        if not self.api_key:
            return "Gemini API 키가 없어 요약을 생성할 수 없습니다. GEMINI_API_KEY를 설정해주세요."

        prompt = f"""
너는 공공 영양 데이터를 바탕으로 쉬운 한국어 답변을 만드는 도우미다.
과장하거나 추측하지 말고, 제공된 정보 안에서만 설명한다.

사용자 질문: {question}
검색 키워드: {", ".join(keywords)}

참고 데이터:
{context}

요청:
- 핵심 영양 정보를 짧고 명확하게 요약한다.
- 가능하면 숫자 비교 포인트를 함께 설명한다.
- 데이터에 없는 내용은 모른다고 말한다.
"""
        output = self._gemini_generate(prompt).strip()
        return output or "Gemini 호출이 실패해 요약을 만들지 못했습니다. 잠시 후 다시 시도해주세요."
