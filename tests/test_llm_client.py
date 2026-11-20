import requests

import llm_client
from llm_client import LLMClient


class FakeResponse:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self._payload = payload or {}

    def json(self):
        return self._payload


def gemini_text(text):
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


def make_client(monkeypatch, responses):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(llm_client.time, "sleep", lambda _: None)
    calls = []

    def fake_post(url, **kwargs):
        calls.append(kwargs)
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(llm_client.requests, "post", fake_post)
    return LLMClient(), calls


def test_keywords_parse_fenced_json(monkeypatch):
    client, _ = make_client(monkeypatch, [FakeResponse(200, gemini_text('```json\n["신라면", "라면"]\n```'))])

    assert client.extract_keywords("신라면 영양 정보 알려줘") == ["신라면", "라면"]


def test_api_key_is_sent_in_header_not_url(monkeypatch):
    client, calls = make_client(monkeypatch, [FakeResponse(200, gemini_text('["두부"]'))])

    client.extract_keywords("두부 단백질")

    assert calls[0]["headers"] == {"x-goog-api-key": "test-key"}
    assert "params" not in calls[0]


def test_retired_model_does_not_retry(monkeypatch):
    client, calls = make_client(monkeypatch, [FakeResponse(404)])

    assert client.extract_keywords("신라면 나트륨") == ["신라면", "나트륨"]
    assert len(calls) == 1


def test_rate_limit_retries_then_succeeds(monkeypatch):
    client, calls = make_client(
        monkeypatch,
        [FakeResponse(429), requests.ConnectionError(), FakeResponse(200, gemini_text('["우유"]'))],
    )

    assert client.extract_keywords("우유 칼슘") == ["우유"]
    assert len(calls) == 3


def test_non_json_reply_falls_back_to_local_keywords(monkeypatch):
    client, _ = make_client(monkeypatch, [FakeResponse(200, gemini_text("신라면입니다"))])

    assert client.extract_keywords("신라면 영양 알려줘") == ["신라면"]


def test_missing_key_skips_network(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(llm_client.requests, "post", lambda *a, **k: (_ for _ in ()).throw(AssertionError))

    client = LLMClient()

    assert client.extract_keywords("현미밥 칼로리") == ["현미밥", "칼로리"]
    assert "GEMINI_API_KEY" in client.answer_with_context("q", ["k"], "ctx")
