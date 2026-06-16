from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_root_endpoint_exposes_metadata() -> None:
    response = client.get("/")

    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "Food Nutrition RAG API"
    assert payload["docs_url"] == "/docs"


def test_health_endpoint_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
