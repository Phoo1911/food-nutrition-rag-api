# Food Nutrition RAG API

공공데이터포털의 식품 영양성분 API 5종을 한 번에 검색하고, 검색된 식품 데이터를 근거로 한국어 답변을 만드는 FastAPI 서비스입니다.

This was my part of **Diabeta**, a team project on meal management for people with diabetes. The team planned the mobile app (food recognition, glucose prediction, wearable sync); I built the nutrition lookup backend described here. Those other features are not in this repository.

## Why

Nutrition data for Korean foods is split across several public APIs (raw ingredients, processed foods, restaurant dishes, MFDS DB), each with different field names. Asking "how much sodium is in Shin Ramyun?" meant querying each one by hand and comparing rows. This service does that in one request and answers only from the rows it found.

## How a request flows

```
POST /ask {"question": "신라면 나트륨 알려줘"}
  │
  ├─ 1. keywords   Gemini extracts food names → ["신라면"]
  │                (no key / Gemini error → rule-based tokenizer)
  ├─ 2. fetch      query each selected public API per keyword
  │                (retries on 429/5xx, connect/read timeouts, errors collected per API)
  ├─ 3. dedupe     merge rows by food name + code
  ├─ 4. retrieve   embed rows, rank against the question (FAISS or NumPy cosine)
  └─ 5. answer     Gemini summarizes the top-k rows; says so when data is missing
```

Each request builds its own index, so concurrent requests never see each other's rows. The embedding model is loaded once and shared.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env          # then fill in DATA_GO_KR_SERVICE_KEY
uvicorn main:app --reload
```

Open http://127.0.0.1:8000/docs to try it from Swagger.

Or with Docker (the embedding model is baked into the image):

```bash
docker build -t food-nutrition-rag-api .
docker run -p 8000:8000 --env-file .env food-nutrition-rag-api
```

| Variable | Required | Notes |
| --- | --- | --- |
| `DATA_GO_KR_SERVICE_KEY` | yes | 공공데이터포털 인증키 (can also be sent as `serviceKey` in the body) |
| `GEMINI_API_KEY` | no | Without it, keywords come from the local tokenizer and no summary is written |
| `GEMINI_MODEL` | no | default `gemini-2.5-flash` |
| `EMBED_MODEL` | no | default `paraphrase-multilingual-MiniLM-L12-v2` (handles Korean) |
| `CONNECT_TIMEOUT` / `READ_TIMEOUT` / `TOTAL_RETRIES` | no | public API call settings |

## Example

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "신라면 나트륨 알려줘", "apis": ["process", "food"], "top_k": 3}'
```

Response shape:

```json
{
  "question": "신라면 나트륨 알려줘",
  "keywords": ["신라면"],
  "built_docs": 12,
  "using_faiss": true,
  "retrieved_count": 3,
  "results": [{"id": "FOOD_00000", "text": "식품명: ...\n[영양]\n나트륨(mg): ..."}],
  "answer": "...",
  "errors": []
}
```

If an API times out or rejects the key, the request still returns whatever the other APIs found, and `errors` lists which call failed (`"food(신라면): TimeoutError ..."`).

Other endpoints: `GET /` (version, enabled APIs) and `GET /health` (FAISS available, Gemini configured).

## Tests

```bash
pytest
PYTHONPATH=. python scripts/smoke_retrieval.py   # loads the real embedding model
```

The tests don't touch the network or download a model. They stub the public APIs, Gemini and the embedder, and cover:

- keyword parsing when Gemini wraps JSON in code fences or returns plain text
- no retry on 4xx (bad key, retired model); backoff and retry on 429 / connection errors
- `/ask` end to end: dedupe, ranking, per-API error reporting, missing service key
- two indexes built back to back keep separate documents

CI runs the tests, the retrieval smoke test with the real multilingual model, and a Docker build that must answer `/health`.

## Limitations

- Answers are only as good as the public data; many processed foods have no row, or have partial nutrient fields.
- Field mapping for the `mfds_db` API has not been rechecked against a live response, so its nutrient columns may come back empty.
- Not deployed publicly, and not validated with real users. This is not medical advice.

## Stack

Python 3.10+, FastAPI, Requests, sentence-transformers, FAISS (optional), Gemini API, pytest
