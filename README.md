# Food Nutrition RAG API

Korean public food and nutrition data APIs plus Gemini-based summarization, wrapped in a FastAPI service.

This project is a small retrieval-augmented generation backend that:

- searches multiple Korean public nutrition datasets with one question
- deduplicates and structures the returned food records
- builds an embedding-based retrieval layer with `sentence-transformers`
- optionally uses FAISS for vector search
- sends the retrieved context to Gemini for a concise Korean answer

## Technical Highlights

This service combines several backend concerns into one workflow:

- external API integration across multiple data sources
- resilient request handling with retries and timeout controls
- LLM fallback design when a model is unavailable
- RAG pipeline design with retrieval and context construction
- FastAPI service design and typed request validation
- lightweight testing and reproducible local setup

## Architecture

1. User sends a question to `/ask`
2. Gemini extracts search keywords, with a local fallback if Gemini fails
3. The app queries several public food and nutrition APIs
4. Results are deduplicated and transformed into RAG documents
5. Embeddings are built with `sentence-transformers`
6. Relevant documents are retrieved with FAISS or cosine similarity
7. Gemini produces a final answer grounded in the retrieved data

## Tech Stack

- Python
- FastAPI
- Requests
- Sentence Transformers
- FAISS
- Gemini API
- Pytest

## Project Structure

```text
.
|- main.py
|- config.py
|- api_client.py
|- llm_client.py
|- rag_store.py
|- models.py
|- utils.py
|- requirements.txt
|- .env.example
`- tests/
```

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn main:app --reload
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

## Environment Variables

| Variable | Required | Description |
| --- | --- | --- |
| `DATA_GO_KR_SERVICE_KEY` | Yes | Public Data Portal service key |
| `GEMINI_API_KEY` | No | Used for keyword extraction and final summarization |
| `GEMINI_MODEL` | No | Default: `gemini-1.5-flash` |
| `CONNECT_TIMEOUT` | No | HTTP connect timeout |
| `READ_TIMEOUT` | No | HTTP read timeout |
| `TOTAL_RETRIES` | No | Retry count for public API calls |

If `GEMINI_API_KEY` is missing, the service still runs with local keyword fallback, but answer summarization becomes limited.

## Example Request

```bash
curl -X POST "http://127.0.0.1:8000/ask" ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"Shin Ramyun nutrition info\",\"top_k\":3,\"top_rows\":50}"
```

## Endpoints

- `GET /` : project metadata
- `GET /health` : runtime health check
- `POST /ask` : nutrition question answering entry point

## Tests

```bash
pytest
```

## Suggested GitHub Repo Description

```text
FastAPI-based Korean food nutrition RAG API using public data, sentence-transformers, FAISS, and Gemini.
```

## Suggested Resume Bullet

```text
Built a FastAPI-based RAG service that integrates multiple Korean public nutrition APIs, vector retrieval with sentence-transformers and FAISS, and Gemini-powered summarization to answer food nutrition questions with grounded data.
```

## Next Improvements

- add response models for the `/ask` endpoint
- cache repeated API queries
- add async I/O and request concurrency
- add Docker support and CI workflow
- log retrieval quality and prompt outputs for evaluation
