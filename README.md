# Banking Policy & Compliance RAG Assistant

A Retrieval-Augmented Generation (RAG) service that answers questions about
banking policies — wire transfers, Regulation CC holds, and Regulation E
disputes — from a set of source documents, with **cited sources** and a
**confidence score** so low-confidence answers can be routed to a human
reviewer instead of presented as fact.

Built with **FastAPI**, **LangChain**, and **FAISS**.

## Why this project

Banking and financial-services teams increasingly use internal AI assistants
to help staff navigate policy documents (wire limits, hold rules, dispute
timelines) without hunting through PDFs. This project is a scoped-down,
end-to-end reference implementation of that pattern: document ingestion →
chunking → embedding → vector retrieval → grounded generation → citation →
confidence-based escalation.

## Architecture

```
data/sample_docs/*.md   →  ingestion.py (load + chunk)
                        →  embeddings.py (OpenAI or local hashing embedder)
                        →  vectorstore.py (FAISS index, persisted to disk)
                        →  rag_chain.py (retrieve top-k, generate grounded answer)
                        →  main.py (FastAPI /query and /health endpoints)
```

## Running it locally (no API key required)

The project ships with `LLM_PROVIDER=local` by default, which uses a
dependency-free deterministic embedding and an extractive answerer — so it
runs and is fully testable with **zero external API keys**.

```bash
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env

python -m app.build_index        # builds the FAISS index from data/sample_docs
uvicorn app.main:app --reload
```

Then open **http://127.0.0.1:8000/docs** for interactive Swagger UI, or:

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the fee for an outgoing domestic wire transfer?"}'
```

## Using a real LLM (OpenAI)

Set the following in `.env` and re-run `python -m app.build_index`:

```
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

This switches both the embedding model and the answer generator to OpenAI,
with the same grounding/citation-enforcing system prompt.

## Running with Docker

```bash
docker compose up --build
```

## Running tests

```bash
pytest -v
```

## Key design choices

- **Confidence-gated answers.** Every response includes a `confidence` score
  derived from vector similarity. Below a threshold, the API flags the
  answer as low-confidence and recommends human review rather than
  presenting a guess as authoritative — a pattern used for high-stakes
  domains like financial compliance.
- **Citations by chunk id**, not just document name, so a reviewer can trace
  an answer back to the exact paragraph it came from.
- **Pluggable embedding/generation backend** so the same pipeline runs
  offline for development/CI and against a real LLM in production without
  changing any application code — only configuration.

## Project structure

```
.
├── app/
│   ├── main.py            # FastAPI app and routes
│   ├── config.py          # environment-driven settings
│   ├── ingestion.py        # document loading + chunking
│   ├── embeddings.py       # OpenAI / local embedding backends
│   ├── vectorstore.py      # FAISS build/load/persist
│   ├── rag_chain.py        # retrieval + grounded generation
│   ├── build_index.py      # CLI: build the index
│   └── models.py           # Pydantic request/response schemas
├── data/sample_docs/       # example banking policy documents
├── tests/                  # pytest suite
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Possible extensions

- Swap FAISS for a managed vector DB (Pinecone, Azure AI Search) for
  multi-instance deployments.
- Add hybrid (keyword + vector) retrieval and re-ranking.
- Add an evaluation harness (groundedness/relevance scoring) over a curated
  test question set.
- Add role-based access control so different policy sets are scoped to
  different user roles.

## License

MIT — see [LICENSE](LICENSE).
