"""
Banking Policy & Compliance RAG Assistant — FastAPI service.

Run locally:
    uvicorn app.main:app --reload

Then open http://127.0.0.1:8000/docs for interactive Swagger UI.
"""
import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.models import HealthResponse, QueryRequest, QueryResponse
from app.rag_chain import answer_question

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("banking_rag")

app = FastAPI(
    title="Banking Policy & Compliance RAG Assistant",
    description=(
        "Retrieval-augmented generation service for answering questions "
        "about wire transfer, hold, and dispute policies with cited "
        "sources and confidence-based escalation to human review."
    ),
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        provider=settings.LLM_PROVIDER,
        index_ready=os.path.isdir(settings.INDEX_DIR),
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    try:
        return answer_question(request.question, top_k=request.top_k)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive catch-all
        logger.exception("Unhandled error answering query")
        raise HTTPException(status_code=500, detail="Internal error") from exc


# Mounted last so it never shadows the /health or /query routes above —
# Starlette matches explicit routes before falling through to a mount.
_UI_DIR = os.path.join(os.path.dirname(__file__), "..", "ui")
if os.path.isdir(_UI_DIR):
    app.mount("/", StaticFiles(directory=_UI_DIR, html=True), name="ui")
