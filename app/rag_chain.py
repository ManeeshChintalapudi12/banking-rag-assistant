"""
Retrieval-augmented generation.

`answer_question` retrieves the top-k most relevant chunks for a
query, then generates a grounded answer that cites the specific
source documents it drew from. Two generation backends are
supported:

- OpenAI (LLM_PROVIDER=openai): a chat model is instructed to answer
  strictly from the provided context and to say so explicitly when
  the context does not contain an answer, reducing hallucination.
- Local (LLM_PROVIDER=local): a dependency-free extractive
  summarizer that composes an answer directly from the retrieved
  excerpts. No external API calls, fully deterministic.

A confidence score (derived from vector similarity of the top match)
is returned alongside the answer so callers can decide whether to
trust it or escalate to a human reviewer — the same
"low-confidence-review" pattern described in the resume for the
production healthcare/banking AI assistants this project models.
"""
from typing import List, Tuple

from langchain_core.documents import Document

from app.config import settings
from app.models import QueryResponse, SourceChunk
from app.vectorstore import get_or_build_index

# Below this similarity score we flag the answer as low-confidence
# rather than presenting it as authoritative.
CONFIDENCE_THRESHOLD = 0.6


def _retrieve(question: str, k: int) -> List[Tuple[Document, float]]:
    store = get_or_build_index()
    # FAISS similarity_search_with_relevance_scores normalizes scores
    # to roughly [0, 1] where higher is more similar.
    return store.similarity_search_with_relevance_scores(question, k=k)


def _format_sources(results: List[Tuple[Document, float]]) -> List[SourceChunk]:
    sources = []
    for doc, score in results:
        excerpt = doc.page_content.strip().replace("\n", " ")
        if len(excerpt) > 240:
            excerpt = excerpt[:240].rsplit(" ", 1)[0] + "..."
        sources.append(
            SourceChunk(
                chunk_id=doc.metadata.get("chunk_id", "unknown"),
                source=doc.metadata.get("source", "unknown"),
                excerpt=excerpt,
                score=round(float(score), 4),
            )
        )
    return sources


def _generate_local(question: str, results: List[Tuple[Document, float]]) -> str:
    if not results:
        return "I don't have enough information in the policy documents to answer that."
    top_doc, _ = results[0]
    lead = top_doc.page_content.strip().split("\n\n")[0]
    citation = top_doc.metadata.get("chunk_id", "unknown")
    return f"{lead}\n\n(Source: {citation})"


def _generate_openai(question: str, results: List[Tuple[Document, float]]) -> str:
    from langchain_openai import ChatOpenAI

    context = "\n\n".join(
        f"[{doc.metadata.get('chunk_id')}]\n{doc.page_content}" for doc, _ in results
    )
    system_prompt = (
        "You are a banking policy assistant. Answer ONLY using the "
        "provided context. Every claim must be traceable to a cited "
        "chunk id in square brackets, e.g. [file.md#chunk-1]. If the "
        "context does not contain the answer, say so explicitly and "
        "do not guess."
    )
    llm = ChatOpenAI(
        model=settings.OPENAI_CHAT_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=0,
    )
    response = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ]
    )
    return response.content


def answer_question(question: str, top_k: int = None) -> QueryResponse:
    k = top_k or settings.TOP_K
    results = _retrieve(question, k)

    top_score = results[0][1] if results else 0.0
    grounded = top_score >= CONFIDENCE_THRESHOLD

    if settings.LLM_PROVIDER == "openai":
        answer_text = _generate_openai(question, results)
    else:
        answer_text = _generate_local(question, results)

    if not grounded:
        answer_text = (
            "I couldn't find a confident match in the available policy "
            "documents for this question. Please route this to a human "
            "reviewer rather than relying on the excerpt below.\n\n"
            + answer_text
        )

    return QueryResponse(
        answer=answer_text,
        grounded=grounded,
        confidence=round(float(top_score), 4),
        sources=_format_sources(results),
    )
