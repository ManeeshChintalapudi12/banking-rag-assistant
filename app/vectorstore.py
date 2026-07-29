"""
FAISS vector store: build an index from chunked documents, persist it
to disk, and reload it for querying without re-embedding on every
process start.
"""
import os

from langchain_community.vectorstores import FAISS

from app.config import settings
from app.embeddings import get_embeddings
from app.ingestion import chunk_documents, load_documents


def _cosine_relevance_score_fn(distance: float) -> float:
    """Map FAISS L2 distance (over unit-normalized vectors) to a [0, 1]
    relevance score. For unit vectors, L2 distance^2 = 2 - 2*cos_sim,
    so cos_sim = 1 - distance^2/2, then rescale from [-1, 1] to [0, 1].
    """
    cos_sim = 1.0 - (distance ** 2) / 2.0
    return (cos_sim + 1.0) / 2.0


def build_index(persist: bool = True) -> FAISS:
    documents = load_documents()
    chunks = chunk_documents(documents)
    embeddings = get_embeddings()

    store = FAISS.from_documents(
        chunks, embeddings, relevance_score_fn=_cosine_relevance_score_fn
    )

    if persist:
        os.makedirs(settings.INDEX_DIR, exist_ok=True)
        store.save_local(settings.INDEX_DIR)

    return store


def load_index() -> FAISS:
    if not os.path.isdir(settings.INDEX_DIR):
        raise FileNotFoundError(
            f"No index found at '{settings.INDEX_DIR}'. Run "
            f"`python -m app.build_index` first."
        )
    embeddings = get_embeddings()
    return FAISS.load_local(
        settings.INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
        relevance_score_fn=_cosine_relevance_score_fn,
    )


def get_or_build_index() -> FAISS:
    try:
        return load_index()
    except FileNotFoundError:
        return build_index()
