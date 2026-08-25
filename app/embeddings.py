"""
Embedding backends.

Two implementations of the LangChain `Embeddings` interface:

- OpenAIEmbeddings (from langchain-openai) when LLM_PROVIDER=openai.
- LocalHashingEmbeddings, a dependency-free, deterministic bag-of-words
  hashing embedding used when running fully offline. It is not a
  substitute for a real embedding model in production, but it makes
  the retrieval pipeline genuinely runnable (and testable in CI)
  without any API keys or GPU/torch downloads.
"""
import hashlib
import math
import re
from langchain_core.embeddings import Embeddings

from app.config import settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# A small stopword list is enough to stop generic query words ("what",
# "is", "the", "for") from dominating similarity scores in the simple
# hashing embedder below — without it, unrelated questions can look
# deceptively similar to policy text purely on shared function words.
_STOPWORDS = frozenset(
    """
    a an the is are was were be been being of in on at to for with
    and or but if then than so as by from into it its this that these
    those what which who whom how when where why do does did can
    could will would shall should may might must not no nor
    """.split()
)


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class LocalHashingEmbeddings(Embeddings):
    """Deterministic hashing-based bag-of-words embedding.

    Each token is hashed into one of `dims` buckets; the resulting
    vector is L2-normalized so cosine similarity behaves sensibly.
    This gives real (if unsophisticated) semantic-ish clustering for
    short policy documents, entirely offline.
    """

    def __init__(self, dims: int = 512) -> None:
        if isinstance(dims, bool) or not isinstance(dims, int) or dims <= 0:
            raise ValueError("dims must be a positive integer")
        self.dims = dims

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dims
        for token in _tokenize(text):
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dims
            sign = 1.0 if (h // self.dims) % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def get_embeddings() -> Embeddings:
    provider = settings.LLM_PROVIDER.strip().lower()
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        if not settings.OPENAI_API_KEY:
            raise RuntimeError(
                "LLM_PROVIDER=openai but OPENAI_API_KEY is not set. "
                "Set it in .env or switch LLM_PROVIDER=local."
            )
        return OpenAIEmbeddings(
            model=settings.OPENAI_EMBEDDING_MODEL,
            api_key=settings.OPENAI_API_KEY,
        )
    if provider == "local":
        return LocalHashingEmbeddings()
    raise ValueError(
        f"Unsupported LLM_PROVIDER={settings.LLM_PROVIDER!r}. "
        "Choose 'local' or 'openai'."
    )
