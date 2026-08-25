import pytest

from app.embeddings import LocalHashingEmbeddings


def test_embedding_is_deterministic():
    emb = LocalHashingEmbeddings()
    v1 = emb.embed_query("wire transfer limit")
    v2 = emb.embed_query("wire transfer limit")
    assert v1 == v2


def test_embedding_is_normalized():
    emb = LocalHashingEmbeddings()
    v = emb.embed_query("international wire transfer compliance review")
    norm = sum(x * x for x in v) ** 0.5
    assert abs(norm - 1.0) < 1e-6


def test_different_text_gives_different_vector():
    emb = LocalHashingEmbeddings()
    v1 = emb.embed_query("wire transfer")
    v2 = emb.embed_query("debit card dispute")
    assert v1 != v2


@pytest.mark.parametrize("dims", [0, -1, 1.5, True])
def test_embedding_rejects_invalid_dimensions(dims):
    with pytest.raises(ValueError, match="positive integer"):
        LocalHashingEmbeddings(dims=dims)


def test_empty_text_returns_zero_vector():
    emb = LocalHashingEmbeddings(dims=8)
    assert emb.embed_query("") == [0.0] * 8
