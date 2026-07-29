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
