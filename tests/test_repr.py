from konan import (
    FixedSizeChunker,
    MarkdownChunker,
    NaiveChunker,
    OpenAIEmbedder,
    RecursiveChunker,
    SemanticChunker,
    SentenceChunker,
    TokenChunker,
)


def test_chunker_reprs_reveal_config():
    assert repr(NaiveChunker(chunk_size=5)) == "NaiveChunker(chunk_size=5)"
    assert repr(FixedSizeChunker(chunk_size=80, chunk_overlap=20)) == (
        "FixedSizeChunker(chunk_size=80, chunk_overlap=20, respect_sentences=True)"
    )
    assert repr(RecursiveChunker(chunk_size=80, chunk_overlap=20)) == (
        "RecursiveChunker(chunk_size=80, chunk_overlap=20, separators=None)"
    )
    assert repr(RecursiveChunker(chunk_size=80, chunk_overlap=20, separators=["\n", " "])) == (
        'RecursiveChunker(chunk_size=80, chunk_overlap=20, separators=["\\n", " "])'
    )
    assert repr(SentenceChunker(max_chars=80, overlap_sentences=1)) == (
        "SentenceChunker(max_chars=80, overlap_sentences=1)"
    )
    assert repr(MarkdownChunker(chunk_size=200, chunk_overlap=0)) == (
        "MarkdownChunker(chunk_size=200, chunk_overlap=0)"
    )
    assert repr(TokenChunker(chunk_size=16, chunk_overlap=4)) == (
        'TokenChunker(chunk_size=16, chunk_overlap=4, encoding="cl100k_base")'
    )


def test_embedder_repr_redacts_api_key():
    secret = repr(OpenAIEmbedder(base_url="http://x/v1", model="m", api_key="sk-secret"))
    assert "sk-secret" not in secret
    assert 'api_key="***"' in secret
    assert 'base_url="http://x/v1"' in secret
    assert "timeout=30.0" in secret
    assert "max_retries=2" in secret
    assert "dimensions=None" in secret

    anon = repr(OpenAIEmbedder(base_url="http://x/v1", model="m", dimensions=256))
    assert "api_key=None" in anon
    assert "dimensions=256" in anon


def test_semantic_chunker_repr():
    embedder = OpenAIEmbedder(base_url="http://x/v1", model="m")
    chunker = SemanticChunker(embedder=embedder, threshold=0.5)
    text = repr(chunker)
    assert text.startswith("SemanticChunker(embedder=OpenAIEmbedder(")
    assert "threshold=0.5" in text
    assert "percentile=95.0" in text

    async def embed(texts):
        return [[1.0, 0.0] for _ in texts]

    callable_repr = repr(SemanticChunker(embedder=embed, percentile=90.0))
    assert "embedder=<async callable>" in callable_repr
    assert "percentile=90.0" in callable_repr


def test_chunk_repr_truncation():
    short = NaiveChunker(chunk_size=10).chunk("tiny text")[0]
    assert "…" not in repr(short)
    assert 'text="tiny text"' in repr(short)

    long = NaiveChunker(chunk_size=50).chunk("word " * 40)[0]
    r = repr(long)
    assert "…" in r
    assert len(long.text) > 40
