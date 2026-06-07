import pytest
from konan import EmbeddingError, OpenAIEmbedder, SemanticChunker

TEXT = (
    "Cats purr softly. My cat naps all day. "
    "Quantum entanglement defies intuition. Quantum computers exploit superposition."
)


async def test_semantic_chunk_async(embeddings_url):
    chunker = SemanticChunker(
        embedder=OpenAIEmbedder(base_url=embeddings_url, model="test"), threshold=0.5
    )
    chunks = await chunker.chunk_async(TEXT)
    assert len(chunks) == 2
    assert "cat" in chunks[0].text.lower()
    assert chunks[1].text.startswith("Quantum")
    for c in chunks:
        assert TEXT[c.start : c.end] == c.text


def test_semantic_chunk_sync(embeddings_url):
    chunker = SemanticChunker(
        embedder=OpenAIEmbedder(base_url=embeddings_url, model="test"), threshold=0.5
    )
    assert len(chunker.chunk(TEXT)) == 2


async def test_embedder_direct(embeddings_url):
    embedder = OpenAIEmbedder(base_url=embeddings_url, model="test")
    vectors = await embedder.embed_async(["a cat", "physics"])
    assert vectors == [[1.0, 0.0], [0.0, 1.0]]


async def test_python_callable_embedder():
    async def embed(texts: list[str]) -> list[list[float]]:
        return [
            [1.0, 0.0] if "cat" in t.lower() else [0.0, 1.0] for t in texts
        ]

    chunker = SemanticChunker(embedder=embed, threshold=0.5)
    chunks = await chunker.chunk_async(TEXT)
    assert len(chunks) == 2


async def test_embedding_error_on_unreachable_endpoint():
    chunker = SemanticChunker(
        embedder=OpenAIEmbedder(base_url="http://127.0.0.1:9", model="x"), threshold=0.5
    )
    with pytest.raises(EmbeddingError):
        await chunker.chunk_async("One sentence. Two sentences. Three now.")


def test_rejects_non_embedder():
    with pytest.raises(ValueError):
        SemanticChunker(embedder=42)
