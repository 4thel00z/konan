import asyncio

from konan import RecursiveChunker, SentenceChunker

PROSE = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump! "
    "Sphinx of black quartz, judge my vow."
)


async def test_chunk_async_matches_sync():
    c = RecursiveChunker(chunk_size=80, chunk_overlap=20)
    assert await c.chunk_async(PROSE) == c.chunk(PROSE)


async def test_chunk_many_async_matches_sync():
    c = SentenceChunker(max_chars=80, overlap_sentences=1)
    texts = [PROSE, PROSE * 3, "One sentence."]
    assert await c.chunk_many_async(texts) == c.chunk_many(texts)


async def test_async_methods_run_concurrently():
    c = RecursiveChunker(chunk_size=80, chunk_overlap=20)
    results = await asyncio.gather(
        c.chunk_async(PROSE), c.chunk_many_async([PROSE, PROSE * 2])
    )
    assert results[0] == c.chunk(PROSE)
    assert results[1] == c.chunk_many([PROSE, PROSE * 2])
