import pytest
from konan import (
    FixedSizeChunker,
    MarkdownChunker,
    NaiveChunker,
    RecursiveChunker,
    SentenceChunker,
    TokenChunker,
)

PROSE = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump! "
    "Sphinx of black quartz, judge my vow."
)
UNICODE = "Cafés sind süß. Zwölf Boxkämpfer jagen Viktor quer über den Sylter Deich. 😀 emoji rocks. Final line here."
LOREM = " ".join(f"word{i}" for i in range(500))

SYNC_CHUNKERS = [
    NaiveChunker(chunk_size=5),
    FixedSizeChunker(chunk_size=80, chunk_overlap=20),
    RecursiveChunker(chunk_size=80, chunk_overlap=20),
    SentenceChunker(max_chars=80, overlap_sentences=1),
    TokenChunker(chunk_size=16, chunk_overlap=4),
]


@pytest.mark.parametrize("chunker", SYNC_CHUNKERS, ids=lambda c: type(c).__name__)
@pytest.mark.parametrize("text", [PROSE, UNICODE], ids=["ascii", "unicode"])
def test_offsets_are_char_accurate(chunker, text):
    chunks = chunker.chunk(text)
    assert chunks
    for c in chunks:
        assert text[c.start : c.end] == c.text


@pytest.mark.parametrize("chunker", SYNC_CHUNKERS, ids=lambda c: type(c).__name__)
def test_empty_input(chunker):
    assert chunker.chunk("") == []


def test_chunk_many_matches_chunk():
    chunker = RecursiveChunker(chunk_size=80, chunk_overlap=20)
    texts = [PROSE, LOREM, "short", UNICODE]
    assert chunker.chunk_many(texts) == [chunker.chunk(t) for t in texts]


def test_invalid_config_raises_value_error():
    with pytest.raises(ValueError):
        RecursiveChunker(chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError):
        TokenChunker(encoding="nope")
    with pytest.raises(ValueError):
        NaiveChunker(chunk_size=0)


def test_chunk_dunder_methods():
    c = RecursiveChunker(chunk_size=80, chunk_overlap=20).chunk(PROSE)[0]
    assert len(c) == len(c.text)
    assert c == c
    assert "Chunk(" in repr(c)
    assert isinstance(hash(c), int)
    assert c.hash


MD = """# Guide

Intro paragraph about the guide.

## Install

Run the installer.

```bash
pip install konan
```
"""


def test_markdown_breadcrumbs():
    chunks = MarkdownChunker(chunk_size=200, chunk_overlap=0).chunk(MD)
    assert chunks[0].text.startswith("# Guide")
    assert any(c.text.startswith("# Guide > ## Install") for c in chunks)
    for c in chunks:
        assert c.text.endswith(MD[c.start : c.end])


def test_markdown_code_fence_never_split():
    chunks = MarkdownChunker(chunk_size=10, chunk_overlap=0).chunk(MD)
    code = [c for c in chunks if "```bash" in c.text]
    assert code and "pip install konan" in code[0].text
