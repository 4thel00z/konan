import pytest
from konan import (
    FixedSizeChunker,
    MarkdownChunker,
    RecursiveChunker,
    SentenceChunker,
    TokenChunker,
)

WORDS = ["river", "stone", "glass", "winter", "signal", "harbor", "copper", "lantern", "orbit"]
PROSE = "\n\n".join(
    " ".join(
        f"{WORDS[(p * 7 + s) % len(WORDS)].capitalize()} "
        + " ".join(WORDS[(p * 3 + s * 5 + k * k) % len(WORDS)] for k in range(4 + (p + s) % 9))
        + "."
        for s in range(6)
    )
    for p in range(150)
)

SizedChunker = RecursiveChunker | FixedSizeChunker | MarkdownChunker | SentenceChunker

TOKEN_CHUNKERS: list[SizedChunker] = [
    RecursiveChunker(chunk_size=64, chunk_overlap=16, encoding="cl100k_base"),
    RecursiveChunker(chunk_size=64, chunk_overlap=16, encoding="o200k_base", stable_boundaries=True),
    FixedSizeChunker(chunk_size=64, chunk_overlap=16, respect_sentences=False, encoding="cl100k_base"),
    MarkdownChunker(chunk_size=64, chunk_overlap=16, encoding="cl100k_base"),
    SentenceChunker(max_chars=64, overlap_sentences=1, encoding="cl100k_base"),
]


def token_count(text: str, encoding: str) -> int:
    return len(TokenChunker(chunk_size=1, chunk_overlap=0, encoding=encoding).chunk(text))


@pytest.mark.parametrize("chunker", TOKEN_CHUNKERS, ids=repr)
def test_token_budget_holds_within_edge_slack(chunker: SizedChunker) -> None:
    encoding = "o200k_base" if "o200k_base" in repr(chunker) else "cl100k_base"
    chunks = chunker.chunk(PROSE)
    assert len(chunks) > 10
    for c in chunks:
        assert PROSE[c.start : c.end] == c.text
        assert token_count(c.text, encoding) <= 64 + 3


def test_markdown_breadcrumbs_count_toward_token_budget() -> None:
    sections = PROSE.split("\n\n")
    md = "\n\n".join(
        f"# Handbook of rivers {i // 10}\n\n## Part {i}: the long winter harbor chapter\n\n{body}"
        for i, body in enumerate(sections)
    )
    chunker = MarkdownChunker(chunk_size=64, chunk_overlap=16, encoding="cl100k_base")
    chunks = chunker.chunk(md)
    assert any(c.text.startswith("# Handbook") for c in chunks)
    assert all(token_count(c.text, "cl100k_base") <= 64 + 3 for c in chunks)


def test_default_is_chars() -> None:
    plain = RecursiveChunker(chunk_size=200, chunk_overlap=40).chunk(PROSE)
    explicit = RecursiveChunker(
        chunk_size=200, chunk_overlap=40, encoding=None, stable_boundaries=False
    ).chunk(PROSE)
    assert plain == explicit
    assert all(len(c.text) <= 200 for c in plain)


def test_unknown_encoding_rejected() -> None:
    with pytest.raises(ValueError, match="unknown encoding"):
        RecursiveChunker(encoding="nope")


def test_stable_boundaries_keep_chunks_after_an_edit() -> None:
    at = PROSE.index(". ", len(PROSE) // 2) + 2
    edited = PROSE[:at] + "Inserted sentence about nothing much. " + PROSE[at:]
    chunker = RecursiveChunker(chunk_size=300, chunk_overlap=0, stable_boundaries=True)
    before = {c.hash for c in chunker.chunk(PROSE)}
    after = chunker.chunk(edited)
    assert sum(c.hash not in before for c in after) <= 5


def test_reprs_show_options_only_when_set() -> None:
    assert repr(RecursiveChunker(chunk_size=80, chunk_overlap=20)) == (
        "RecursiveChunker(chunk_size=80, chunk_overlap=20, separators=None)"
    )
    assert repr(
        RecursiveChunker(chunk_size=80, chunk_overlap=20, encoding="cl100k_base", stable_boundaries=True)
    ) == (
        'RecursiveChunker(chunk_size=80, chunk_overlap=20, separators=None, '
        'encoding="cl100k_base", stable_boundaries=True)'
    )
    assert repr(FixedSizeChunker(chunk_size=80, chunk_overlap=20, stable_boundaries=True)) == (
        "FixedSizeChunker(chunk_size=80, chunk_overlap=20, respect_sentences=True, stable_boundaries=True)"
    )
    assert repr(MarkdownChunker(chunk_size=200, chunk_overlap=0, encoding="o200k_base")) == (
        'MarkdownChunker(chunk_size=200, chunk_overlap=0, encoding="o200k_base")'
    )
    assert repr(SentenceChunker(max_chars=80, overlap_sentences=1, encoding="cl100k_base")) == (
        'SentenceChunker(max_chars=80, overlap_sentences=1, encoding="cl100k_base")'
    )
