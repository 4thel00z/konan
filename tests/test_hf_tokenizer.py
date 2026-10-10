from pathlib import Path

import pytest
from konan import FixedSizeChunker, MarkdownChunker, RecursiveChunker, SentenceChunker
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers

WORDS = ["river", "stone", "glass", "winter", "signal", "harbor", "copper", "lantern", "orbit"]
PROSE = "\n\n".join(
    " ".join(
        f"{WORDS[(p * 7 + s) % len(WORDS)].capitalize()} "
        + " ".join(WORDS[(p * 3 + s * 5 + k * k) % len(WORDS)] for k in range(4 + (p + s) % 9))
        + "."
        for s in range(6)
    )
    for p in range(150)
) + "\n\nSchöne Grüße aus Köln, 😀 bis bald!"


def word_level() -> Tokenizer:
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Lowercase()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.enable_truncation(4)
    return tokenizer


def byte_level_bpe() -> Tokenizer:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=400, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tokenizer.train_from_iterator([PROSE], trainer=trainer)
    return tokenizer


def count(tokenizer: Tokenizer, text: str) -> int:
    tokenizer.no_truncation()
    return len(tokenizer.encode(text, add_special_tokens=False).ids)


def test_word_level_chunks_fit_exactly() -> None:
    tokenizer = word_level()
    for chunker in [
        RecursiveChunker(chunk_size=32, chunk_overlap=0, encoding=tokenizer),
        RecursiveChunker(chunk_size=32, chunk_overlap=8, encoding=tokenizer, stable_boundaries=True),
        FixedSizeChunker(chunk_size=32, chunk_overlap=8, respect_sentences=False, encoding=tokenizer),
        MarkdownChunker(chunk_size=32, chunk_overlap=0, encoding=tokenizer),
        SentenceChunker(max_chars=32, overlap_sentences=0, encoding=tokenizer),
    ]:
        chunks = chunker.chunk(PROSE)
        assert len(chunks) > 10, chunker
        for c in chunks:
            assert PROSE[c.start : c.end] == c.text
            assert count(tokenizer, c.text) <= 32, (chunker, c)


def test_byte_level_bpe_fits_within_edge_slack() -> None:
    tokenizer = byte_level_bpe()
    chunks = RecursiveChunker(chunk_size=48, chunk_overlap=0, encoding=tokenizer).chunk(PROSE)
    assert len(chunks) > 10
    for c in chunks:
        assert PROSE[c.start : c.end] == c.text
        assert count(tokenizer, c.text) <= 48 + 3


def test_tokenizer_json_path(tmp_path: Path) -> None:
    path = tmp_path / "tokenizer.json"
    word_level().save(str(path))
    from_path = RecursiveChunker(chunk_size=32, chunk_overlap=0, encoding=path).chunk(PROSE)
    from_object = RecursiveChunker(chunk_size=32, chunk_overlap=0, encoding=word_level()).chunk(PROSE)
    assert from_path == from_object


def test_reprs() -> None:
    assert repr(RecursiveChunker(chunk_size=32, chunk_overlap=0, encoding=word_level())) == (
        "RecursiveChunker(chunk_size=32, chunk_overlap=0, separators=None, "
        "encoding=<tokenizers.Tokenizer>)"
    )


def test_bad_encodings_rejected(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="encoding must be"):
        RecursiveChunker(encoding=42)
    with pytest.raises(ValueError):
        RecursiveChunker(encoding=tmp_path / "missing.json")
    with pytest.raises(ValueError, match="pathlib.Path"):
        RecursiveChunker(encoding="tokenizer.json")
