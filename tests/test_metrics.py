import sys

import pytest

from konan import TextMetrics
from konan import measure_text


def test_measure_text_counts_unicode_content_without_copying_whitespace() -> None:
    metrics = measure_text(" A\u00a0é日\0! ", 10)

    assert metrics.content_characters == 4
    assert metrics.alphanumeric_characters == 3
    assert metrics.exceeded_limit is False


def test_measure_text_stops_after_source_limit() -> None:
    metrics = measure_text("a \n b c trailing content", 2)

    assert metrics.content_characters == 1
    assert metrics.alphanumeric_characters == 1
    assert metrics.exceeded_limit is True


def test_measure_text_accepts_exact_limit() -> None:
    metrics = measure_text("a b", 3)

    assert metrics.content_characters == 2
    assert metrics.alphanumeric_characters == 2
    assert metrics.exceeded_limit is False


@pytest.mark.parametrize("text", [" " * 1_000, "\0" * 1_000])
def test_measure_text_bounds_whitespace_and_nul_source(text: str) -> None:
    metrics = measure_text(text, 10)

    assert metrics.content_characters == 0
    assert metrics.alphanumeric_characters == 0
    assert metrics.exceeded_limit is True


def test_measure_text_zero_limit() -> None:
    assert measure_text("", 0).exceeded_limit is False
    metrics = measure_text("a", 0)
    assert metrics.content_characters == 0
    assert metrics.alphanumeric_characters == 0
    assert metrics.exceeded_limit is True


def test_measure_text_rejects_negative_limit() -> None:
    with pytest.raises(OverflowError):
        measure_text("text", -1)


def test_measure_text_accepts_maximum_platform_limit() -> None:
    metrics = measure_text("text", 2 * sys.maxsize + 1)
    assert metrics.content_characters == 4
    with pytest.raises(OverflowError):
        measure_text("text", 2 * sys.maxsize + 2)


def test_measure_text_rejects_lone_surrogates() -> None:
    with pytest.raises(UnicodeEncodeError):
        measure_text("\ud800", 1)


def test_text_metrics_is_an_immutable_value() -> None:
    first = measure_text("a b", 3)
    second = measure_text("a b", 3)

    assert isinstance(first, TextMetrics)
    assert type(first).__module__ == "konan"
    assert first == second
    assert repr(first) == (
        "TextMetrics(content_characters=2, alphanumeric_characters=2, "
        "exceeded_limit=False)"
    )
    with pytest.raises(TypeError):
        hash(first)
    with pytest.raises(AttributeError):
        first.content_characters = 3


def test_measure_text_matches_python_unicode_classification() -> None:
    scalars = (
        chr(codepoint)
        for codepoint in range(sys.maxunicode + 1)
        if not 0xD800 <= codepoint <= 0xDFFF
    )
    whitespace: list[str] = []
    non_whitespace: list[str] = []
    alphanumeric: list[str] = []
    non_alphanumeric_content: list[str] = []
    for character in scalars:
        if character.isspace():
            whitespace.append(character)
        else:
            non_whitespace.append(character)
            if character != "\0":
                if character.isalnum():
                    alphanumeric.append(character)
                else:
                    non_alphanumeric_content.append(character)

    whitespace_metrics = measure_text("".join(whitespace), len(whitespace))
    assert whitespace_metrics.content_characters == 0

    non_whitespace_metrics = measure_text(
        "".join(non_whitespace), len(non_whitespace)
    )
    assert non_whitespace_metrics.content_characters == len(non_whitespace) - 1

    alphanumeric_metrics = measure_text("".join(alphanumeric), len(alphanumeric))
    assert alphanumeric_metrics.alphanumeric_characters == len(alphanumeric)

    non_alphanumeric_metrics = measure_text(
        "".join(non_alphanumeric_content), len(non_alphanumeric_content)
    )
    assert non_alphanumeric_metrics.alphanumeric_characters == 0
