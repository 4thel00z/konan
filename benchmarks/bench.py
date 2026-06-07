"""konan benchmarks — reproduce the README table.

Run:
    uv run --extra bench benchmarks/bench.py

Without the extra, the library-comparison section is skipped:
    uv run benchmarks/bench.py

Prints ready-to-paste markdown. Throughput is decimal MB/s (1 MB = 1e6
bytes of UTF-8). Times are the median of REPEATS runs.
"""

from __future__ import annotations

import platform
import random
import statistics
import time
from collections.abc import Callable

import konan

REPEATS = 5
CORPUS_BYTES = 1_000_000
PARALLEL_DOCS = 64
PARALLEL_DOC_BYTES = 256_000

WORDS = (
    "the quick brown fox jumps over lazy dog while seventy wizards "
    "briskly mix jugs of liquid oxygen under pale moonlight and every "
    "sphinx of black quartz judges vows with quiet certainty because "
    "language models prefer well chunked context windows full of prose"
).split()


def make_prose(n_bytes: int, seed: int = 42) -> str:
    rng = random.Random(seed)
    parts: list[str] = []
    size = 0
    while size < n_bytes:
        sentence_words = [rng.choice(WORDS) for _ in range(rng.randint(6, 18))]
        sentence = " ".join(sentence_words).capitalize() + rng.choice(". . . ! ?".split())
        parts.append(sentence + " ")
        size += len(parts[-1])
        if rng.random() < 0.08:
            parts.append("\n\n")
            size += 2
    return "".join(parts)[:n_bytes]


def make_markdown(n_bytes: int, seed: int = 7) -> str:
    rng = random.Random(seed)
    parts: list[str] = []
    size = 0
    section = 0
    while size < n_bytes:
        section += 1
        parts.append(f"# Section {section}\n\n")
        for sub in range(rng.randint(1, 3)):
            parts.append(f"## Topic {section}.{sub + 1}\n\n")
            parts.append(make_prose(rng.randint(400, 1200), seed=rng.randint(0, 9999)) + "\n\n")
            if rng.random() < 0.3:
                parts.append("```python\nfor i in range(10):\n    print(i)\n```\n\n")
        size = sum(len(p) for p in parts)
    return "".join(parts)[:n_bytes]


def time_call(fn: Callable[[], object]) -> float:
    """Median wall-clock seconds over REPEATS runs (1 warmup)."""
    fn()
    samples = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def mb_per_s(n_bytes: int, seconds: float) -> float:
    return n_bytes / 1e6 / seconds


def hardware_line() -> str:
    machine = platform.machine()
    try:
        import subprocess

        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        chip = platform.processor() or machine
    return (
        f"{chip} ({machine}), Python {platform.python_version()}, "
        f"konan {konan.__version__}"
    )


def bench_strategies(prose: str, markdown: str) -> None:
    n = len(prose.encode())
    strategies = [
        ("NaiveChunker", "chunk_size=200 words", konan.NaiveChunker(chunk_size=200), prose),
        (
            "FixedSizeChunker",
            "1000 chars, 200 overlap",
            konan.FixedSizeChunker(chunk_size=1000, chunk_overlap=200),
            prose,
        ),
        (
            "RecursiveChunker",
            "1000 chars, 200 overlap",
            konan.RecursiveChunker(chunk_size=1000, chunk_overlap=200),
            prose,
        ),
        (
            "SentenceChunker",
            "1000 chars, 1 overlap",
            konan.SentenceChunker(max_chars=1000, overlap_sentences=1),
            prose,
        ),
        (
            "MarkdownChunker",
            "1000 chars, 200 overlap",
            konan.MarkdownChunker(chunk_size=1000, chunk_overlap=200),
            markdown,
        ),
        (
            "TokenChunker",
            "512 tokens, 64 overlap (cl100k)",
            konan.TokenChunker(chunk_size=512, chunk_overlap=64),
            prose,
        ),
    ]
    print(f"\n### Throughput per strategy ({n / 1e6:.0f} MB document)\n")
    print("| Chunker | Config | Throughput | Chunks |")
    print("|---|---|---:|---:|")
    for name, config, chunker, text in strategies:
        seconds = time_call(lambda c=chunker, t=text: c.chunk(t))
        chunks = len(chunker.chunk(text))
        print(f"| `{name}` | {config} | {mb_per_s(len(text.encode()), seconds):,.0f} MB/s | {chunks} |")


def bench_parallel(prose_docs: list[str]) -> None:
    chunker = konan.RecursiveChunker(chunk_size=1000, chunk_overlap=200)
    total = sum(len(d.encode()) for d in prose_docs)
    seq = time_call(lambda: [chunker.chunk(d) for d in prose_docs])
    par = time_call(lambda: chunker.chunk_many(prose_docs))
    print(
        f"\n### Parallel scaling — `chunk_many` "
        f"({len(prose_docs)} docs × {len(prose_docs[0]) / 1e3:.0f} KB, RecursiveChunker)\n"
    )
    print("| Mode | Time | Throughput | Speedup |")
    print("|---|---:|---:|---:|")
    print(f"| sequential `chunk()` loop | {seq * 1e3:,.1f} ms | {mb_per_s(total, seq):,.0f} MB/s | 1.0× |")
    print(
        f"| `chunk_many()` (rayon) | {par * 1e3:,.1f} ms | "
        f"{mb_per_s(total, par):,.0f} MB/s | {seq / par:.1f}× |"
    )


def bench_competitors(prose: str) -> None:
    n = len(prose.encode())
    rows: list[tuple[str, str, str, str]] = []  # (strategy, library, result, chunks)

    def measure(strategy: str, library: str, make: Callable[[], Callable[[], object]]) -> None:
        """`make` builds the chunker (untimed) and returns the timed call."""
        try:
            fn = make()
            seconds = time_call(fn)
            chunks = len(fn())  # type: ignore[arg-type]
            rows.append((strategy, library, f"{mb_per_s(n, seconds):,.1f} MB/s", str(chunks)))
        except Exception as exc:  # missing dep, API drift — report, don't crash
            rows.append((strategy, library, f"n/a ({type(exc).__name__})", "—"))

    # --- recursive: chars for all three ---------------------------------
    def konan_recursive():
        chunker = konan.RecursiveChunker(chunk_size=1000, chunk_overlap=200)
        return lambda: chunker.chunk(prose)

    measure("recursive", "konan", konan_recursive)

    def lc_recursive():
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return lambda: splitter.split_text(prose)

    measure("recursive", "langchain-text-splitters", lc_recursive)

    def chonkie_recursive():
        from chonkie import RecursiveChunker

        chunker = RecursiveChunker(chunk_size=1000)
        return lambda: chunker.chunk(prose)

    measure("recursive", "chonkie", chonkie_recursive)

    # --- token: cl100k_base, 512/64 for all three -----------------------
    def konan_token():
        chunker = konan.TokenChunker(chunk_size=512, chunk_overlap=64)
        return lambda: chunker.chunk(prose)

    measure("token", "konan", konan_token)

    def lc_token():
        from langchain_text_splitters import TokenTextSplitter

        splitter = TokenTextSplitter(encoding_name="cl100k_base", chunk_size=512, chunk_overlap=64)
        return lambda: splitter.split_text(prose)

    measure("token", "langchain-text-splitters", lc_token)

    def chonkie_token():
        import tiktoken
        from chonkie import TokenChunker

        chunker = TokenChunker(
            tokenizer=tiktoken.get_encoding("cl100k_base"), chunk_size=512, chunk_overlap=64
        )
        return lambda: chunker.chunk(prose)

    measure("token", "chonkie", chonkie_token)

    # --- sentence: konan vs chonkie (configs differ; see caveat) --------
    def konan_sentence():
        chunker = konan.SentenceChunker(max_chars=1000, overlap_sentences=0)
        return lambda: chunker.chunk(prose)

    measure("sentence", "konan", konan_sentence)

    def chonkie_sentence():
        from chonkie import SentenceChunker

        chunker = SentenceChunker(chunk_size=512)
        return lambda: chunker.chunk(prose)

    measure("sentence", "chonkie", chonkie_sentence)

    print(f"\n### vs other libraries ({n / 1e6:.0f} MB document, higher is better)\n")
    print("| Strategy | Library | Throughput | Chunks |")
    print("|---|---|---:|---:|")
    for strategy, library, result, chunks in rows:
        marker = " **(konan)**" if library == "konan" else ""
        print(f"| {strategy} | {library}{marker} | {result} | {chunks} |")
    print(
        "\n_Configs: recursive = 1000 chars / 200 overlap everywhere; token = "
        "cl100k_base, 512 / 64 everywhere; sentence configs are not directly "
        "comparable (konan groups by chars, chonkie by tokens) — read those two "
        "rows as per-library sentence-chunking cost, not head-to-head._"
    )


def main() -> None:
    print(f"_Benchmarked on {hardware_line()}._")
    prose = make_prose(CORPUS_BYTES)
    markdown = make_markdown(CORPUS_BYTES)
    docs = [make_prose(PARALLEL_DOC_BYTES, seed=i) for i in range(PARALLEL_DOCS)]

    bench_strategies(prose, markdown)
    bench_parallel(docs)
    bench_competitors(prose)


if __name__ == "__main__":
    main()
