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


UNICODE_WORDS = (
    "schön übermäßig größer données déjà café naïve 日本語 文章 分割 "
    "处理 中文 текст разбиение 😀 🚀 emoji καλημέρα κόσμος"
).split() + WORDS


def make_unicode_prose(n_bytes: int, seed: int = 99) -> str:
    """Mixed-script prose (~35% non-ASCII words) to bench the exact
    char-offset path instead of the ASCII fast path."""
    rng = random.Random(seed)
    parts: list[str] = []
    size = 0
    while size < n_bytes:
        words = [rng.choice(UNICODE_WORDS) for _ in range(rng.randint(6, 18))]
        sentence = " ".join(words).capitalize() + rng.choice(". . . ! ?".split())
        parts.append(sentence + " ")
        size += len(parts[-1].encode())
        if rng.random() < 0.08:
            parts.append("\n\n")
            size += 2
    return "".join(parts)


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
    results: list[tuple[str, str, float, int]] = []
    print(f"\n### Throughput per strategy ({n / 1e6:.0f} MB document)\n")
    print("| Chunker | Config | Throughput | Chunks |")
    print("|---|---|---:|---:|")
    for name, config, chunker, text in strategies:
        seconds = time_call(lambda c=chunker, t=text: c.chunk(t))
        chunks = len(chunker.chunk(text))
        mbps = mb_per_s(len(text.encode()), seconds)
        results.append((name, config, mbps, chunks))
        print(f"| `{name}` | {config} | {mbps:,.0f} MB/s | {chunks} |")
    return results


def bench_parallel(prose_docs: list[str]) -> dict:
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
    return {
        "seq_mbps": mb_per_s(total, seq),
        "par_mbps": mb_per_s(total, par),
        "speedup": seq / par,
    }


def bench_competitors(prose: str) -> list[tuple[str, str, float | None, str]]:
    n = len(prose.encode())
    # (strategy, library, MB/s or None on failure, chunks-or-error note)
    rows: list[tuple[str, str, float | None, str]] = []

    def measure(strategy: str, library: str, make: Callable[[], Callable[[], object]]) -> None:
        """`make` builds the chunker (untimed) and returns the timed call."""
        try:
            fn = make()
            seconds = time_call(fn)
            chunks = len(fn())  # type: ignore[arg-type]
            rows.append((strategy, library, mb_per_s(n, seconds), str(chunks)))
        except Exception as exc:  # missing dep, API drift — report, don't crash
            rows.append((strategy, library, None, type(exc).__name__))

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

    def sts_recursive():
        from semantic_text_splitter import TextSplitter

        splitter = TextSplitter(1000, overlap=200)
        return lambda: splitter.chunks(prose)

    measure("recursive", "semantic-text-splitter", sts_recursive)

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

    def sts_token():
        from semantic_text_splitter import TextSplitter

        # gpt-3.5-turbo => cl100k_base
        splitter = TextSplitter.from_tiktoken_model("gpt-3.5-turbo", 512, overlap=64)
        return lambda: splitter.chunks(prose)

    measure("token", "semantic-text-splitter", sts_token)

    def semchunk_token():
        import semchunk

        chunker = semchunk.chunkerify("cl100k_base", 512)
        try:  # overlap support varies by semchunk version
            chunker(prose[:512], overlap=64)
            return lambda: chunker(prose, overlap=64)
        except TypeError:
            return lambda: chunker(prose)

    measure("token", "semchunk", semchunk_token)

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

    # --- recursive on mixed-script text: konan's exact (non-ASCII) path --
    unicode_prose = make_unicode_prose(n)

    def konan_recursive_unicode():
        chunker = konan.RecursiveChunker(chunk_size=1000, chunk_overlap=200)
        return lambda: chunker.chunk(unicode_prose)

    def sts_recursive_unicode():
        from semantic_text_splitter import TextSplitter

        splitter = TextSplitter(1000, overlap=200)
        return lambda: splitter.chunks(unicode_prose)

    def chonkie_recursive_unicode():
        from chonkie import RecursiveChunker

        chunker = RecursiveChunker(chunk_size=1000)
        return lambda: chunker.chunk(unicode_prose)

    n_unicode = len(unicode_prose.encode())

    def measure_unicode(library: str, make: Callable[[], Callable[[], object]]) -> None:
        try:
            fn = make()
            seconds = time_call(fn)
            chunks = len(fn())  # type: ignore[arg-type]
            rows.append(("recursive (unicode)", library, mb_per_s(n_unicode, seconds), str(chunks)))
        except Exception as exc:
            rows.append(("recursive (unicode)", library, None, type(exc).__name__))

    measure_unicode("konan", konan_recursive_unicode)
    measure_unicode("semantic-text-splitter", sts_recursive_unicode)
    measure_unicode("chonkie", chonkie_recursive_unicode)

    print(f"\n### vs other libraries ({n / 1e6:.0f} MB document, higher is better)\n")
    print("| Strategy | Library | Throughput | Chunks |")
    print("|---|---|---:|---:|")
    for strategy, library, mbps, note in rows:
        marker = " **(konan)**" if library == "konan" else ""
        result = f"{mbps:,.1f} MB/s" if mbps is not None else f"n/a ({note})"
        chunks = note if mbps is not None else "—"
        print(f"| {strategy} | {library}{marker} | {result} | {chunks} |")
    print(
        "\n_Configs: recursive = 1000 chars / 200 overlap everywhere; token = "
        "cl100k_base, 512 / 64 everywhere; sentence configs are not directly "
        "comparable (konan groups by chars, chonkie by tokens) — read those two "
        "rows as per-library sentence-chunking cost, not head-to-head._"
    )
    return rows


KONAN_COLOR = "#f97316"
OTHER_COLOR = "#94a3b8"


def make_plots(
    strategy_results: list[tuple[str, str, float, int]],
    parallel: dict,
    comp_rows: list[tuple[str, str, float | None, str]],
    outdir: str,
) -> list[str]:
    """Render SVG charts of the results. Returns written paths; empty if
    matplotlib is unavailable (bench extra not installed)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    written: list[str] = []

    def style(ax, title: str):
        ax.set_title(title, fontsize=12, fontweight="bold", loc="left", pad=12)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(left=False)
        ax.xaxis.grid(True, color="#e2e8f0", linewidth=0.8)
        ax.set_axisbelow(True)

    def bar_labels(ax, bars, values, suffix=" MB/s"):
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_width() + max(values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{v:,.0f}{suffix}",
                va="center",
                fontsize=9,
            )

    # --- 1. throughput per strategy -------------------------------------
    names = [name for name, _, _, _ in strategy_results][::-1]
    values = [mbps for _, _, mbps, _ in strategy_results][::-1]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    bars = ax.barh(names, values, color=KONAN_COLOR, height=0.62)
    bar_labels(ax, bars, values)
    style(ax, "konan throughput per strategy (1 MB document, MB/s)")
    ax.set_xlim(0, max(values) * 1.28)
    fig.tight_layout()
    path = f"{outdir}/throughput.svg"
    fig.savefig(path)
    plt.close(fig)
    written.append(path)

    # --- 2. library comparison (grouped) ---------------------------------
    groups: dict[str, list[tuple[str, float]]] = {}
    for strategy, library, mbps, _ in comp_rows:
        if mbps is not None:
            groups.setdefault(strategy, []).append((library, mbps))
    labels: list[str] = []
    vals: list[float] = []
    colors: list[str] = []
    ypos: list[float] = []
    group_label_pos: list[tuple[float, str]] = []
    y = 0.0
    for strategy, entries in groups.items():
        group_label_pos.append((y - 0.1, strategy))
        for library, mbps in sorted(entries, key=lambda e: e[1]):
            labels.append(library)
            vals.append(mbps)
            colors.append(KONAN_COLOR if library == "konan" else OTHER_COLOR)
            ypos.append(y)
            y += 1.0
        y += 0.9  # gap between groups
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(vals) + 1.6))
    bars = ax.barh(ypos, vals, color=colors, height=0.78)
    ax.set_yticks(ypos, labels, fontsize=9)
    for gy, gname in group_label_pos:
        ax.text(0, gy - 0.45, gname, fontsize=10, fontweight="bold", color="#334155")
    bar_labels(ax, bars, vals)
    ax.invert_yaxis()
    style(ax, "konan vs other libraries (1 MB document, MB/s, higher is better)")
    ax.set_xlim(0, max(vals) * 1.28)
    fig.tight_layout()
    path = f"{outdir}/comparison.svg"
    fig.savefig(path)
    plt.close(fig)
    written.append(path)

    # --- 3. parallel scaling ---------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 1.9))
    modes = ["chunk_many() — rayon", "sequential chunk() loop"]
    pvals = [parallel["par_mbps"], parallel["seq_mbps"]]
    bars = ax.barh(modes, pvals, color=[KONAN_COLOR, OTHER_COLOR], height=0.58)
    bar_labels(ax, bars, pvals)
    style(
        ax,
        f"chunk_many parallel scaling — {parallel['speedup']:.1f}× "
        "(64 × 256 KB docs, RecursiveChunker, MB/s)",
    )
    ax.set_xlim(0, max(pvals) * 1.22)
    fig.tight_layout()
    path = f"{outdir}/parallel.svg"
    fig.savefig(path)
    plt.close(fig)
    written.append(path)

    return written


def main() -> None:
    print(f"_Benchmarked on {hardware_line()}._")
    prose = make_prose(CORPUS_BYTES)
    markdown = make_markdown(CORPUS_BYTES)
    docs = [make_prose(PARALLEL_DOC_BYTES, seed=i) for i in range(PARALLEL_DOCS)]

    strategy_results = bench_strategies(prose, markdown)
    parallel = bench_parallel(docs)
    comp_rows = bench_competitors(prose)

    import os

    outdir = os.path.dirname(os.path.abspath(__file__))
    plots = make_plots(strategy_results, parallel, comp_rows, outdir)
    if plots:
        print("\nPlots written:")
        for p in plots:
            print(f"  {p}")
    else:
        print("\n(matplotlib not installed — skipped plots; use --extra bench)")


if __name__ == "__main__":
    main()
