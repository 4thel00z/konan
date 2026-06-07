<p align="center">
  <img src="https://raw.githubusercontent.com/4thel00z/konan/master/logo.png" alt="konan logo" width="280" />
</p>

<h1 align="center">konan</h1>

<p align="center">
  <em>Like the paper angel of the Akatsuki, konan folds your documents into precise pieces — Rust chunkers behind a Python API.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/konan/"><img alt="PyPI" src="https://img.shields.io/pypi/v/konan"></a>
  <a href="https://github.com/4thel00z/konan/actions/workflows/ci.yaml"><img alt="CI" src="https://github.com/4thel00z/konan/actions/workflows/ci.yaml/badge.svg"></a>
  <a href="https://github.com/4thel00z/konan/actions/workflows/python-ci.yml"><img alt="python-ci" src="https://github.com/4thel00z/konan/actions/workflows/python-ci.yml/badge.svg"></a>
  <a href="#installation"><img alt="Python 3.12+" src="https://img.shields.io/badge/python-3.12%2B-blue"></a>
  <img alt="Rust" src="https://img.shields.io/badge/core-rust-orange">
  <img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green">
</p>

---

## Why konan?

- 🦀 **Rust core** — all chunking runs in native code, no Python-loop overhead
- ⚡ **Multithreaded by default** — `chunk_many()` fans out across all cores via rayon and releases the GIL
- 🌀 **Real async** — `chunk_async` / `chunk_many_async` are native `async def`, not thread-pool wrappers
- 🎯 **Char-accurate offsets** — `text[chunk.start:chunk.end] == chunk.text`, always (Python slicing semantics, emoji-safe)
- 🧠 **Semantic chunking** — splits on topic shifts using any OpenAI-compatible embeddings endpoint, or your own async embedder
- 🔌 **Ports & adapters** — the `Embedder` port is injectable; bring your own backend

## Strategies

| Chunker | Splits by | Best for |
|---|---|---|
| `NaiveChunker` | fixed word count | quick & dirty baselines |
| `FixedSizeChunker` | chars, sentence-aware, overlap | classic RAG pipelines |
| `RecursiveChunker` | separator hierarchy (`\n\n` → `\n` → … ) | general text, LangChain-compatible |
| `SentenceChunker` | unicode sentence boundaries | prose, multilingual text |
| `MarkdownChunker` | document structure + heading breadcrumbs | docs, wikis, READMEs |
| `TokenChunker` | exact token counts (`cl100k_base`, `o200k_base`) | embedding-model token limits |
| `SemanticChunker` | embedding similarity drops | topic-coherent chunks |

## Installation

```bash
uv add konan        # or: pip install konan
```

## Quickstart

```python
from konan import RecursiveChunker

chunker = RecursiveChunker(chunk_size=1000, chunk_overlap=200)

chunks = chunker.chunk(open("moby_dick.txt").read())
print(chunks[0].text, chunks[0].start, chunks[0].end, chunks[0].hash)
```

### Parallel — all cores, one call

```python
# rayon work-stealing across every core, GIL released:
all_chunks = chunker.chunk_many(documents)
```

### Async — real `async def`, no thread-pool wrappers

```python
chunks = await chunker.chunk_async(text)
batches = await chunker.chunk_many_async(documents)
```

### Markdown with breadcrumbs

```python
from konan import MarkdownChunker

chunks = MarkdownChunker(chunk_size=800).chunk(readme_text)
# chunk text is prefixed with its heading trail: "# Guide > ## Install\n\n..."
# code fences are never split
```

### Token-exact chunks

```python
from konan import TokenChunker

chunker = TokenChunker(chunk_size=512, chunk_overlap=64, encoding="o200k_base")
```

### Semantic chunking

Point it at any OpenAI-compatible `/embeddings` endpoint (OpenAI, vLLM,
Ollama, LiteLLM, …):

```python
from konan import OpenAIEmbedder, SemanticChunker

embedder = OpenAIEmbedder(
    base_url="https://api.openai.com/v1",
    model="text-embedding-3-small",
    api_key="sk-...",
    batch_size=128,    # texts per request
    timeout=30.0,      # request timeout, seconds
    max_retries=2,     # exponential backoff on 429/5xx/connect errors
    dimensions=512,    # optional: shorten text-embedding-3-* vectors
)
chunker = SemanticChunker(embedder=embedder, threshold=0.75)
chunks = await chunker.chunk_async(article)
```

Or inject your own embedder — any async callable works:

```python
async def my_embedder(texts: list[str]) -> list[list[float]]:
    return await my_model.embed(texts)

chunker = SemanticChunker(embedder=my_embedder, percentile=95.0)
chunks = await chunker.chunk_async(article)   # async-only for Python embedders
```

> Python embedders must return `list[list[float]]` — call `.tolist()` on
> numpy arrays. They are async-only: `chunk()`/`chunk_many()` raise a
> `RuntimeError` pointing you at the `_async` variants.

## The `Chunk` object

```python
chunk.text    # the chunk's text
chunk.start   # char offset into the source (Python slicing semantics)
chunk.end     # char offset, exclusive
chunk.index   # 0-based position
chunk.hash    # xxh3-64 content hash
```

## Benchmarks

_Benchmarked on Apple M3 Pro (arm64), Python 3.12.10. Decimal MB/s, median
of 5 runs, measured from Python (the numbers you actually get). Reproduce
both tables and plots with `uv run --extra bench benchmarks/bench.py`;
Rust-level criterion benches: `cargo bench -p konan-core`._

### vs other libraries (same 1 MB document, identical configs)

<p align="center">
  <img src="https://raw.githubusercontent.com/4thel00z/konan/master/benchmarks/comparison.svg" alt="konan vs other libraries" width="760" />
</p>

| Strategy | Library | Throughput | Chunks |
|---|---|---:|---:|
| recursive | **konan** | 282 MB/s | 1298 |
| recursive | semantic-text-splitter | 129 MB/s | 1368 |
| recursive | chonkie | 87 MB/s | 1413 |
| recursive | langchain-text-splitters | 24 MB/s | 1468 |
| token | **konan** | 68 MB/s | 438 |
| token | semchunk | 24 MB/s | 616 |
| token | langchain-text-splitters | 23 MB/s | 438 |
| token | chonkie | 18 MB/s | 438 |
| token | semantic-text-splitter | 4 MB/s | 505 |
| sentence | **konan** | 305 MB/s | 1042 |
| sentence | chonkie | 3 MB/s | 2124 |
| recursive (unicode) | semantic-text-splitter | 167 MB/s | 1150 |
| recursive (unicode) | **konan** | 166 MB/s | 1092 |
| recursive (unicode) | chonkie | 57 MB/s | 1171 |

### Throughput per strategy (1 MB document)

<p align="center">
  <img src="https://raw.githubusercontent.com/4thel00z/konan/master/benchmarks/throughput.svg" alt="konan throughput per strategy" width="760" />
</p>

| Chunker | Config | Throughput | Chunks |
|---|---|---:|---:|
| `NaiveChunker` | 200 words | 553 MB/s | 804 |
| `FixedSizeChunker` | 1000 chars, 200 overlap | 1,602 MB/s | 1255 |
| `RecursiveChunker` | 1000 chars, 200 overlap | 282 MB/s | 1298 |
| `SentenceChunker` | 1000 chars, 1 overlap | 306 MB/s | 1130 |
| `MarkdownChunker` | 1000 chars, 200 overlap | 448 MB/s | 1511 |
| `TokenChunker` | 512 tokens, 64 overlap (cl100k) | 69 MB/s | 438 |

### Parallel scaling — rayon goes brrr

<p align="center">
  <img src="https://raw.githubusercontent.com/4thel00z/konan/master/benchmarks/parallel.svg" alt="chunk_many parallel scaling" width="760" />
</p>

64 docs × 256 KB through `RecursiveChunker`:

| Mode | Time | Throughput | Speedup |
|---|---:|---:|---:|
| sequential `chunk()` loop | 61 ms | 267 MB/s | 1.0× |
| `chunk_many()` (rayon, GIL released) | 10 ms | 1,568 MB/s | **5.9×** |

(Pure-Rust criterion puts the same workload at ~6 GiB/s; the Python numbers
include chunk-object conversion.)

Caveats, honestly: recursive/token use identical configs across libraries
(1000 chars / 200 overlap; cl100k, 512 / 64 — note the identical token chunk
counts for the libraries with the same windowing semantics). Sentence
configs are not directly comparable (konan groups by chars, chonkie by
tokens) — read those rows as per-library cost, not head-to-head. The
`recursive (unicode)` rows run mixed German/CJK/Cyrillic/emoji prose, off
konan's ASCII fast path. konan tokenizes with
[`bpe-openai`](https://crates.io/crates/bpe-openai) (tiktoken-equivalent
output, much faster encoder).

## Development

```bash
uv sync                       # set up the venv (builds the extension)
uv run maturin develop --uv   # rebuild after Rust changes
cargo test --workspace        # rust unit tests
uv run pytest -q              # python integration tests
```

The workspace is hexagonal: [`crates/konan-core`](crates/konan-core) is pure
Rust (no PyO3) with `Chunker` and `Embedder` ports;
[`crates/konan-py`](crates/konan-py) adapts it to Python.

## License

MIT

---

<p align="center"><sub>Named after Konan of the Akatsuki — the only one who could fold paper into anything. 🗞️</sub></p>
