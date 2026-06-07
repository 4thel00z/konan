<p align="center">
  <img src="logo.png" alt="konan logo" width="280" />
</p>

<h1 align="center">konan</h1>

<p align="center">
  <em>Like the paper angel of the Akatsuki, konan folds your documents into perfect pieces — blazingly fast chunkers written in Rust, wrapped for pythonic bliss.</em>
</p>

<p align="center">
  <a href="#installation"><img alt="Python 3.12+" src="https://img.shields.io/badge/python-3.12%2B-blue"></a>
  <img alt="Rust" src="https://img.shields.io/badge/core-rust-orange">
  <img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green">
</p>

---

## Why konan?

- 🦀 **Rust core** — all chunking runs in native code, no Python-loop overhead
- ⚡ **Multithreaded out of the box** — `chunk_many()` fans out across all cores via rayon and releases the GIL
- 🌀 **First-class async** — every chunker ships real `async def` flavours (`chunk_async`, `chunk_many_async`)
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

### Parallel — all cores, zero setup

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

## The `Chunk` object

```python
chunk.text    # the chunk's text
chunk.start   # char offset into the source (Python slicing semantics)
chunk.end     # char offset, exclusive
chunk.index   # 0-based position
chunk.hash    # xxh3-64 content hash
```

## Benchmarks

_Coming soon — rayon goes brrr._

## Development

```bash
uv sync                       # set up the venv (builds the extension)
uv run maturin develop --uv   # rebuild after Rust changes
cargo test -p konan-core      # rust unit tests
uv run pytest -q              # python integration tests
```

The workspace is hexagonal: [`crates/konan-core`](crates/konan-core) is pure
Rust (no PyO3) with `Chunker` and `Embedder` ports;
[`crates/konan-py`](crates/konan-py) adapts it to Python.

## License

MIT

---

<p align="center"><sub>Named after Konan of the Akatsuki — the only one who could fold paper into anything. 🗞️</sub></p>
