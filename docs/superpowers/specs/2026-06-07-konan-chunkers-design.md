# konan — Fast Rust Chunkers with Pythonic Bindings

**Date:** 2026-06-07
**Status:** Approved

## Purpose

konan is a standalone, generic text-chunking library: a pure-Rust core implementing
fast chunking strategies, wrapped for Python via PyO3/maturin. It ports the three
chunkers from the holmes `chunker.py` (naive, fixed-size, recursive) and adds
advanced strategies (sentence, markdown, token, semantic). It exposes rayon-based
native multithreading for batch chunking and real `async def` flavours of every
operation.

Non-goals: holmes domain model compatibility (no `document_id`/`page_id`/`uri`
fields — holmes can wrap konan trivially), embedded local embedding models
(fastembed/candle adapters may come later).

## Architecture

Cargo workspace with a hexagonal split:

```
konan/
├── Cargo.toml              # workspace: members = ["crates/konan-core", "crates/konan-py"]
├── crates/
│   ├── konan-core/         # pure Rust, zero PyO3, crates.io-publishable
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── chunk.rs            # Chunk struct
│   │       ├── chunker.rs          # Chunker trait (port) + chunk_many (rayon)
│   │       ├── error.rs            # KonanError
│   │       ├── embedder.rs         # Embedder trait (port) + OpenAIEmbedder adapter
│   │       ├── semantic.rs         # SemanticChunker<E: Embedder>
│   │       └── strategies/
│   │           ├── naive.rs
│   │           ├── fixed_size.rs
│   │           ├── recursive.rs
│   │           ├── sentence.rs
│   │           ├── markdown.rs
│   │           └── token.rs
│   └── konan-py/           # PyO3 bindings, cdylib, built by maturin
│       └── src/            # lib.rs + per-class binding modules
├── python/konan/           # __init__.py, py.typed, __init__.pyi stubs
├── pyproject.toml          # maturin build-backend, manifest-path = crates/konan-py/Cargo.toml
├── tests/                  # pytest (incl. pytest-asyncio) against built extension
├── logo.png
└── README.md
```

The existing `main.py` is deleted; `pyproject.toml` is rewritten for maturin.
Development flow stays uv-native: `uv run maturin develop`.

## Core (`konan-core`)

### Chunk

```rust
pub struct Chunk {
    pub text: String,
    pub start: usize,   // char offset in source (not bytes)
    pub end: usize,     // char offset in source
    pub index: usize,   // 0-based chunk index
    pub hash: String,   // content hash (xxh3 or sha256 of text)
}
```

Offsets are **character positions** computed during chunking — not recovered
afterwards via `source.find()` backtracking as in the original Python (which was
fragile with repeated substrings).

### Chunker trait (port)

```rust
pub trait Chunker: Send + Sync {
    fn chunk(&self, text: &str) -> Result<Vec<Chunk>, KonanError>;
}

pub fn chunk_many<C: Chunker>(chunker: &C, texts: &[String]) -> Result<Vec<Vec<Chunk>>, KonanError>
// rayon par_iter over texts
```

Empty input → empty `Vec` (no "null" placeholder chunk; that was holmes
pipeline cruft).

### Ported strategies

| Strategy | Config | Behavior |
|---|---|---|
| `NaiveChunker` | `chunk_size` (words, default 200) | Whitespace split, fixed word groups. May break mid-sentence. |
| `FixedSizeChunker` | `chunk_size` (chars, 1000), `chunk_overlap` (200), `respect_sentences` (true) | Character-based with overlap; tries sentence boundaries (`[.!?]+\s+`), falls back to word boundaries. |
| `RecursiveChunker` | `chunk_size` (1000), `chunk_overlap` (200), `separators` (default `["\n\n", "\n", " ", ".", ",", ""]`) | Separator-hierarchy recursion, LangChain-compatible semantics. |

### New strategies

| Strategy | Config | Behavior |
|---|---|---|
| `SentenceChunker` | `max_chars` (1000), `overlap_sentences` (1) | `unicode-segmentation` sentence boundaries, groups sentences into chunks, sentence-level overlap. |
| `MarkdownChunker` | `chunk_size` (1000), `chunk_overlap` (200) | `pulldown-cmark` event stream; splits along structure (headings, paragraphs, lists); never splits fenced code blocks; prepends heading-breadcrumb context (`# A > ## B`) to each chunk. |
| `TokenChunker` | `chunk_size` (tokens, 512), `chunk_overlap` (64), `encoding` (`"cl100k_base"` \| `"o200k_base"`) | `tiktoken-rs` token-exact chunking for embedding-model limits. |

### Semantic chunking (port/adapter)

```rust
#[async_trait]
pub trait Embedder: Send + Sync {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, KonanError>;
}

pub struct OpenAIEmbedder { base_url, api_key, model, batch_size }
// POST {base_url}/embeddings (OpenAI-compatible), reqwest, batched

pub struct SemanticChunker<E: Embedder> {
    threshold: Option<f32>,        // absolute cosine-similarity cut
    percentile: Option<f32>,       // or distance-percentile cut (default p95)
    min_chunk_size: usize,
    max_chunk_size: usize,
}
```

Algorithm: split into sentences → embed all sentences (batched) → compute cosine
similarity between adjacent sentences → break where similarity drops below
threshold (or where distance exceeds the configured percentile) → enforce
min/max chunk sizes.

The `Embedder` port is injectable from outside; future adapters (fastembed,
candle) implement the same trait.

## Python API (`konan-py`)

Every strategy class exposes the same four methods:

```python
from konan import RecursiveChunker, SemanticChunker, OpenAIEmbedder

c = RecursiveChunker(chunk_size=1000, chunk_overlap=200)
c.chunk(text)                      # list[Chunk], sync
c.chunk_many(texts)                # list[list[Chunk]], rayon-parallel, GIL released
await c.chunk_async(text)          # real async def (tokio blocking pool)
await c.chunk_many_async(texts)    # async + parallel

emb = OpenAIEmbedder(base_url="http://localhost:8000/v1", api_key="...", model="...")
s = SemanticChunker(embedder=emb, threshold=0.75)
await s.chunk_async(text)          # natively async (HTTP)
s.chunk(text)                      # sync wrapper (runs tokio runtime internally)
```

- Async via `pyo3-async-runtimes` (tokio); methods are coroutine-returning,
  exported so they type as `async def` in stubs.
- CPU-bound strategies release the GIL in `chunk_many` (`py.allow_threads`)
  and run on tokio's blocking pool for async variants.
- **Python embedder adapter:** `SemanticChunker(embedder=...)` also accepts any
  Python async callable `async def (list[str]) -> list[list[float]]`, wrapped
  in a `PyEmbedder` adapter implementing the `Embedder` trait.
- `Chunk` is a frozen pyclass: `.text`, `.start`, `.end`, `.index`, `.hash`,
  with `__repr__`, `__len__` (chars), `__eq__`/`__hash__`.
- Full `.pyi` stubs + `py.typed` shipped in `python/konan/`.

## Error handling

- `KonanError` enum in core: `InvalidConfig` (e.g. `chunk_overlap >= chunk_size`),
  `Embedding` (HTTP status/network/deserialization), `Tokenizer` (unknown encoding).
- Mapping in bindings: `InvalidConfig` → `ValueError`; `Embedding` →
  `konan.EmbeddingError` (custom exception); `Tokenizer` → `ValueError`.
- Config validated at construction time (constructor raises), not at chunk time.

## Testing

- **Rust unit tests** per strategy: offset correctness (chunk text equals
  `source[start..end]` by chars), overlap honored, UTF-8/emoji/CJK safety,
  empty input, single-word-larger-than-chunk edge cases.
- **Rust tests for semantic**: mocked `Embedder` impl (no network).
- **pytest integration**: built extension via `maturin develop`; sync + parallel
  + async variants; `pytest-asyncio`; mocked OpenAI-compatible endpoint for
  `OpenAIEmbedder`; Python-callable embedder round-trip.

## README

Modern README: centered `logo.png`, badges (PyPI, CI, license, Rust),
one-line pitch with Naruto flavor ("paper chakra — folds documents into perfect
pieces"), feature table, install (`uv add konan` / `pip install konan`),
quickstart, per-strategy examples, async + parallel examples, semantic chunker
with OpenAI-compatible endpoint example, benchmark table placeholder,
development section (uv + maturin), license.

## Dependencies

Core: `rayon`, `regex`, `unicode-segmentation`, `pulldown-cmark`, `tiktoken-rs`,
`reqwest` (json, rustls), `tokio`, `async-trait`, `serde`/`serde_json`,
`thiserror`, `xxhash-rust` (or `sha2`).
Bindings: `pyo3` (abi3-py312), `pyo3-async-runtimes` (tokio runtime).
Python dev: `maturin`, `pytest`, `pytest-asyncio`.
