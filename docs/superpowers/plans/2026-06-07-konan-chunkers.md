# konan Chunkers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build konan — a Rust workspace (`konan-core` pure Rust + `konan-py` PyO3 bindings) exposing 7 fast chunking strategies to Python with rayon parallelism and real `async def` methods, plus a modern README.

**Architecture:** Hexagonal: `konan-core` has zero PyO3 and defines `Chunker` and `Embedder` ports; strategies produce byte spans converted to char-offset `Chunk`s via a shared `OffsetMap`. `konan-py` wraps core types in pyclasses; sync batch methods release the GIL and use rayon, async methods use `pyo3-async-runtimes` + tokio. Spec: `docs/superpowers/specs/2026-06-07-konan-chunkers-design.md`.

**Tech Stack:** Rust 2021, pyo3 0.25 (abi3-py312), pyo3-async-runtimes 0.25, rayon, reqwest (rustls), tiktoken-rs, pulldown-cmark, unicode-segmentation, maturin, uv, pytest + pytest-asyncio.

**Conventions for all tasks:**
- Rust tests live in `#[cfg(test)] mod tests` at the bottom of each file shown.
- Run core tests with `cargo test -p konan-core`. Build the extension with `uv run maturin develop --uv`. Run Python tests with `uv run pytest -q`.
- Every chunker upholds the invariant: `chunk.text == source[chunk.start:chunk.end]` (char offsets, Python slicing semantics). Sole exception: `MarkdownChunker` chunks whose text is prefixed with a heading breadcrumb — there `chunk.text.endswith(source_slice)`.

---

### Task 1: Workspace scaffold & build plumbing

**Files:**
- Create: `Cargo.toml`, `crates/konan-core/Cargo.toml`, `crates/konan-core/src/lib.rs`, `crates/konan-py/Cargo.toml`, `crates/konan-py/src/lib.rs`, `python/konan/__init__.py`, `.gitignore`
- Rewrite: `pyproject.toml`
- Delete: `main.py`

- [ ] **Step 1: Root `Cargo.toml`**

```toml
[workspace]
resolver = "2"
members = ["crates/konan-core", "crates/konan-py"]

[workspace.package]
version = "0.1.0"
edition = "2021"
license = "MIT"
```

- [ ] **Step 2: `crates/konan-core/Cargo.toml`**

```toml
[package]
name = "konan-core"
description = "Blazingly fast text chunking strategies"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
async-trait = "0.1"
futures = "0.3"
pulldown-cmark = { version = "0.13", default-features = false }
rayon = "1.10"
regex = "1"
reqwest = { version = "0.12", default-features = false, features = ["json", "rustls-tls"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
thiserror = "2"
tiktoken-rs = "0.7"
unicode-segmentation = "1.12"
xxhash-rust = { version = "0.8", features = ["xxh3"] }

[dev-dependencies]
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

- [ ] **Step 3: `crates/konan-core/src/lib.rs` (placeholder)**

```rust
//! konan-core: blazingly fast text chunking strategies.
```

- [ ] **Step 4: `crates/konan-py/Cargo.toml`**

```toml
[package]
name = "konan-py"
description = "Python bindings for konan-core"
version.workspace = true
edition.workspace = true
license.workspace = true

[lib]
name = "_konan"
crate-type = ["cdylib"]

[dependencies]
async-trait = "0.1"
futures = "0.3"
konan-core = { path = "../konan-core" }
pyo3 = { version = "0.25", features = ["abi3-py312"] }
pyo3-async-runtimes = { version = "0.25", features = ["tokio-runtime"] }
tokio = { version = "1", features = ["rt-multi-thread"] }
```

- [ ] **Step 5: `crates/konan-py/src/lib.rs` (stub module)**

```rust
use pyo3::prelude::*;

#[pymodule]
fn _konan(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
```

- [ ] **Step 6: Rewrite `pyproject.toml`**

```toml
[build-system]
requires = ["maturin>=1.7,<2.0"]
build-backend = "maturin"

[project]
name = "konan"
version = "0.1.0"
description = "Blazingly fast text chunkers in Rust with pythonic bindings"
readme = "README.md"
requires-python = ">=3.12"
classifiers = [
    "Programming Language :: Rust",
    "Programming Language :: Python :: Implementation :: CPython",
]

[tool.maturin]
python-source = "python"
module-name = "konan._konan"
manifest-path = "crates/konan-py/Cargo.toml"
features = ["pyo3/extension-module"]

[dependency-groups]
dev = [
    "maturin>=1.7",
    "pytest>=8",
    "pytest-asyncio>=0.25",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 7: `python/konan/__init__.py` (minimal)**

```python
from konan._konan import __version__

__all__ = ["__version__"]
```

- [ ] **Step 8: `.gitignore`**

```
/target
.venv/
__pycache__/
*.so
dist/
```

- [ ] **Step 9: Delete `main.py`**

Run: `rm main.py`

- [ ] **Step 10: Verify build end-to-end**

Run: `cargo check --workspace && uv sync && uv run maturin develop --uv && uv run python -c "import konan; print(konan.__version__)"`
Expected: prints `0.1.0`

- [ ] **Step 11: Commit**

```bash
git add -A
git commit -m "feat: scaffold konan workspace (konan-core + konan-py, maturin)"
```

---

### Task 2: Core primitives — error, chunk, text utilities

**Files:**
- Create: `crates/konan-core/src/error.rs`, `crates/konan-core/src/chunk.rs`, `crates/konan-core/src/text.rs`
- Modify: `crates/konan-core/src/lib.rs`

- [ ] **Step 1: `error.rs`**

```rust
#[derive(Debug, thiserror::Error)]
pub enum KonanError {
    #[error("invalid config: {0}")]
    InvalidConfig(String),
    #[error("embedding error: {0}")]
    Embedding(String),
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
}
```

- [ ] **Step 2: `chunk.rs`**

```rust
use xxhash_rust::xxh3::xxh3_64;

/// A chunk of text with char offsets into its source.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize)]
pub struct Chunk {
    pub text: String,
    /// Char offset (not bytes) of the chunk content in the source text.
    pub start: usize,
    /// Char offset (exclusive) of the chunk content in the source text.
    pub end: usize,
    /// 0-based chunk index.
    pub index: usize,
    /// xxh3-64 hex digest of `text`.
    pub hash: String,
}

impl Chunk {
    pub fn new(text: impl Into<String>, start: usize, end: usize, index: usize) -> Self {
        let text = text.into();
        let hash = format!("{:016x}", xxh3_64(text.as_bytes()));
        Self { text, start, end, index, hash }
    }
}
```

- [ ] **Step 3: `text.rs` with failing tests**

```rust
//! Shared span helpers. All spans are byte offsets into the source;
//! `OffsetMap` converts byte offsets to char offsets at the boundary.

use crate::chunk::Chunk;

pub(crate) struct OffsetMap {
    /// Byte offset of each char, plus a sentinel equal to text.len().
    char_starts: Vec<usize>,
}

impl OffsetMap {
    pub fn new(text: &str) -> Self {
        let mut char_starts: Vec<usize> = text.char_indices().map(|(b, _)| b).collect();
        char_starts.push(text.len());
        Self { char_starts }
    }

    /// Char index for a byte offset lying on a char boundary.
    pub fn char_idx(&self, byte: usize) -> usize {
        self.char_starts.binary_search(&byte).unwrap_or_else(|i| i - 1)
    }

    /// Char length of the byte span [start, end).
    pub fn char_len(&self, start: usize, end: usize) -> usize {
        self.char_idx(end) - self.char_idx(start)
    }
}

/// Byte spans of whitespace-separated words.
pub(crate) fn word_spans(text: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut start: Option<usize> = None;
    for (i, c) in text.char_indices() {
        if c.is_whitespace() {
            if let Some(s) = start.take() {
                spans.push((s, i));
            }
        } else if start.is_none() {
            start = Some(i);
        }
    }
    if let Some(s) = start {
        spans.push((s, text.len()));
    }
    spans
}

/// Byte spans of sentences via unicode segmentation, trimmed of whitespace.
pub(crate) fn sentence_spans(text: &str) -> Vec<(usize, usize)> {
    use unicode_segmentation::UnicodeSegmentation;
    text.split_sentence_bound_indices()
        .filter_map(|(start, s)| {
            let lead = s.len() - s.trim_start().len();
            let trimmed = s.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some((start + lead, start + lead + trimmed.len()))
            }
        })
        .collect()
}

/// Merge unit spans into chunk spans of at most `chunk_size` chars, carrying
/// at most `chunk_overlap` trailing chars of units into the next chunk.
/// A single unit larger than `chunk_size` becomes its own chunk.
pub(crate) fn merge_spans(
    map: &OffsetMap,
    spans: &[(usize, usize)],
    chunk_size: usize,
    chunk_overlap: usize,
) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::new();
    let mut window: Vec<(usize, usize)> = Vec::new();
    for &span in spans {
        if !window.is_empty() && map.char_len(window[0].0, span.1) > chunk_size {
            let merged = (window[0].0, window.last().unwrap().1);
            if out.last() != Some(&merged) {
                out.push(merged);
            }
            let mut keep: Vec<(usize, usize)> = Vec::new();
            let mut acc = 0;
            for &w in window.iter().rev() {
                let len = map.char_len(w.0, w.1);
                if acc + len > chunk_overlap {
                    break;
                }
                keep.insert(0, w);
                acc += len;
            }
            window = keep;
        }
        window.push(span);
    }
    if !window.is_empty() {
        let merged = (window[0].0, window.last().unwrap().1);
        if out.last() != Some(&merged) {
            out.push(merged);
        }
    }
    out
}

/// Convert chunk byte spans into `Chunk`s with char offsets.
pub(crate) fn spans_to_chunks(text: &str, map: &OffsetMap, spans: &[(usize, usize)]) -> Vec<Chunk> {
    spans
        .iter()
        .enumerate()
        .map(|(i, &(s, e))| Chunk::new(&text[s..e], map.char_idx(s), map.char_idx(e), i))
        .collect()
}

#[cfg(test)]
pub(crate) fn assert_char_offsets(text: &str, chunks: &[Chunk]) {
    for c in chunks {
        let expect: String = text.chars().skip(c.start).take(c.end - c.start).collect();
        assert_eq!(c.text, expect, "chunk {} offsets are wrong", c.index);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offset_map_handles_multibyte() {
        let map = OffsetMap::new("a😀b");
        assert_eq!(map.char_idx(0), 0);
        assert_eq!(map.char_idx(1), 1);
        assert_eq!(map.char_idx(5), 2);
        assert_eq!(map.char_idx(6), 3);
        assert_eq!(map.char_len(1, 5), 1);
    }

    #[test]
    fn word_spans_basic() {
        assert_eq!(word_spans("one  two\nthree "), vec![(0, 3), (5, 8), (9, 14)]);
        assert!(word_spans("   ").is_empty());
    }

    #[test]
    fn sentence_spans_trims() {
        let spans = sentence_spans("Hello there. Second one! ");
        let text = "Hello there. Second one! ";
        let sents: Vec<&str> = spans.iter().map(|&(s, e)| &text[s..e]).collect();
        assert_eq!(sents, vec!["Hello there.", "Second one!"]);
    }

    #[test]
    fn merge_spans_with_overlap() {
        let text = "aaa bbb ccc ddd";
        let map = OffsetMap::new(text);
        let spans = word_spans(text);
        let merged = merge_spans(&map, &spans, 7, 3);
        assert_eq!(merged, vec![(0, 7), (4, 11), (8, 15)]);
    }

    #[test]
    fn merge_spans_oversized_unit_alone() {
        let text = "tiny enormousunit tiny";
        let map = OffsetMap::new(text);
        let merged = merge_spans(&map, &word_spans(text), 6, 0);
        assert_eq!(merged, vec![(0, 4), (5, 17), (18, 22)]);
    }
}
```

- [ ] **Step 4: Wire up `lib.rs`**

```rust
//! konan-core: blazingly fast text chunking strategies.

pub mod chunk;
pub mod error;
pub(crate) mod text;

pub use chunk::Chunk;
pub use error::KonanError;
```

- [ ] **Step 5: Run tests**

Run: `cargo test -p konan-core`
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add crates/konan-core
git commit -m "feat(core): Chunk, KonanError, span/offset utilities"
```

---

### Task 3: Chunker port + rayon `chunk_many`

**Files:**
- Create: `crates/konan-core/src/chunker.rs`
- Modify: `crates/konan-core/src/lib.rs`

- [ ] **Step 1: `chunker.rs` with test**

```rust
use crate::chunk::Chunk;
use crate::error::KonanError;

/// Port: a text chunking strategy.
pub trait Chunker: Send + Sync {
    fn chunk(&self, text: &str) -> Result<Vec<Chunk>, KonanError>;
}

/// Chunk many texts in parallel with rayon.
pub fn chunk_many<C: Chunker + ?Sized>(
    chunker: &C,
    texts: &[String],
) -> Result<Vec<Vec<Chunk>>, KonanError> {
    use rayon::prelude::*;
    texts.par_iter().map(|t| chunker.chunk(t)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Echo;
    impl Chunker for Echo {
        fn chunk(&self, text: &str) -> Result<Vec<Chunk>, KonanError> {
            Ok(vec![Chunk::new(text, 0, text.chars().count(), 0)])
        }
    }

    #[test]
    fn chunk_many_preserves_order() {
        let texts: Vec<String> = (0..64).map(|i| format!("text-{i}")).collect();
        let out = chunk_many(&Echo, &texts).unwrap();
        assert_eq!(out.len(), 64);
        for (i, chunks) in out.iter().enumerate() {
            assert_eq!(chunks[0].text, format!("text-{i}"));
        }
    }
}
```

- [ ] **Step 2: Add to `lib.rs`** (after `pub mod chunk;`)

```rust
pub mod chunker;
```
and to the re-exports:
```rust
pub use chunker::{chunk_many, Chunker};
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p konan-core`
Expected: all pass

- [ ] **Step 4: Commit**

```bash
git add crates/konan-core
git commit -m "feat(core): Chunker port and rayon chunk_many"
```

---

### Task 4: NaiveChunker

**Files:**
- Create: `crates/konan-core/src/strategies/mod.rs`, `crates/konan-core/src/strategies/naive.rs`
- Modify: `crates/konan-core/src/lib.rs`

- [ ] **Step 1: `strategies/mod.rs`**

```rust
pub mod naive;
```

- [ ] **Step 2: `strategies/naive.rs` with tests**

```rust
use crate::chunk::Chunk;
use crate::chunker::Chunker;
use crate::error::KonanError;
use crate::text::{spans_to_chunks, word_spans, OffsetMap};

/// Simple word-based chunker without overlap. Fast, may break mid-sentence.
pub struct NaiveChunker {
    chunk_size: usize,
}

impl NaiveChunker {
    /// `chunk_size` is the number of words per chunk.
    pub fn new(chunk_size: usize) -> Result<Self, KonanError> {
        if chunk_size == 0 {
            return Err(KonanError::InvalidConfig("chunk_size must be > 0".into()));
        }
        Ok(Self { chunk_size })
    }
}

impl Chunker for NaiveChunker {
    fn chunk(&self, text: &str) -> Result<Vec<Chunk>, KonanError> {
        let words = word_spans(text);
        if words.is_empty() {
            return Ok(Vec::new());
        }
        let map = OffsetMap::new(text);
        let spans: Vec<(usize, usize)> = words
            .chunks(self.chunk_size)
            .map(|group| (group[0].0, group.last().unwrap().1))
            .collect();
        Ok(spans_to_chunks(text, &map, &spans))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::assert_char_offsets;

    #[test]
    fn groups_words() {
        let text = "one two three four five";
        let chunks = NaiveChunker::new(2).unwrap().chunk(text).unwrap();
        let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(texts, vec!["one two", "three four", "five"]);
        assert_eq!((chunks[1].start, chunks[1].end), (8, 18));
        assert_char_offsets(text, &chunks);
    }

    #[test]
    fn empty_and_whitespace_yield_nothing() {
        let c = NaiveChunker::new(10).unwrap();
        assert!(c.chunk("").unwrap().is_empty());
        assert!(c.chunk("  \n ").unwrap().is_empty());
    }

    #[test]
    fn rejects_zero_chunk_size() {
        assert!(NaiveChunker::new(0).is_err());
    }

    #[test]
    fn unicode_offsets() {
        let text = "héllo wörld 😀ok done";
        let chunks = NaiveChunker::new(2).unwrap().chunk(text).unwrap();
        assert_char_offsets(text, &chunks);
    }
}
```

- [ ] **Step 3: Wire `lib.rs`**

```rust
pub mod strategies;
```
re-export:
```rust
pub use strategies::naive::NaiveChunker;
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p konan-core`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add crates/konan-core
git commit -m "feat(core): NaiveChunker (word-based)"
```

---

### Task 5: FixedSizeChunker

**Files:**
- Create: `crates/konan-core/src/strategies/fixed_size.rs`
- Modify: `crates/konan-core/src/strategies/mod.rs`, `crates/konan-core/src/lib.rs`, `crates/konan-core/src/text.rs`

- [ ] **Step 1: Add regex sentence splitter to `text.rs`** (below `sentence_spans`)

```rust
/// Byte spans of sentences split on a punctuation regex like `[.!?]+\s+`.
/// Each sentence keeps its punctuation, drops the trailing whitespace.
pub(crate) fn regex_sentence_spans(text: &str, re: &regex::Regex) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut start = 0;
    for m in re.find_iter(text) {
        let punct_end = m.start() + m.as_str().trim_end().len();
        if punct_end > start {
            spans.push((start, punct_end));
        }
        start = m.end();
    }
    if start < text.len() {
        let tail = text[start..].trim_end();
        if !tail.is_empty() {
            spans.push((start, start + tail.len()));
        }
    }
    spans
}
```

- [ ] **Step 2: `strategies/fixed_size.rs` with tests**

```rust
use crate::chunk::Chunk;
use crate::chunker::Chunker;
use crate::error::KonanError;
use crate::text::{merge_spans, regex_sentence_spans, spans_to_chunks, word_spans, OffsetMap};

/// Character-based chunker with overlap and sentence-boundary awareness.
pub struct FixedSizeChunker {
    chunk_size: usize,
    chunk_overlap: usize,
    respect_sentences: bool,
    sentence_re: regex::Regex,
}

impl FixedSizeChunker {
    pub fn new(
        chunk_size: usize,
        chunk_overlap: usize,
        respect_sentences: bool,
    ) -> Result<Self, KonanError> {
        if chunk_size == 0 {
            return Err(KonanError::InvalidConfig("chunk_size must be > 0".into()));
        }
        if chunk_overlap >= chunk_size {
            return Err(KonanError::InvalidConfig(
                "chunk_overlap must be smaller than chunk_size".into(),
            ));
        }
        Ok(Self {
            chunk_size,
            chunk_overlap,
            respect_sentences,
            sentence_re: regex::Regex::new(r"[.!?]+\s+").expect("static regex"),
        })
    }
}

impl Chunker for FixedSizeChunker {
    fn chunk(&self, text: &str) -> Result<Vec<Chunk>, KonanError> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }
        let map = OffsetMap::new(text);
        if map.char_len(0, text.len()) <= self.chunk_size {
            return Ok(spans_to_chunks(text, &map, &[(0, text.len())]));
        }
        let units = if self.respect_sentences {
            let sents = regex_sentence_spans(text, &self.sentence_re);
            if sents.len() > 1 {
                sents
            } else {
                word_spans(text)
            }
        } else {
            word_spans(text)
        };
        let merged = merge_spans(&map, &units, self.chunk_size, self.chunk_overlap);
        Ok(spans_to_chunks(text, &map, &merged))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::assert_char_offsets;

    const TEXT: &str =
        "First sentence here. Second sentence follows! Third one now? Fourth and final.";

    #[test]
    fn respects_sentence_boundaries() {
        let chunks = FixedSizeChunker::new(45, 0, true).unwrap().chunk(TEXT).unwrap();
        assert!(chunks.len() >= 2);
        for c in &chunks {
            assert!(c.text.ends_with(['.', '!', '?']), "broke mid-sentence: {:?}", c.text);
        }
        assert_char_offsets(TEXT, &chunks);
    }

    #[test]
    fn overlap_repeats_content() {
        let chunks = FixedSizeChunker::new(45, 25, true).unwrap().chunk(TEXT).unwrap();
        assert!(chunks.len() >= 2);
        assert!(chunks[1].start < chunks[0].end, "no overlap produced");
        assert_char_offsets(TEXT, &chunks);
    }

    #[test]
    fn short_text_is_single_chunk() {
        let chunks = FixedSizeChunker::new(1000, 200, true).unwrap().chunk("tiny").unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "tiny");
    }

    #[test]
    fn word_fallback_without_sentences() {
        let text = "word ".repeat(50);
        let chunks = FixedSizeChunker::new(60, 10, true).unwrap().chunk(&text).unwrap();
        assert!(chunks.len() > 1);
        assert_char_offsets(&text, &chunks);
    }

    #[test]
    fn rejects_overlap_ge_size() {
        assert!(FixedSizeChunker::new(100, 100, true).is_err());
    }
}
```

- [ ] **Step 3: Wire up** — `strategies/mod.rs`: add `pub mod fixed_size;`. `lib.rs` re-exports: add `pub use strategies::fixed_size::FixedSizeChunker;`

- [ ] **Step 4: Run tests**

Run: `cargo test -p konan-core`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add crates/konan-core
git commit -m "feat(core): FixedSizeChunker with overlap and sentence awareness"
```

---

### Task 6: RecursiveChunker

**Files:**
- Create: `crates/konan-core/src/strategies/recursive.rs`
- Modify: `crates/konan-core/src/strategies/mod.rs`, `crates/konan-core/src/lib.rs`

- [ ] **Step 1: `strategies/recursive.rs` with tests**

```rust
use crate::chunk::Chunk;
use crate::chunker::Chunker;
use crate::error::KonanError;
use crate::text::{merge_spans, spans_to_chunks, OffsetMap};

/// Recursively splits text using a hierarchy of separators
/// (paragraph -> line -> space -> punctuation -> chars), then merges
/// pieces into chunks with overlap. LangChain-compatible semantics.
pub struct RecursiveChunker {
    chunk_size: usize,
    chunk_overlap: usize,
    separators: Vec<String>,
}

impl RecursiveChunker {
    pub fn new(
        chunk_size: usize,
        chunk_overlap: usize,
        separators: Option<Vec<String>>,
    ) -> Result<Self, KonanError> {
        if chunk_size == 0 {
            return Err(KonanError::InvalidConfig("chunk_size must be > 0".into()));
        }
        if chunk_overlap >= chunk_size {
            return Err(KonanError::InvalidConfig(
                "chunk_overlap must be smaller than chunk_size".into(),
            ));
        }
        let separators = separators.unwrap_or_else(|| {
            ["\n\n", "\n", " ", ".", ",", ""].iter().map(|s| s.to_string()).collect()
        });
        Ok(Self { chunk_size, chunk_overlap, separators })
    }

    /// Split `span` into unit spans each at most `chunk_size` chars (or
    /// unsplittable), using the separator hierarchy starting at `sep_idx`.
    /// Each piece keeps its trailing separator so chunks reproduce the source.
    pub(crate) fn split_units(
        &self,
        text: &str,
        map: &OffsetMap,
        span: (usize, usize),
        sep_idx: usize,
        out: &mut Vec<(usize, usize)>,
    ) {
        if map.char_len(span.0, span.1) <= self.chunk_size || sep_idx >= self.separators.len() {
            out.push(span);
            return;
        }
        let sep = &self.separators[sep_idx];
        if sep.is_empty() {
            // Last resort: split into chunk_size-char pieces.
            let mut start = span.0;
            let mut count = 0;
            for (b, _) in text[span.0..span.1].char_indices() {
                let abs = span.0 + b;
                if count == self.chunk_size {
                    out.push((start, abs));
                    start = abs;
                    count = 0;
                }
                count += 1;
            }
            if start < span.1 {
                out.push((start, span.1));
            }
            return;
        }
        let slice = &text[span.0..span.1];
        if !slice.contains(sep.as_str()) {
            self.split_units(text, map, span, sep_idx + 1, out);
            return;
        }
        let mut piece_start = span.0;
        let mut search_from = 0;
        while let Some(pos) = slice[search_from..].find(sep.as_str()) {
            let sep_end = span.0 + search_from + pos + sep.len();
            if sep_end > piece_start {
                self.split_units(text, map, (piece_start, sep_end), sep_idx + 1, out);
            }
            piece_start = sep_end;
            search_from = search_from + pos + sep.len();
        }
        if piece_start < span.1 {
            self.split_units(text, map, (piece_start, span.1), sep_idx + 1, out);
        }
    }
}

impl Chunker for RecursiveChunker {
    fn chunk(&self, text: &str) -> Result<Vec<Chunk>, KonanError> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }
        let map = OffsetMap::new(text);
        let mut units = Vec::new();
        self.split_units(text, &map, (0, text.len()), 0, &mut units);
        let merged = merge_spans(&map, &units, self.chunk_size, self.chunk_overlap);
        Ok(spans_to_chunks(text, &map, &merged))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::assert_char_offsets;

    #[test]
    fn prefers_paragraph_breaks() {
        let text = "Para one is here.\n\nPara two is here.\n\nPara three is here.";
        let chunks = RecursiveChunker::new(25, 0, None).unwrap().chunk(text).unwrap();
        assert_eq!(chunks.len(), 3);
        assert!(chunks[0].text.starts_with("Para one"));
        assert!(chunks[1].text.starts_with("Para two"));
        assert_char_offsets(text, &chunks);
    }

    #[test]
    fn falls_back_to_words_then_chars() {
        let text = "abcdefghij klmnopqrst";
        let chunks = RecursiveChunker::new(5, 0, None).unwrap().chunk(text).unwrap();
        for c in &chunks {
            assert!(c.end - c.start <= 5);
        }
        assert_char_offsets(text, &chunks);
    }

    #[test]
    fn overlap_carries_units() {
        let text = "alpha beta gamma delta epsilon zeta eta theta iota kappa";
        let chunks = RecursiveChunker::new(20, 8, None).unwrap().chunk(text).unwrap();
        assert!(chunks.len() >= 2);
        assert!(chunks[1].start < chunks[0].end);
        assert_char_offsets(text, &chunks);
    }

    #[test]
    fn custom_separators() {
        let text = "a|b|c|d";
        let seps = Some(vec!["|".to_string(), "".to_string()]);
        let chunks = RecursiveChunker::new(3, 0, seps).unwrap().chunk(text).unwrap();
        assert!(chunks.len() >= 2);
        assert_char_offsets(text, &chunks);
    }

    #[test]
    fn unicode_safe_char_fallback() {
        let text = "😀😁😂🤣😃😄😅😆😉😊";
        let chunks = RecursiveChunker::new(3, 0, None).unwrap().chunk(text).unwrap();
        assert_eq!(chunks.len(), 4);
        assert_char_offsets(text, &chunks);
    }
}
```

- [ ] **Step 2: Wire up** — `strategies/mod.rs`: add `pub mod recursive;`. `lib.rs`: add `pub use strategies::recursive::RecursiveChunker;`

- [ ] **Step 3: Run tests**

Run: `cargo test -p konan-core`
Expected: all pass

- [ ] **Step 4: Commit**

```bash
git add crates/konan-core
git commit -m "feat(core): RecursiveChunker with separator hierarchy"
```

---

### Task 7: SentenceChunker

**Files:**
- Create: `crates/konan-core/src/strategies/sentence.rs`
- Modify: `crates/konan-core/src/strategies/mod.rs`, `crates/konan-core/src/lib.rs`

- [ ] **Step 1: `strategies/sentence.rs` with tests**

```rust
use crate::chunk::Chunk;
use crate::chunker::Chunker;
use crate::error::KonanError;
use crate::text::{sentence_spans, spans_to_chunks, OffsetMap};

/// Unicode-aware sentence chunker: groups whole sentences into chunks of at
/// most `max_chars`, overlapping by `overlap_sentences` sentences.
pub struct SentenceChunker {
    max_chars: usize,
    overlap_sentences: usize,
}

impl SentenceChunker {
    pub fn new(max_chars: usize, overlap_sentences: usize) -> Result<Self, KonanError> {
        if max_chars == 0 {
            return Err(KonanError::InvalidConfig("max_chars must be > 0".into()));
        }
        Ok(Self { max_chars, overlap_sentences })
    }
}

impl Chunker for SentenceChunker {
    fn chunk(&self, text: &str) -> Result<Vec<Chunk>, KonanError> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }
        let map = OffsetMap::new(text);
        let sents = sentence_spans(text);
        let mut spans: Vec<(usize, usize)> = Vec::new();
        let mut start_idx = 0;
        while start_idx < sents.len() {
            let mut end_idx = start_idx;
            while end_idx + 1 < sents.len()
                && map.char_len(sents[start_idx].0, sents[end_idx + 1].1) <= self.max_chars
            {
                end_idx += 1;
            }
            spans.push((sents[start_idx].0, sents[end_idx].1));
            if end_idx + 1 >= sents.len() {
                break;
            }
            let mut next_start = (end_idx + 1).saturating_sub(self.overlap_sentences);
            if next_start <= start_idx {
                next_start = start_idx + 1; // guarantee progress
            }
            start_idx = next_start;
        }
        Ok(spans_to_chunks(text, &map, &spans))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::assert_char_offsets;

    const TEXT: &str = "Sentence number one. Sentence number two. Sentence number three. Sentence number four.";

    #[test]
    fn groups_whole_sentences() {
        let chunks = SentenceChunker::new(45, 0).unwrap().chunk(TEXT).unwrap();
        assert!(chunks.len() >= 2);
        for c in &chunks {
            assert!(c.text.starts_with("Sentence"));
            assert!(c.text.ends_with('.'));
        }
        assert_char_offsets(TEXT, &chunks);
    }

    #[test]
    fn sentence_overlap() {
        let chunks = SentenceChunker::new(45, 1).unwrap().chunk(TEXT).unwrap();
        assert!(chunks.len() >= 2);
        // chunk 1 starts at the last sentence of chunk 0
        assert!(chunks[1].start < chunks[0].end);
        assert_char_offsets(TEXT, &chunks);
    }

    #[test]
    fn oversized_sentence_emitted_alone() {
        let text = "Short. This single sentence is much longer than the limit allows. End.";
        let chunks = SentenceChunker::new(20, 0).unwrap().chunk(text).unwrap();
        assert!(chunks.iter().any(|c| c.text.contains("much longer")));
        assert_char_offsets(text, &chunks);
    }
}
```

- [ ] **Step 2: Wire up** — `strategies/mod.rs`: add `pub mod sentence;`. `lib.rs`: add `pub use strategies::sentence::SentenceChunker;`

- [ ] **Step 3: Run tests**

Run: `cargo test -p konan-core`
Expected: all pass

- [ ] **Step 4: Commit**

```bash
git add crates/konan-core
git commit -m "feat(core): SentenceChunker (unicode segmentation)"
```

---

### Task 8: MarkdownChunker

**Files:**
- Create: `crates/konan-core/src/strategies/markdown.rs`
- Modify: `crates/konan-core/src/strategies/mod.rs`, `crates/konan-core/src/lib.rs`

- [ ] **Step 1: `strategies/markdown.rs` with tests**

````rust
use crate::chunk::Chunk;
use crate::chunker::Chunker;
use crate::error::KonanError;
use crate::strategies::recursive::RecursiveChunker;
use crate::text::{merge_spans, OffsetMap};

/// Markdown-structure-aware chunker. Splits along top-level blocks, never
/// splits fenced code blocks, and prefixes each chunk with its heading
/// breadcrumb (e.g. "# A > ## B"). For breadcrumbed chunks,
/// `start`/`end` refer to the source content span (the breadcrumb prefix is
/// not part of the source).
pub struct MarkdownChunker {
    chunk_size: usize,
    chunk_overlap: usize,
    splitter: RecursiveChunker,
}

struct Block {
    span: (usize, usize),
    is_code: bool,
    heading: Option<u32>,
}

fn heading_level(level: pulldown_cmark::HeadingLevel) -> u32 {
    use pulldown_cmark::HeadingLevel::*;
    match level {
        H1 => 1,
        H2 => 2,
        H3 => 3,
        H4 => 4,
        H5 => 5,
        H6 => 6,
    }
}

fn parse_blocks(text: &str) -> Vec<Block> {
    use pulldown_cmark::{Event, Options, Parser, Tag};
    let mut blocks = Vec::new();
    let mut depth = 0usize;
    for (event, range) in Parser::new_ext(text, Options::empty()).into_offset_iter() {
        match event {
            Event::Start(tag) => {
                if depth == 0 {
                    let heading = match &tag {
                        Tag::Heading { level, .. } => Some(heading_level(*level)),
                        _ => None,
                    };
                    blocks.push(Block {
                        span: (range.start, range.end),
                        is_code: matches!(tag, Tag::CodeBlock(_)),
                        heading,
                    });
                }
                depth += 1;
            }
            Event::End(_) => depth -= 1,
            Event::Rule | Event::Html(_) if depth == 0 => {
                blocks.push(Block { span: (range.start, range.end), is_code: false, heading: None });
            }
            _ => {}
        }
    }
    blocks
}

impl MarkdownChunker {
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Result<Self, KonanError> {
        if chunk_size == 0 {
            return Err(KonanError::InvalidConfig("chunk_size must be > 0".into()));
        }
        if chunk_overlap >= chunk_size {
            return Err(KonanError::InvalidConfig(
                "chunk_overlap must be smaller than chunk_size".into(),
            ));
        }
        let splitter = RecursiveChunker::new(chunk_size, chunk_overlap, None)?;
        Ok(Self { chunk_size, chunk_overlap, splitter })
    }

    fn flush_section(
        &self,
        text: &str,
        map: &OffsetMap,
        section: &[(usize, usize, bool)],
        breadcrumb: &str,
        chunks: &mut Vec<Chunk>,
    ) {
        if section.is_empty() {
            return;
        }
        // Oversized non-code blocks get recursively split; code blocks stay atomic.
        let mut units: Vec<(usize, usize)> = Vec::new();
        for &(s, e, is_code) in section {
            if !is_code && map.char_len(s, e) > self.chunk_size {
                self.splitter.split_units(text, map, (s, e), 0, &mut units);
            } else {
                units.push((s, e));
            }
        }
        for (s, e) in merge_spans(map, &units, self.chunk_size, self.chunk_overlap) {
            let content = text[s..e].trim_end();
            let chunk_text = if breadcrumb.is_empty() {
                content.to_string()
            } else {
                format!("{breadcrumb}\n\n{content}")
            };
            let index = chunks.len();
            let end = map.char_idx(s) + content.chars().count();
            chunks.push(Chunk::new(chunk_text, map.char_idx(s), end, index));
        }
    }
}

impl Chunker for MarkdownChunker {
    fn chunk(&self, text: &str) -> Result<Vec<Chunk>, KonanError> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }
        let map = OffsetMap::new(text);
        let mut chunks: Vec<Chunk> = Vec::new();
        let mut stack: Vec<(u32, String)> = Vec::new();
        let mut section: Vec<(usize, usize, bool)> = Vec::new();
        let mut breadcrumb = String::new();

        for block in parse_blocks(text) {
            if let Some(level) = block.heading {
                self.flush_section(text, &map, &section, &breadcrumb, &mut chunks);
                section.clear();
                let title = text[block.span.0..block.span.1]
                    .trim()
                    .trim_start_matches('#')
                    .trim()
                    .to_string();
                while stack.last().is_some_and(|(l, _)| *l >= level) {
                    stack.pop();
                }
                stack.push((level, title));
                breadcrumb = stack
                    .iter()
                    .map(|(l, t)| format!("{} {}", "#".repeat(*l as usize), t))
                    .collect::<Vec<_>>()
                    .join(" > ");
            } else {
                section.push((block.span.0, block.span.1, block.is_code));
            }
        }
        self.flush_section(text, &map, &section, &breadcrumb, &mut chunks);
        Ok(chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MD: &str = "# Guide\n\nIntro paragraph about the guide.\n\n## Install\n\nRun the installer.\n\n```bash\npip install konan\n```\n";

    fn source_slice(text: &str, c: &Chunk) -> String {
        text.chars().skip(c.start).take(c.end - c.start).collect()
    }

    #[test]
    fn breadcrumbs_prefix_chunks() {
        let chunks = MarkdownChunker::new(200, 0).unwrap().chunk(MD).unwrap();
        assert!(chunks[0].text.starts_with("# Guide\n\nIntro paragraph"));
        assert!(chunks.iter().any(|c| c.text.starts_with("# Guide > ## Install")));
        for c in &chunks {
            assert!(c.text.ends_with(&source_slice(MD, c)));
        }
    }

    #[test]
    fn code_fences_never_split() {
        let chunks = MarkdownChunker::new(10, 0).unwrap().chunk(MD).unwrap();
        let code: Vec<_> = chunks.iter().filter(|c| c.text.contains("```bash")).collect();
        assert_eq!(code.len(), 1);
        assert!(code[0].text.contains("pip install konan"));
    }

    #[test]
    fn plain_text_without_headings() {
        let text = "Just a paragraph.\n\nAnother paragraph.";
        let chunks = MarkdownChunker::new(200, 0).unwrap().chunk(text).unwrap();
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.starts_with("Just a paragraph."));
    }

    #[test]
    fn sibling_heading_replaces_breadcrumb() {
        let md = "## A\n\none\n\n## B\n\ntwo\n";
        let chunks = MarkdownChunker::new(200, 0).unwrap().chunk(md).unwrap();
        assert!(chunks[0].text.starts_with("## A"));
        assert!(chunks[1].text.starts_with("## B"));
        assert!(!chunks[1].text.contains("## A"));
    }
}
````

- [ ] **Step 2: Wire up** — `strategies/mod.rs`: add `pub mod markdown;`. `lib.rs`: add `pub use strategies::markdown::MarkdownChunker;`

- [ ] **Step 3: Run tests**

Run: `cargo test -p konan-core`
Expected: all pass

- [ ] **Step 4: Commit**

```bash
git add crates/konan-core
git commit -m "feat(core): MarkdownChunker with breadcrumbs and atomic code fences"
```

---

### Task 9: TokenChunker

**Files:**
- Create: `crates/konan-core/src/strategies/token.rs`
- Modify: `crates/konan-core/src/strategies/mod.rs`, `crates/konan-core/src/lib.rs`

- [ ] **Step 1: `strategies/token.rs` with tests**

```rust
use crate::chunk::Chunk;
use crate::chunker::Chunker;
use crate::error::KonanError;
use crate::text::OffsetMap;
use tiktoken_rs::CoreBPE;

/// Token-exact chunker via tiktoken. `chunk_size`/`chunk_overlap` are in
/// tokens. Supported encodings: "cl100k_base", "o200k_base".
pub struct TokenChunker {
    bpe: CoreBPE,
    chunk_size: usize,
    chunk_overlap: usize,
}

impl TokenChunker {
    pub fn new(chunk_size: usize, chunk_overlap: usize, encoding: &str) -> Result<Self, KonanError> {
        if chunk_size == 0 {
            return Err(KonanError::InvalidConfig("chunk_size must be > 0".into()));
        }
        if chunk_overlap >= chunk_size {
            return Err(KonanError::InvalidConfig(
                "chunk_overlap must be smaller than chunk_size".into(),
            ));
        }
        let bpe = match encoding {
            "cl100k_base" => tiktoken_rs::cl100k_base(),
            "o200k_base" => tiktoken_rs::o200k_base(),
            other => return Err(KonanError::Tokenizer(format!("unknown encoding: {other}"))),
        }
        .map_err(|e| KonanError::Tokenizer(e.to_string()))?;
        Ok(Self { bpe, chunk_size, chunk_overlap })
    }
}

impl Chunker for TokenChunker {
    fn chunk(&self, text: &str) -> Result<Vec<Chunk>, KonanError> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let tokens = self.bpe.encode_ordinary(text);
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        // Byte offset where each token starts, plus a sentinel.
        // NOTE: if `decode_bytes` is not public in the pinned tiktoken-rs,
        // use `self.bpe.split_by_token_iter(text, false)` piece byte-lengths instead.
        let mut offsets = Vec::with_capacity(tokens.len() + 1);
        let mut pos = 0usize;
        for &tok in &tokens {
            offsets.push(pos);
            let bytes = self
                .bpe
                .decode_bytes(&[tok])
                .map_err(|e| KonanError::Tokenizer(e.to_string()))?;
            pos += bytes.len();
        }
        offsets.push(pos);

        let map = OffsetMap::new(text);
        let step = self.chunk_size - self.chunk_overlap;
        let mut chunks = Vec::new();
        let mut i = 0;
        while i < tokens.len() {
            let j = (i + self.chunk_size).min(tokens.len());
            let mut start_b = offsets[i];
            let mut end_b = offsets[j];
            // Tokens can split multi-byte chars: snap to char boundaries.
            while start_b > 0 && !text.is_char_boundary(start_b) {
                start_b -= 1;
            }
            while end_b < text.len() && !text.is_char_boundary(end_b) {
                end_b += 1;
            }
            let index = chunks.len();
            chunks.push(Chunk::new(
                &text[start_b..end_b],
                map.char_idx(start_b),
                map.char_idx(end_b),
                index,
            ));
            if j == tokens.len() {
                break;
            }
            i += step;
        }
        Ok(chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::assert_char_offsets;

    #[test]
    fn token_counts_respected() {
        let chunker = TokenChunker::new(8, 0, "cl100k_base").unwrap();
        let text = "the quick brown fox jumps over the lazy dog ".repeat(5);
        let chunks = chunker.chunk(&text).unwrap();
        assert!(chunks.len() > 1);
        for c in &chunks {
            let n = chunker.bpe.encode_ordinary(&c.text).len();
            assert!(n <= 9, "chunk has {n} tokens"); // +1 slack for boundary snapping
        }
        assert_char_offsets(&text, &chunks);
    }

    #[test]
    fn overlap_in_tokens() {
        let chunker = TokenChunker::new(8, 4, "cl100k_base").unwrap();
        let text = "one two three four five six seven eight nine ten eleven twelve";
        let chunks = chunker.chunk(text).unwrap();
        assert!(chunks.len() >= 2);
        assert!(chunks[1].start < chunks[0].end);
        assert_char_offsets(text, &chunks);
    }

    #[test]
    fn unknown_encoding_rejected() {
        assert!(matches!(
            TokenChunker::new(8, 0, "nope"),
            Err(KonanError::Tokenizer(_))
        ));
    }

    #[test]
    fn unicode_boundary_snapping() {
        let chunker = TokenChunker::new(2, 0, "cl100k_base").unwrap();
        let text = "😀😁😂🤣😃 schön müde";
        let chunks = chunker.chunk(text).unwrap();
        assert_char_offsets(text, &chunks);
    }
}
```

- [ ] **Step 2: Wire up** — `strategies/mod.rs`: add `pub mod token;`. `lib.rs`: add `pub use strategies::token::TokenChunker;`

- [ ] **Step 3: Run tests**

Run: `cargo test -p konan-core`
Expected: all pass

- [ ] **Step 4: Commit**

```bash
git add crates/konan-core
git commit -m "feat(core): TokenChunker (tiktoken cl100k/o200k)"
```

---

### Task 10: Embedder port + OpenAIEmbedder adapter

**Files:**
- Create: `crates/konan-core/src/embedder.rs`
- Modify: `crates/konan-core/src/lib.rs`

- [ ] **Step 1: `embedder.rs` with tests**

```rust
use crate::error::KonanError;
use async_trait::async_trait;

/// Port: an embedding backend. Adapters: OpenAIEmbedder (HTTP), or anything
/// injected from the outside (e.g. a Python async callable in konan-py).
#[async_trait]
pub trait Embedder: Send + Sync {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, KonanError>;
}

#[async_trait]
impl<T: Embedder + ?Sized> Embedder for std::sync::Arc<T> {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, KonanError> {
        (**self).embed(texts).await
    }
}

/// Adapter: any OpenAI-compatible `/embeddings` endpoint.
pub struct OpenAIEmbedder {
    client: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
    model: String,
    batch_size: usize,
}

impl OpenAIEmbedder {
    pub fn new(
        base_url: impl Into<String>,
        model: impl Into<String>,
        api_key: Option<String>,
        batch_size: usize,
    ) -> Result<Self, KonanError> {
        if batch_size == 0 {
            return Err(KonanError::InvalidConfig("batch_size must be > 0".into()));
        }
        Ok(Self {
            client: reqwest::Client::new(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key,
            model: model.into(),
            batch_size,
        })
    }
}

#[derive(serde::Deserialize)]
struct EmbeddingsResponse {
    data: Vec<EmbeddingItem>,
}

#[derive(serde::Deserialize)]
struct EmbeddingItem {
    index: usize,
    embedding: Vec<f32>,
}

#[async_trait]
impl Embedder for OpenAIEmbedder {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, KonanError> {
        let url = format!("{}/embeddings", self.base_url);
        let mut out: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
        for batch in texts.chunks(self.batch_size) {
            let mut req = self.client.post(&url).json(&serde_json::json!({
                "model": self.model,
                "input": batch,
            }));
            if let Some(key) = &self.api_key {
                req = req.bearer_auth(key);
            }
            let resp = req.send().await.map_err(|e| KonanError::Embedding(e.to_string()))?;
            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                return Err(KonanError::Embedding(format!("HTTP {status}: {body}")));
            }
            let mut parsed: EmbeddingsResponse =
                resp.json().await.map_err(|e| KonanError::Embedding(e.to_string()))?;
            if parsed.data.len() != batch.len() {
                return Err(KonanError::Embedding(format!(
                    "expected {} embeddings, got {}",
                    batch.len(),
                    parsed.data.len()
                )));
            }
            parsed.data.sort_by_key(|d| d.index);
            out.extend(parsed.data.into_iter().map(|d| d.embedding));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_openai_response_shape() {
        let json = r#"{"object":"list","model":"m","data":[
            {"object":"embedding","index":1,"embedding":[0.0,1.0]},
            {"object":"embedding","index":0,"embedding":[1.0,0.0]}
        ]}"#;
        let mut parsed: EmbeddingsResponse = serde_json::from_str(json).unwrap();
        parsed.data.sort_by_key(|d| d.index);
        assert_eq!(parsed.data[0].embedding, vec![1.0, 0.0]);
        assert_eq!(parsed.data[1].embedding, vec![0.0, 1.0]);
    }

    #[tokio::test]
    async fn unreachable_endpoint_is_embedding_error() {
        let e = OpenAIEmbedder::new("http://127.0.0.1:9", "m", None, 16).unwrap();
        let err = e.embed(&["x".to_string()]).await.unwrap_err();
        assert!(matches!(err, KonanError::Embedding(_)));
    }
}
```

- [ ] **Step 2: Wire up `lib.rs`** — add `pub mod embedder;` and `pub use embedder::{Embedder, OpenAIEmbedder};`

- [ ] **Step 3: Run tests**

Run: `cargo test -p konan-core`
Expected: all pass

- [ ] **Step 4: Commit**

```bash
git add crates/konan-core
git commit -m "feat(core): Embedder port and OpenAI-compatible adapter"
```

---

### Task 11: SemanticChunker (core)

**Files:**
- Create: `crates/konan-core/src/semantic.rs`
- Modify: `crates/konan-core/src/lib.rs`

- [ ] **Step 1: `semantic.rs` with tests**

```rust
use crate::chunk::Chunk;
use crate::embedder::Embedder;
use crate::error::KonanError;
use crate::text::{merge_spans, sentence_spans, spans_to_chunks, OffsetMap};

/// Splits where embedding similarity between adjacent sentences drops.
/// Break rule: `threshold` (absolute cosine similarity) if set, otherwise the
/// `percentile`-th percentile of adjacent distances (default p95).
pub struct SemanticChunker<E: Embedder> {
    embedder: E,
    threshold: Option<f32>,
    percentile: f32,
    min_chunk_size: usize,
    max_chunk_size: Option<usize>,
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

impl<E: Embedder> SemanticChunker<E> {
    pub fn new(
        embedder: E,
        threshold: Option<f32>,
        percentile: f32,
        min_chunk_size: usize,
        max_chunk_size: Option<usize>,
    ) -> Result<Self, KonanError> {
        if let Some(t) = threshold {
            if !(-1.0..=1.0).contains(&t) {
                return Err(KonanError::InvalidConfig("threshold must be in [-1, 1]".into()));
            }
        }
        if !(0.0..=100.0).contains(&percentile) {
            return Err(KonanError::InvalidConfig("percentile must be in [0, 100]".into()));
        }
        if let Some(m) = max_chunk_size {
            if m == 0 || m < min_chunk_size {
                return Err(KonanError::InvalidConfig(
                    "max_chunk_size must be > 0 and >= min_chunk_size".into(),
                ));
            }
        }
        Ok(Self { embedder, threshold, percentile, min_chunk_size, max_chunk_size })
    }

    pub async fn chunk(&self, text: &str) -> Result<Vec<Chunk>, KonanError> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }
        let map = OffsetMap::new(text);
        let sents = sentence_spans(text);
        if sents.len() <= 1 {
            return Ok(spans_to_chunks(text, &map, &sents));
        }
        let sentence_texts: Vec<String> =
            sents.iter().map(|&(s, e)| text[s..e].to_string()).collect();
        let embeddings = self.embedder.embed(&sentence_texts).await?;
        if embeddings.len() != sents.len() {
            return Err(KonanError::Embedding(format!(
                "embedder returned {} embeddings for {} sentences",
                embeddings.len(),
                sents.len()
            )));
        }
        let sims: Vec<f32> =
            embeddings.windows(2).map(|w| cosine_similarity(&w[0], &w[1])).collect();
        let cutoff = match self.threshold {
            Some(t) => t,
            None => {
                let mut distances: Vec<f32> = sims.iter().map(|s| 1.0 - s).collect();
                distances.sort_by(|a, b| a.partial_cmp(b).expect("no NaN distances"));
                let rank = ((self.percentile / 100.0) * (distances.len() - 1) as f32).round() as usize;
                1.0 - distances[rank]
            }
        };

        // Group sentences, breaking where similarity drops below the cutoff.
        let mut groups: Vec<Vec<(usize, usize)>> = vec![vec![sents[0]]];
        for (i, &sim) in sims.iter().enumerate() {
            if sim < cutoff {
                groups.push(Vec::new());
            }
            groups.last_mut().unwrap().push(sents[i + 1]);
        }

        // Enforce min_chunk_size: groups still too small absorb the next group.
        if self.min_chunk_size > 0 {
            let mut merged: Vec<Vec<(usize, usize)>> = Vec::new();
            for g in groups {
                let prev_small = merged.last().is_some_and(|p: &Vec<(usize, usize)>| {
                    map.char_len(p[0].0, p.last().unwrap().1) < self.min_chunk_size
                });
                if prev_small {
                    merged.last_mut().unwrap().extend(g);
                } else {
                    merged.push(g);
                }
            }
            groups = merged;
        }

        // Enforce max_chunk_size: oversized groups re-split by sentence.
        if let Some(maxs) = self.max_chunk_size {
            let mut split: Vec<Vec<(usize, usize)>> = Vec::new();
            for g in groups {
                if map.char_len(g[0].0, g.last().unwrap().1) <= maxs {
                    split.push(g);
                } else {
                    for span in merge_spans(&map, &g, maxs, 0) {
                        split.push(vec![span]);
                    }
                }
            }
            groups = split;
        }

        let spans: Vec<(usize, usize)> = groups
            .iter()
            .filter(|g| !g.is_empty())
            .map(|g| (g[0].0, g.last().unwrap().1))
            .collect();
        Ok(spans_to_chunks(text, &map, &spans))
    }

    pub async fn chunk_many(&self, texts: &[String]) -> Result<Vec<Vec<Chunk>>, KonanError> {
        futures::future::try_join_all(texts.iter().map(|t| self.chunk(t))).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    struct FakeEmbedder;

    #[async_trait]
    impl Embedder for FakeEmbedder {
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, KonanError> {
            Ok(texts
                .iter()
                .map(|t| if t.to_lowercase().contains("cat") { vec![1.0, 0.0] } else { vec![0.0, 1.0] })
                .collect())
        }
    }

    const TEXT: &str = "Cats purr softly. My cat naps all day. Quantum entanglement defies intuition. Quantum computers exploit superposition.";

    #[tokio::test]
    async fn splits_on_topic_shift_with_threshold() {
        let c = SemanticChunker::new(FakeEmbedder, Some(0.5), 95.0, 0, None).unwrap();
        let chunks = c.chunk(TEXT).await.unwrap();
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].text.contains("cat naps"));
        assert!(chunks[1].text.starts_with("Quantum"));
        crate::text::assert_char_offsets(TEXT, &chunks);
    }

    #[tokio::test]
    async fn percentile_mode_splits_largest_gap() {
        let c = SemanticChunker::new(FakeEmbedder, None, 50.0, 0, None).unwrap();
        let chunks = c.chunk(TEXT).await.unwrap();
        assert!(chunks.len() >= 2);
    }

    #[tokio::test]
    async fn single_sentence_is_single_chunk() {
        let c = SemanticChunker::new(FakeEmbedder, Some(0.5), 95.0, 0, None).unwrap();
        let chunks = c.chunk("Just one sentence here.").await.unwrap();
        assert_eq!(chunks.len(), 1);
    }

    #[tokio::test]
    async fn max_chunk_size_resplits() {
        let c = SemanticChunker::new(FakeEmbedder, Some(0.5), 95.0, 0, Some(25)).unwrap();
        let chunks = c.chunk(TEXT).await.unwrap();
        assert!(chunks.len() > 2);
    }

    #[test]
    fn validates_config() {
        assert!(SemanticChunker::new(FakeEmbedder, Some(2.0), 95.0, 0, None).is_err());
        assert!(SemanticChunker::new(FakeEmbedder, None, 200.0, 0, None).is_err());
        assert!(SemanticChunker::new(FakeEmbedder, None, 95.0, 100, Some(50)).is_err());
    }
}
```

- [ ] **Step 2: Wire up `lib.rs`** — add `pub mod semantic;` and `pub use semantic::SemanticChunker;`

- [ ] **Step 3: Run tests**

Run: `cargo test -p konan-core`
Expected: all pass

- [ ] **Step 4: Commit**

```bash
git add crates/konan-core
git commit -m "feat(core): SemanticChunker over the Embedder port"
```

---

### Task 12: Bindings — PyChunk, errors, shared helpers, six chunker classes

**Files:**
- Create: `crates/konan-py/src/chunk.rs`, `crates/konan-py/src/errors.rs`, `crates/konan-py/src/common.rs`, `crates/konan-py/src/chunkers.rs`
- Modify: `crates/konan-py/src/lib.rs`

- [ ] **Step 1: `errors.rs`**

```rust
use konan_core::KonanError;
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyValueError};

create_exception!(konan, EmbeddingError, PyException, "Embedding backend failure.");

pub(crate) fn to_py_err(err: KonanError) -> pyo3::PyErr {
    match err {
        KonanError::InvalidConfig(msg) | KonanError::Tokenizer(msg) => PyValueError::new_err(msg),
        KonanError::Embedding(msg) => EmbeddingError::new_err(msg),
    }
}
```

- [ ] **Step 2: `chunk.rs`**

```rust
use pyo3::prelude::*;

#[pyclass(name = "Chunk", module = "konan", frozen)]
#[derive(Clone)]
pub struct PyChunk {
    pub(crate) inner: konan_core::Chunk,
}

impl From<konan_core::Chunk> for PyChunk {
    fn from(inner: konan_core::Chunk) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyChunk {
    #[getter]
    fn text(&self) -> &str {
        &self.inner.text
    }
    #[getter]
    fn start(&self) -> usize {
        self.inner.start
    }
    #[getter]
    fn end(&self) -> usize {
        self.inner.end
    }
    #[getter]
    fn index(&self) -> usize {
        self.inner.index
    }
    #[getter]
    fn hash(&self) -> &str {
        &self.inner.hash
    }
    fn __len__(&self) -> usize {
        self.inner.text.chars().count()
    }
    fn __repr__(&self) -> String {
        let preview: String = self.inner.text.chars().take(40).collect();
        format!(
            "Chunk(index={}, start={}, end={}, text={:?})",
            self.inner.index, self.inner.start, self.inner.end, preview
        )
    }
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
    fn __hash__(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        self.inner.hash(&mut h);
        h.finish()
    }
}
```

- [ ] **Step 3: `common.rs` (shared sync/async method bodies)**

```rust
use std::sync::Arc;

use konan_core::{chunk_many, Chunker};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::chunk::PyChunk;
use crate::errors::to_py_err;

pub(crate) fn do_chunk(py: Python<'_>, chunker: Arc<dyn Chunker>, text: String) -> PyResult<Vec<PyChunk>> {
    let chunks = py.allow_threads(move || chunker.chunk(&text)).map_err(to_py_err)?;
    Ok(chunks.into_iter().map(PyChunk::from).collect())
}

pub(crate) fn do_chunk_many(
    py: Python<'_>,
    chunker: Arc<dyn Chunker>,
    texts: Vec<String>,
) -> PyResult<Vec<Vec<PyChunk>>> {
    let results = py.allow_threads(move || chunk_many(&*chunker, &texts)).map_err(to_py_err)?;
    Ok(results
        .into_iter()
        .map(|cs| cs.into_iter().map(PyChunk::from).collect())
        .collect())
}

pub(crate) fn do_chunk_async<'p>(
    py: Python<'p>,
    chunker: Arc<dyn Chunker>,
    text: String,
) -> PyResult<Bound<'p, PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let chunks = tokio::task::spawn_blocking(move || chunker.chunk(&text))
            .await
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .map_err(to_py_err)?;
        Ok(chunks.into_iter().map(PyChunk::from).collect::<Vec<_>>())
    })
}

pub(crate) fn do_chunk_many_async<'p>(
    py: Python<'p>,
    chunker: Arc<dyn Chunker>,
    texts: Vec<String>,
) -> PyResult<Bound<'p, PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let results = tokio::task::spawn_blocking(move || chunk_many(&*chunker, &texts))
            .await
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .map_err(to_py_err)?;
        Ok(results
            .into_iter()
            .map(|cs| cs.into_iter().map(PyChunk::from).collect::<Vec<PyChunk>>())
            .collect::<Vec<_>>())
    })
}
```

- [ ] **Step 4: `chunkers.rs` (six classes; methods are identical one-line delegations)**

```rust
use std::sync::Arc;

use konan_core::{
    Chunker, FixedSizeChunker, MarkdownChunker, NaiveChunker, RecursiveChunker, SentenceChunker,
    TokenChunker,
};
use pyo3::prelude::*;

use crate::chunk::PyChunk;
use crate::common::{do_chunk, do_chunk_async, do_chunk_many, do_chunk_many_async};
use crate::errors::to_py_err;

#[pyclass(name = "NaiveChunker", module = "konan", frozen)]
pub struct PyNaiveChunker {
    inner: Arc<dyn Chunker>,
}

#[pymethods]
impl PyNaiveChunker {
    #[new]
    #[pyo3(signature = (chunk_size=200))]
    fn new(chunk_size: usize) -> PyResult<Self> {
        Ok(Self { inner: Arc::new(NaiveChunker::new(chunk_size).map_err(to_py_err)?) })
    }
    fn chunk(&self, py: Python<'_>, text: String) -> PyResult<Vec<PyChunk>> {
        do_chunk(py, Arc::clone(&self.inner), text)
    }
    fn chunk_many(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<Vec<Vec<PyChunk>>> {
        do_chunk_many(py, Arc::clone(&self.inner), texts)
    }
    fn chunk_async<'p>(&self, py: Python<'p>, text: String) -> PyResult<Bound<'p, PyAny>> {
        do_chunk_async(py, Arc::clone(&self.inner), text)
    }
    fn chunk_many_async<'p>(&self, py: Python<'p>, texts: Vec<String>) -> PyResult<Bound<'p, PyAny>> {
        do_chunk_many_async(py, Arc::clone(&self.inner), texts)
    }
}

#[pyclass(name = "FixedSizeChunker", module = "konan", frozen)]
pub struct PyFixedSizeChunker {
    inner: Arc<dyn Chunker>,
}

#[pymethods]
impl PyFixedSizeChunker {
    #[new]
    #[pyo3(signature = (chunk_size=1000, chunk_overlap=200, respect_sentences=true))]
    fn new(chunk_size: usize, chunk_overlap: usize, respect_sentences: bool) -> PyResult<Self> {
        Ok(Self {
            inner: Arc::new(
                FixedSizeChunker::new(chunk_size, chunk_overlap, respect_sentences)
                    .map_err(to_py_err)?,
            ),
        })
    }
    fn chunk(&self, py: Python<'_>, text: String) -> PyResult<Vec<PyChunk>> {
        do_chunk(py, Arc::clone(&self.inner), text)
    }
    fn chunk_many(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<Vec<Vec<PyChunk>>> {
        do_chunk_many(py, Arc::clone(&self.inner), texts)
    }
    fn chunk_async<'p>(&self, py: Python<'p>, text: String) -> PyResult<Bound<'p, PyAny>> {
        do_chunk_async(py, Arc::clone(&self.inner), text)
    }
    fn chunk_many_async<'p>(&self, py: Python<'p>, texts: Vec<String>) -> PyResult<Bound<'p, PyAny>> {
        do_chunk_many_async(py, Arc::clone(&self.inner), texts)
    }
}

#[pyclass(name = "RecursiveChunker", module = "konan", frozen)]
pub struct PyRecursiveChunker {
    inner: Arc<dyn Chunker>,
}

#[pymethods]
impl PyRecursiveChunker {
    #[new]
    #[pyo3(signature = (chunk_size=1000, chunk_overlap=200, separators=None))]
    fn new(chunk_size: usize, chunk_overlap: usize, separators: Option<Vec<String>>) -> PyResult<Self> {
        Ok(Self {
            inner: Arc::new(
                RecursiveChunker::new(chunk_size, chunk_overlap, separators).map_err(to_py_err)?,
            ),
        })
    }
    fn chunk(&self, py: Python<'_>, text: String) -> PyResult<Vec<PyChunk>> {
        do_chunk(py, Arc::clone(&self.inner), text)
    }
    fn chunk_many(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<Vec<Vec<PyChunk>>> {
        do_chunk_many(py, Arc::clone(&self.inner), texts)
    }
    fn chunk_async<'p>(&self, py: Python<'p>, text: String) -> PyResult<Bound<'p, PyAny>> {
        do_chunk_async(py, Arc::clone(&self.inner), text)
    }
    fn chunk_many_async<'p>(&self, py: Python<'p>, texts: Vec<String>) -> PyResult<Bound<'p, PyAny>> {
        do_chunk_many_async(py, Arc::clone(&self.inner), texts)
    }
}

#[pyclass(name = "SentenceChunker", module = "konan", frozen)]
pub struct PySentenceChunker {
    inner: Arc<dyn Chunker>,
}

#[pymethods]
impl PySentenceChunker {
    #[new]
    #[pyo3(signature = (max_chars=1000, overlap_sentences=1))]
    fn new(max_chars: usize, overlap_sentences: usize) -> PyResult<Self> {
        Ok(Self {
            inner: Arc::new(SentenceChunker::new(max_chars, overlap_sentences).map_err(to_py_err)?),
        })
    }
    fn chunk(&self, py: Python<'_>, text: String) -> PyResult<Vec<PyChunk>> {
        do_chunk(py, Arc::clone(&self.inner), text)
    }
    fn chunk_many(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<Vec<Vec<PyChunk>>> {
        do_chunk_many(py, Arc::clone(&self.inner), texts)
    }
    fn chunk_async<'p>(&self, py: Python<'p>, text: String) -> PyResult<Bound<'p, PyAny>> {
        do_chunk_async(py, Arc::clone(&self.inner), text)
    }
    fn chunk_many_async<'p>(&self, py: Python<'p>, texts: Vec<String>) -> PyResult<Bound<'p, PyAny>> {
        do_chunk_many_async(py, Arc::clone(&self.inner), texts)
    }
}

#[pyclass(name = "MarkdownChunker", module = "konan", frozen)]
pub struct PyMarkdownChunker {
    inner: Arc<dyn Chunker>,
}

#[pymethods]
impl PyMarkdownChunker {
    #[new]
    #[pyo3(signature = (chunk_size=1000, chunk_overlap=200))]
    fn new(chunk_size: usize, chunk_overlap: usize) -> PyResult<Self> {
        Ok(Self {
            inner: Arc::new(MarkdownChunker::new(chunk_size, chunk_overlap).map_err(to_py_err)?),
        })
    }
    fn chunk(&self, py: Python<'_>, text: String) -> PyResult<Vec<PyChunk>> {
        do_chunk(py, Arc::clone(&self.inner), text)
    }
    fn chunk_many(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<Vec<Vec<PyChunk>>> {
        do_chunk_many(py, Arc::clone(&self.inner), texts)
    }
    fn chunk_async<'p>(&self, py: Python<'p>, text: String) -> PyResult<Bound<'p, PyAny>> {
        do_chunk_async(py, Arc::clone(&self.inner), text)
    }
    fn chunk_many_async<'p>(&self, py: Python<'p>, texts: Vec<String>) -> PyResult<Bound<'p, PyAny>> {
        do_chunk_many_async(py, Arc::clone(&self.inner), texts)
    }
}

#[pyclass(name = "TokenChunker", module = "konan", frozen)]
pub struct PyTokenChunker {
    inner: Arc<dyn Chunker>,
}

#[pymethods]
impl PyTokenChunker {
    #[new]
    #[pyo3(signature = (chunk_size=512, chunk_overlap=64, encoding="cl100k_base"))]
    fn new(chunk_size: usize, chunk_overlap: usize, encoding: &str) -> PyResult<Self> {
        Ok(Self {
            inner: Arc::new(
                TokenChunker::new(chunk_size, chunk_overlap, encoding).map_err(to_py_err)?,
            ),
        })
    }
    fn chunk(&self, py: Python<'_>, text: String) -> PyResult<Vec<PyChunk>> {
        do_chunk(py, Arc::clone(&self.inner), text)
    }
    fn chunk_many(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<Vec<Vec<PyChunk>>> {
        do_chunk_many(py, Arc::clone(&self.inner), texts)
    }
    fn chunk_async<'p>(&self, py: Python<'p>, text: String) -> PyResult<Bound<'p, PyAny>> {
        do_chunk_async(py, Arc::clone(&self.inner), text)
    }
    fn chunk_many_async<'p>(&self, py: Python<'p>, texts: Vec<String>) -> PyResult<Bound<'p, PyAny>> {
        do_chunk_many_async(py, Arc::clone(&self.inner), texts)
    }
}
```

- [ ] **Step 5: Update `lib.rs`**

```rust
use pyo3::prelude::*;

mod chunk;
mod chunkers;
mod common;
mod errors;

#[pymodule]
fn _konan(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<chunk::PyChunk>()?;
    m.add_class::<chunkers::PyNaiveChunker>()?;
    m.add_class::<chunkers::PyFixedSizeChunker>()?;
    m.add_class::<chunkers::PyRecursiveChunker>()?;
    m.add_class::<chunkers::PySentenceChunker>()?;
    m.add_class::<chunkers::PyMarkdownChunker>()?;
    m.add_class::<chunkers::PyTokenChunker>()?;
    m.add("EmbeddingError", py.get_type::<errors::EmbeddingError>())?;
    Ok(())
}
```

- [ ] **Step 6: Build & smoke-test**

Run: `uv run maturin develop --uv && uv run python -c "from konan._konan import RecursiveChunker; print(RecursiveChunker(chunk_size=10, chunk_overlap=0).chunk('hello world again'))"`
Expected: prints a list of Chunk reprs

- [ ] **Step 7: Commit**

```bash
git add crates/konan-py
git commit -m "feat(py): bind six chunkers with sync, parallel and async methods"
```

---

### Task 13: Bindings — SemanticChunker, OpenAIEmbedder, Python-callable embedder

**Files:**
- Create: `crates/konan-py/src/semantic.rs`
- Modify: `crates/konan-py/src/lib.rs`

- [ ] **Step 1: `semantic.rs`**

```rust
use std::sync::Arc;

use async_trait::async_trait;
use konan_core::{Embedder, KonanError, OpenAIEmbedder, SemanticChunker};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::chunk::PyChunk;
use crate::errors::to_py_err;

#[pyclass(name = "OpenAIEmbedder", module = "konan", frozen)]
#[derive(Clone)]
pub struct PyOpenAIEmbedder {
    pub(crate) inner: Arc<OpenAIEmbedder>,
}

#[pymethods]
impl PyOpenAIEmbedder {
    #[new]
    #[pyo3(signature = (base_url, model, api_key=None, batch_size=128))]
    fn new(base_url: String, model: String, api_key: Option<String>, batch_size: usize) -> PyResult<Self> {
        Ok(Self {
            inner: Arc::new(
                OpenAIEmbedder::new(base_url, model, api_key, batch_size).map_err(to_py_err)?,
            ),
        })
    }

    /// Embed texts directly — handy for verifying the endpoint.
    fn embed_async<'p>(&self, py: Python<'p>, texts: Vec<String>) -> PyResult<Bound<'p, PyAny>> {
        let inner = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner.embed(&texts).await.map_err(to_py_err)
        })
    }
}

/// Adapter: a Python `async def (list[str]) -> list[list[float]]` callable
/// plugged into the core Embedder port. Async-only — sync chunk() cannot
/// drive a Python coroutine without a running asyncio loop.
struct PyCallableEmbedder {
    callable: Py<PyAny>,
}

#[async_trait]
impl Embedder for PyCallableEmbedder {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, KonanError> {
        let texts = texts.to_vec();
        let fut = Python::with_gil(|py| {
            let coro = self
                .callable
                .bind(py)
                .call1((texts,))
                .map_err(|e| KonanError::Embedding(e.to_string()))?;
            pyo3_async_runtimes::tokio::into_future(coro)
                .map_err(|e| KonanError::Embedding(e.to_string()))
        })?;
        let result = fut.await.map_err(|e| KonanError::Embedding(e.to_string()))?;
        Python::with_gil(|py| {
            result
                .bind(py)
                .extract::<Vec<Vec<f32>>>()
                .map_err(|e| KonanError::Embedding(e.to_string()))
        })
    }
}

#[pyclass(name = "SemanticChunker", module = "konan", frozen)]
pub struct PySemanticChunker {
    inner: Arc<SemanticChunker<Arc<dyn Embedder>>>,
}

#[pymethods]
impl PySemanticChunker {
    #[new]
    #[pyo3(signature = (embedder, threshold=None, percentile=95.0, min_chunk_size=0, max_chunk_size=None))]
    fn new(
        embedder: Bound<'_, PyAny>,
        threshold: Option<f32>,
        percentile: f32,
        min_chunk_size: usize,
        max_chunk_size: Option<usize>,
    ) -> PyResult<Self> {
        let port: Arc<dyn Embedder> = if let Ok(native) = embedder.extract::<PyOpenAIEmbedder>() {
            Arc::clone(&native.inner) as Arc<dyn Embedder>
        } else if embedder.is_callable() {
            Arc::new(PyCallableEmbedder { callable: embedder.unbind() })
        } else {
            return Err(PyValueError::new_err(
                "embedder must be an OpenAIEmbedder or an async callable",
            ));
        };
        let inner = SemanticChunker::new(port, threshold, percentile, min_chunk_size, max_chunk_size)
            .map_err(to_py_err)?;
        Ok(Self { inner: Arc::new(inner) })
    }

    fn chunk(&self, py: Python<'_>, text: String) -> PyResult<Vec<PyChunk>> {
        let inner = Arc::clone(&self.inner);
        let chunks = py
            .allow_threads(move || {
                pyo3_async_runtimes::tokio::get_runtime().block_on(inner.chunk(&text))
            })
            .map_err(to_py_err)?;
        Ok(chunks.into_iter().map(PyChunk::from).collect())
    }

    fn chunk_many(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<Vec<Vec<PyChunk>>> {
        let inner = Arc::clone(&self.inner);
        let results = py
            .allow_threads(move || {
                pyo3_async_runtimes::tokio::get_runtime().block_on(inner.chunk_many(&texts))
            })
            .map_err(to_py_err)?;
        Ok(results
            .into_iter()
            .map(|cs| cs.into_iter().map(PyChunk::from).collect())
            .collect())
    }

    fn chunk_async<'p>(&self, py: Python<'p>, text: String) -> PyResult<Bound<'p, PyAny>> {
        let inner = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let chunks = inner.chunk(&text).await.map_err(to_py_err)?;
            Ok(chunks.into_iter().map(PyChunk::from).collect::<Vec<_>>())
        })
    }

    fn chunk_many_async<'p>(&self, py: Python<'p>, texts: Vec<String>) -> PyResult<Bound<'p, PyAny>> {
        let inner = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let results = inner.chunk_many(&texts).await.map_err(to_py_err)?;
            Ok(results
                .into_iter()
                .map(|cs| cs.into_iter().map(PyChunk::from).collect::<Vec<PyChunk>>())
                .collect::<Vec<_>>())
        })
    }
}
```

- [ ] **Step 2: Wire `lib.rs`** — add `mod semantic;` and in `_konan`:

```rust
    m.add_class::<semantic::PyOpenAIEmbedder>()?;
    m.add_class::<semantic::PySemanticChunker>()?;
```

- [ ] **Step 3: Build**

Run: `uv run maturin develop --uv`
Expected: builds clean

- [ ] **Step 4: Commit**

```bash
git add crates/konan-py
git commit -m "feat(py): SemanticChunker, OpenAIEmbedder, Python-callable embedder port"
```

---

### Task 14: Python package surface — `__init__.py`, stubs, `py.typed`

**Files:**
- Modify: `python/konan/__init__.py`
- Create: `python/konan/py.typed`, `python/konan/_konan.pyi`

- [ ] **Step 1: `python/konan/__init__.py`**

```python
"""konan — blazingly fast text chunkers, forged in Rust."""

from konan._konan import (
    Chunk,
    EmbeddingError,
    FixedSizeChunker,
    MarkdownChunker,
    NaiveChunker,
    OpenAIEmbedder,
    RecursiveChunker,
    SemanticChunker,
    SentenceChunker,
    TokenChunker,
    __version__,
)

__all__ = [
    "Chunk",
    "EmbeddingError",
    "FixedSizeChunker",
    "MarkdownChunker",
    "NaiveChunker",
    "OpenAIEmbedder",
    "RecursiveChunker",
    "SemanticChunker",
    "SentenceChunker",
    "TokenChunker",
    "__version__",
]
```

- [ ] **Step 2: `python/konan/py.typed`** — empty file.

- [ ] **Step 3: `python/konan/_konan.pyi`**

```python
from collections.abc import Awaitable, Callable, Sequence

__version__: str

class Chunk:
    @property
    def text(self) -> str: ...
    @property
    def start(self) -> int: ...
    @property
    def end(self) -> int: ...
    @property
    def index(self) -> int: ...
    @property
    def hash(self) -> str: ...
    def __len__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

class EmbeddingError(Exception): ...

class NaiveChunker:
    def __init__(self, chunk_size: int = 200) -> None: ...
    def chunk(self, text: str) -> list[Chunk]: ...
    def chunk_many(self, texts: Sequence[str]) -> list[list[Chunk]]: ...
    async def chunk_async(self, text: str) -> list[Chunk]: ...
    async def chunk_many_async(self, texts: Sequence[str]) -> list[list[Chunk]]: ...

class FixedSizeChunker:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        respect_sentences: bool = True,
    ) -> None: ...
    def chunk(self, text: str) -> list[Chunk]: ...
    def chunk_many(self, texts: Sequence[str]) -> list[list[Chunk]]: ...
    async def chunk_async(self, text: str) -> list[Chunk]: ...
    async def chunk_many_async(self, texts: Sequence[str]) -> list[list[Chunk]]: ...

class RecursiveChunker:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
    ) -> None: ...
    def chunk(self, text: str) -> list[Chunk]: ...
    def chunk_many(self, texts: Sequence[str]) -> list[list[Chunk]]: ...
    async def chunk_async(self, text: str) -> list[Chunk]: ...
    async def chunk_many_async(self, texts: Sequence[str]) -> list[list[Chunk]]: ...

class SentenceChunker:
    def __init__(self, max_chars: int = 1000, overlap_sentences: int = 1) -> None: ...
    def chunk(self, text: str) -> list[Chunk]: ...
    def chunk_many(self, texts: Sequence[str]) -> list[list[Chunk]]: ...
    async def chunk_async(self, text: str) -> list[Chunk]: ...
    async def chunk_many_async(self, texts: Sequence[str]) -> list[list[Chunk]]: ...

class MarkdownChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None: ...
    def chunk(self, text: str) -> list[Chunk]: ...
    def chunk_many(self, texts: Sequence[str]) -> list[list[Chunk]]: ...
    async def chunk_async(self, text: str) -> list[Chunk]: ...
    async def chunk_many_async(self, texts: Sequence[str]) -> list[list[Chunk]]: ...

class TokenChunker:
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        encoding: str = "cl100k_base",
    ) -> None: ...
    def chunk(self, text: str) -> list[Chunk]: ...
    def chunk_many(self, texts: Sequence[str]) -> list[list[Chunk]]: ...
    async def chunk_async(self, text: str) -> list[Chunk]: ...
    async def chunk_many_async(self, texts: Sequence[str]) -> list[list[Chunk]]: ...

class OpenAIEmbedder:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        batch_size: int = 128,
    ) -> None: ...
    async def embed_async(self, texts: Sequence[str]) -> list[list[float]]: ...

EmbedderLike = OpenAIEmbedder | Callable[[list[str]], Awaitable[list[list[float]]]]

class SemanticChunker:
    def __init__(
        self,
        embedder: EmbedderLike,
        threshold: float | None = None,
        percentile: float = 95.0,
        min_chunk_size: int = 0,
        max_chunk_size: int | None = None,
    ) -> None: ...
    def chunk(self, text: str) -> list[Chunk]: ...
    def chunk_many(self, texts: Sequence[str]) -> list[list[Chunk]]: ...
    async def chunk_async(self, text: str) -> list[Chunk]: ...
    async def chunk_many_async(self, texts: Sequence[str]) -> list[list[Chunk]]: ...
```

- [ ] **Step 4: Rebuild & verify imports**

Run: `uv run maturin develop --uv && uv run python -c "import konan; print(sorted(konan.__all__))"`
Expected: lists all 10 exports + `__version__`

- [ ] **Step 5: Commit**

```bash
git add python
git commit -m "feat(py): package surface with typed stubs and py.typed"
```

---

### Task 15: Python integration tests

**Files:**
- Create: `tests/conftest.py`, `tests/test_chunkers.py`, `tests/test_async.py`, `tests/test_semantic.py`

- [ ] **Step 1: `tests/conftest.py` (mock OpenAI-compatible endpoint)**

```python
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest


def _embed(text: str) -> list[float]:
    # Deterministic 2D embeddings: cat-topic vs everything else.
    return [1.0, 0.0] if "cat" in text.lower() else [0.0, 1.0]


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        data = [
            {"object": "embedding", "index": i, "embedding": _embed(t)}
            for i, t in enumerate(payload["input"])
        ]
        body = json.dumps(
            {"object": "list", "model": payload["model"], "data": data}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture(scope="session")
def embeddings_url():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1"
    server.shutdown()
```

- [ ] **Step 2: `tests/test_chunkers.py`**

````python
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
````

- [ ] **Step 3: `tests/test_async.py`**

```python
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
```

- [ ] **Step 4: `tests/test_semantic.py`**

```python
import pytest
from konan import EmbeddingError, OpenAIEmbedder, SemanticChunker

TEXT = (
    "Cats purr softly. My cat naps all day. "
    "Quantum entanglement defies intuition. Quantum computers exploit superposition."
)


async def test_semantic_chunk_async(embeddings_url):
    chunker = SemanticChunker(
        embedder=OpenAIEmbedder(base_url=embeddings_url, model="test"), threshold=0.5
    )
    chunks = await chunker.chunk_async(TEXT)
    assert len(chunks) == 2
    assert "cat" in chunks[0].text.lower()
    assert chunks[1].text.startswith("Quantum")
    for c in chunks:
        assert TEXT[c.start : c.end] == c.text


def test_semantic_chunk_sync(embeddings_url):
    chunker = SemanticChunker(
        embedder=OpenAIEmbedder(base_url=embeddings_url, model="test"), threshold=0.5
    )
    assert len(chunker.chunk(TEXT)) == 2


async def test_embedder_direct(embeddings_url):
    embedder = OpenAIEmbedder(base_url=embeddings_url, model="test")
    vectors = await embedder.embed_async(["a cat", "physics"])
    assert vectors == [[1.0, 0.0], [0.0, 1.0]]


async def test_python_callable_embedder():
    async def embed(texts: list[str]) -> list[list[float]]:
        return [
            [1.0, 0.0] if "cat" in t.lower() else [0.0, 1.0] for t in texts
        ]

    chunker = SemanticChunker(embedder=embed, threshold=0.5)
    chunks = await chunker.chunk_async(TEXT)
    assert len(chunks) == 2


async def test_embedding_error_on_unreachable_endpoint():
    chunker = SemanticChunker(
        embedder=OpenAIEmbedder(base_url="http://127.0.0.1:9", model="x"), threshold=0.5
    )
    with pytest.raises(EmbeddingError):
        await chunker.chunk_async("One sentence. Two sentences. Three now.")


def test_rejects_non_embedder():
    with pytest.raises(ValueError):
        SemanticChunker(embedder=42)
```

- [ ] **Step 5: Run the suite**

Run: `uv run maturin develop --uv && uv run pytest -q`
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add tests
git commit -m "test(py): integration tests for sync, parallel, async and semantic chunking"
```

---

### Task 16: README

**Files:**
- Rewrite: `README.md`

- [ ] **Step 1: Write `README.md`**

`````markdown
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
`````

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: modern README with logo, strategy table and examples"
```

---

### Task 17: Final verification

- [ ] **Step 1: Full check**

Run: `cargo test --workspace && uv run maturin develop --uv && uv run pytest -q`
Expected: everything green

- [ ] **Step 2: Stub/API drift check**

Run: `uv run python - <<'EOF'
import inspect, konan
expected = {"chunk", "chunk_many", "chunk_async", "chunk_many_async"}
for name in ["NaiveChunker", "FixedSizeChunker", "RecursiveChunker", "SentenceChunker", "MarkdownChunker", "TokenChunker", "SemanticChunker"]:
    cls = getattr(konan, name)
    missing = expected - set(dir(cls))
    assert not missing, f"{name} missing {missing}"
print("API surface OK")
EOF`
Expected: `API surface OK`

- [ ] **Step 3: Commit any stragglers**

```bash
git add -A
git commit -m "chore: final verification fixes" || echo "nothing to commit"
```
