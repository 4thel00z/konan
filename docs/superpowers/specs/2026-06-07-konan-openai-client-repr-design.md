# konan — async-openai adapter, richer embedder configs, `__repr__` polish

**Date:** 2026-06-07
**Status:** Approved

## Purpose

Three follow-ups to the initial konan build:

1. Back `OpenAIEmbedder` with the de facto standard `async-openai` client instead
   of hand-rolled reqwest HTTP (typed request/response, maintained client).
2. Expose more OpenAI configs: `timeout`, `max_retries`, `dimensions`.
3. Add config-revealing `__repr__` to every pyclass and fix `Chunk.__repr__`'s
   silent truncation.

## 1. OpenAIEmbedder on async-openai (konan-core)

- `konan-core/Cargo.toml`: replace `reqwest` with `async-openai` (rustls
  features; reqwest stays as a transitive dep). Hand-rolled
  `EmbeddingsResponse`/`EmbeddingItem` structs and POST code are deleted.
- Client: `OpenAIConfig` with `api_base = base_url` (trailing `/` trimmed) and
  the api key (empty when `None`); custom `reqwest::Client` carrying the
  configured timeout.
- New constructor params (Rust, Python signature, `.pyi`):
  - `timeout: f64` seconds, default **30.0** (must be > 0)
  - `max_retries: u32`, default **2** — configures async-openai's own
    `OpenAIRetryLayer` (the client's default executor always retries 3×; we
    replace it via the `middleware` feature so the knob is honored). The layer
    retries 429 (except insufficient_quota), 5xx and connection errors with
    exponential backoff, honoring `Retry-After`.
  - `dimensions: Option<u32>`, default **None** — forwarded on the embeddings
    request when set.
- Unchanged: `Embedder` port, batching by `batch_size`, sort-by-index,
  count-mismatch guard, error mapping (`OpenAIError` →
  `KonanError::Embedding` → `konan.EmbeddingError`).

## 2. `__repr__` (konan-py)

- `Chunk`: append `…` to the preview when text exceeds 40 chars.
- Six chunker pyclasses store their construction config and render it, e.g.
  `RecursiveChunker(chunk_size=1000, chunk_overlap=200, separators=None)`.
- `OpenAIEmbedder`: all configs; `api_key='***'` when set, `None` otherwise.
- `SemanticChunker`: threshold/percentile/min/max plus the embedder's repr
  (`<async callable>` for Python embedders).

## 3. Tests

- pytest: repr assertions; `dimensions` passthrough (mock returns vectors of
  the requested length); retry behavior (stateful mock: 500 then success →
  succeeds; `max_retries=0` → `EmbeddingError`).
- conftest mock gains the `usage` field (required by async-openai's response
  type) and reads `dimensions` from the payload.
- Rust: constructor validation for the new params; the deleted-struct parsing
  test is removed; unreachable-endpoint test stays.
- README + `.pyi` stubs updated for the new signature.
