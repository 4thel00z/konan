# Changelog

## [0.2.2](https://github.com/4thel00z/konan/compare/v0.2.1...v0.2.2) (2026-06-07)


### Bug Fixes

* include workspace-root LICENSE in sdist, skip-existing on publish ([84a090a](https://github.com/4thel00z/konan/commit/84a090abb2997088f9543103671fa2a2fcb4f5e7))

## [0.2.1](https://github.com/4thel00z/konan/compare/v0.2.0...v0.2.1) (2026-06-07)


### Performance Improvements

* direct char counting for non-ASCII text + wider benchmark field with plots ([1c355a5](https://github.com/4thel00z/konan/commit/1c355a54168638be2797474d4322c49eb08b95ad))

## [0.2.0](https://github.com/4thel00z/konan/compare/v0.1.0...v0.2.0) (2026-06-07)


### Features

* async-openai embedder backend, richer configs, config-revealing reprs ([6caf2c2](https://github.com/4thel00z/konan/commit/6caf2c23dec453acf101722ca02ad9fe511a9d2b))
* benchmarks — bench.py, criterion benches, README results ([5ed3bfa](https://github.com/4thel00z/konan/commit/5ed3bfaaa8f3a0d2f9cf26cf8fafc123c373fec2))
* **core:** Chunk, KonanError, span/offset utilities ([3f40368](https://github.com/4thel00z/konan/commit/3f403686d5d1dcd33c6ce3f707e838ec3136df6d))
* **core:** Chunker port and rayon chunk_many ([f6a62d1](https://github.com/4thel00z/konan/commit/f6a62d1cd8554a46bf3ff47da11d3ffe7c599295))
* **core:** Embedder port and OpenAI-compatible adapter ([7a7d7c2](https://github.com/4thel00z/konan/commit/7a7d7c2fc7d9b6034f5f612ce7305d54a040468b))
* **core:** FixedSizeChunker with overlap and sentence awareness ([fe36188](https://github.com/4thel00z/konan/commit/fe36188f993ba7010609b630710ef0c4c7bdf39d))
* **core:** MarkdownChunker with breadcrumbs and atomic code fences ([c4bd62b](https://github.com/4thel00z/konan/commit/c4bd62b88527f44067d3c2d1833ed79feaa55a43))
* **core:** NaiveChunker (word-based) ([99f2738](https://github.com/4thel00z/konan/commit/99f2738dbce4498f9ab5b0c91acf5a182718d86e))
* **core:** RecursiveChunker with separator hierarchy ([3a1cbb1](https://github.com/4thel00z/konan/commit/3a1cbb10ef9aaededf9f490cfc847d10e7b9b5b2))
* **core:** SemanticChunker over the Embedder port ([22621f8](https://github.com/4thel00z/konan/commit/22621f8a1e821536c0975f519be133328901dac6))
* **core:** SentenceChunker (unicode segmentation) ([bf57195](https://github.com/4thel00z/konan/commit/bf571952d77dd1fba71bca7389f8e57f4e836901))
* **core:** TokenChunker (tiktoken cl100k/o200k) ([d24ef27](https://github.com/4thel00z/konan/commit/d24ef27a50eec64216634ad348ac27d3a945595e))
* **py:** bind six chunkers with sync, parallel and async methods ([f115929](https://github.com/4thel00z/konan/commit/f1159293275de677623103402db62e01955df030))
* **py:** package surface with typed stubs and py.typed ([b5fb787](https://github.com/4thel00z/konan/commit/b5fb787bfb857c9283a1e814d072fb1b0c4b50bb))
* **py:** SemanticChunker, OpenAIEmbedder, Python-callable embedder port ([a8324ba](https://github.com/4thel00z/konan/commit/a8324ba15228728f875f81a23e65dbf61e594df8))
* scaffold konan workspace (konan-core + konan-py, maturin) ([8e2e74f](https://github.com/4thel00z/konan/commit/8e2e74f840fe5002d30871e736549dad80af6893))


### Bug Fixes

* absolute logo URL so it renders on PyPI ([618e185](https://github.com/4thel00z/konan/commit/618e18568299b4e96bd9954673f07b85b8c6bc20))
* clippy is_multiple_of lint in benches (CI stable toolchain) ([1e7bb44](https://github.com/4thel00z/konan/commit/1e7bb446561bfa25c049511c44b000002a1ef7ef))
* **core:** MarkdownChunker setext heading breadcrumbs ([a242ce4](https://github.com/4thel00z/konan/commit/a242ce4c45f7e0bcf0dea14f305e22a7715ae3b9))
* harden semantic chunking edge cases from review backlog ([9e8e673](https://github.com/4thel00z/konan/commit/9e8e673495fdde051bbca5910b8f69f07954ba29))


### Performance Improvements

* bpe-openai tokenizer + ASCII OffsetMap fast path ([e877ea2](https://github.com/4thel00z/konan/commit/e877ea222c25bad754370467ea9dac1c098bd88e))


### Documentation

* add async-openai adapter + repr polish design spec ([f396cbe](https://github.com/4thel00z/konan/commit/f396cbeed3215a99b0190b96eef385d51f81d745))
* add konan chunkers design spec ([dc9b24a](https://github.com/4thel00z/konan/commit/dc9b24afd978c91c7caf82157a1a43ba2bfef1f8))
* add konan chunkers implementation plan ([0384847](https://github.com/4thel00z/konan/commit/038484780b625c596c2a067244a2bee051caf1d7))
* modern README with logo, strategy table and examples ([d3106e3](https://github.com/4thel00z/konan/commit/d3106e340022336658fc5f5f592a975c2414596b))
