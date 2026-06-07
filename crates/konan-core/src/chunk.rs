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
