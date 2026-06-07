//! konan-core: blazingly fast text chunking strategies.

pub mod chunk;
pub mod chunker;
pub mod embedder;
pub mod error;
pub mod strategies;
pub(crate) mod text;

pub use chunk::Chunk;
pub use chunker::{chunk_many, Chunker};
pub use embedder::{Embedder, OpenAIEmbedder};
pub use error::KonanError;
pub use strategies::fixed_size::FixedSizeChunker;
pub use strategies::markdown::MarkdownChunker;
pub use strategies::naive::NaiveChunker;
pub use strategies::recursive::RecursiveChunker;
pub use strategies::sentence::SentenceChunker;
pub use strategies::token::TokenChunker;
