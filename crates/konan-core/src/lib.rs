//! konan-core: blazingly fast text chunking strategies.

pub mod chunk;
pub mod chunker;
pub mod embedder;
pub mod error;
#[cfg(feature = "hf-tokenizers")]
pub mod hf;
pub mod semantic;
pub mod size;
pub mod strategies;
pub(crate) mod text;

pub use chunk::Chunk;
pub use chunker::{chunk_many, Chunker};
pub use embedder::{Embedder, OpenAIEmbedder};
pub use error::KonanError;
#[cfg(feature = "hf-tokenizers")]
pub use hf::HfTokenizer;
pub use semantic::SemanticChunker;
pub use size::{Encoding, SizeUnit, TokenEnds};
pub use strategies::fixed_size::FixedSizeChunker;
pub use strategies::markdown::MarkdownChunker;
pub use strategies::naive::NaiveChunker;
pub use strategies::recursive::RecursiveChunker;
pub use strategies::sentence::SentenceChunker;
pub use strategies::token::TokenChunker;
