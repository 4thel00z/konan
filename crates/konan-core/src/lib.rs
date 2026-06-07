//! konan-core: blazingly fast text chunking strategies.

pub mod chunk;
pub mod chunker;
pub mod error;
pub(crate) mod text;

pub use chunk::Chunk;
pub use chunker::{chunk_many, Chunker};
pub use error::KonanError;
