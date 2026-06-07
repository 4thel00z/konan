//! konan-core: blazingly fast text chunking strategies.

pub mod chunk;
pub mod error;
pub(crate) mod text;

pub use chunk::Chunk;
pub use error::KonanError;
