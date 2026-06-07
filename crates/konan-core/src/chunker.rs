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
