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
