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
        let chunks = FixedSizeChunker::new(45, 0, true)
            .unwrap()
            .chunk(TEXT)
            .unwrap();
        assert!(chunks.len() >= 2);
        for c in &chunks {
            assert!(
                c.text.ends_with(['.', '!', '?']),
                "broke mid-sentence: {:?}",
                c.text
            );
        }
        assert_char_offsets(TEXT, &chunks);
    }

    #[test]
    fn overlap_repeats_content() {
        let chunks = FixedSizeChunker::new(45, 25, true)
            .unwrap()
            .chunk(TEXT)
            .unwrap();
        assert!(chunks.len() >= 2);
        assert!(chunks[1].start < chunks[0].end, "no overlap produced");
        assert_char_offsets(TEXT, &chunks);
    }

    #[test]
    fn short_text_is_single_chunk() {
        let chunks = FixedSizeChunker::new(1000, 200, true)
            .unwrap()
            .chunk("tiny")
            .unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "tiny");
    }

    #[test]
    fn word_fallback_without_sentences() {
        let text = "word ".repeat(50);
        let chunks = FixedSizeChunker::new(60, 10, true)
            .unwrap()
            .chunk(&text)
            .unwrap();
        assert!(chunks.len() > 1);
        assert_char_offsets(&text, &chunks);
    }

    #[test]
    fn rejects_overlap_ge_size() {
        assert!(FixedSizeChunker::new(100, 100, true).is_err());
    }
}
