use crate::chunk::Chunk;
use crate::chunker::Chunker;
use crate::error::KonanError;
use crate::text::{sentence_spans, spans_to_chunks, OffsetMap};

/// Unicode-aware sentence chunker: groups whole sentences into chunks of at
/// most `max_chars`, overlapping by `overlap_sentences` sentences.
pub struct SentenceChunker {
    max_chars: usize,
    overlap_sentences: usize,
}

impl SentenceChunker {
    pub fn new(max_chars: usize, overlap_sentences: usize) -> Result<Self, KonanError> {
        if max_chars == 0 {
            return Err(KonanError::InvalidConfig("max_chars must be > 0".into()));
        }
        Ok(Self {
            max_chars,
            overlap_sentences,
        })
    }
}

impl Chunker for SentenceChunker {
    fn chunk(&self, text: &str) -> Result<Vec<Chunk>, KonanError> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }
        let map = OffsetMap::new(text);
        let sents = sentence_spans(text);
        let mut spans: Vec<(usize, usize)> = Vec::new();
        let mut start_idx = 0;
        while start_idx < sents.len() {
            let mut end_idx = start_idx;
            // Window char count grown incrementally: extending by one sentence
            // adds char_len(prev_end, next_end), gap included — identical to
            // re-measuring char_len(window_start, next_end) from scratch.
            let mut win = map.char_len(sents[start_idx].0, sents[start_idx].1);
            while end_idx + 1 < sents.len() {
                let added = map.char_len(sents[end_idx].1, sents[end_idx + 1].1);
                if win + added > self.max_chars {
                    break;
                }
                win += added;
                end_idx += 1;
            }
            spans.push((sents[start_idx].0, sents[end_idx].1));
            if end_idx + 1 >= sents.len() {
                break;
            }
            let mut next_start = (end_idx + 1).saturating_sub(self.overlap_sentences);
            if next_start <= start_idx {
                next_start = start_idx + 1; // guarantee progress
            }
            start_idx = next_start;
        }
        Ok(spans_to_chunks(text, &map, &spans))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::assert_char_offsets;

    const TEXT: &str =
        "Sentence number one. Sentence number two. Sentence number three. Sentence number four.";

    #[test]
    fn groups_whole_sentences() {
        let chunks = SentenceChunker::new(45, 0).unwrap().chunk(TEXT).unwrap();
        assert!(chunks.len() >= 2);
        for c in &chunks {
            assert!(c.text.starts_with("Sentence"));
            assert!(c.text.ends_with('.'));
        }
        assert_char_offsets(TEXT, &chunks);
    }

    #[test]
    fn sentence_overlap() {
        let chunks = SentenceChunker::new(45, 1).unwrap().chunk(TEXT).unwrap();
        assert!(chunks.len() >= 2);
        // chunk 1 starts at the last sentence of chunk 0
        assert!(chunks[1].start < chunks[0].end);
        assert_char_offsets(TEXT, &chunks);
    }

    #[test]
    fn oversized_sentence_emitted_alone() {
        let text = "Short. This single sentence is much longer than the limit allows. End.";
        let chunks = SentenceChunker::new(20, 0).unwrap().chunk(text).unwrap();
        assert!(chunks.iter().any(|c| c.text.contains("much longer")));
        assert_char_offsets(text, &chunks);
    }
}
