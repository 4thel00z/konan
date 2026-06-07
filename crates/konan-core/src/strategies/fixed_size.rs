use crate::chunk::Chunk;
use crate::chunker::Chunker;
use crate::error::KonanError;
use crate::text::{merge_span_iter, spans_to_chunks, OffsetMap, SentenceUnits, WordUnits};

/// Character-based chunker with overlap and sentence-boundary awareness.
///
/// Unit boundaries (sentences, or words as the fallback) are produced by
/// lazy memchr-accelerated scanners and merged on the fly — the merge
/// semantics are byte-identical to the original materialise-then-merge
/// implementation (enforced by the equivalence test below), but nothing
/// allocates per unit, which makes the strategy several times faster on
/// sentence-dense prose.
pub struct FixedSizeChunker {
    chunk_size: usize,
    chunk_overlap: usize,
    respect_sentences: bool,
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
        // Sentence mode applies when at least two sentence units exist
        // (the legacy `spans.len() > 1` check), probed lazily.
        let merged = if self.respect_sentences {
            let mut probe = SentenceUnits::new(text);
            let first = probe.next();
            let second = probe.next();
            match (first, second) {
                (Some(a), Some(b)) => {
                    let units = [a, b].into_iter().chain(probe);
                    merge_span_iter(&map, units, self.chunk_size, self.chunk_overlap)
                }
                _ => merge_span_iter(
                    &map,
                    WordUnits::new(text),
                    self.chunk_size,
                    self.chunk_overlap,
                ),
            }
        } else {
            merge_span_iter(
                &map,
                WordUnits::new(text),
                self.chunk_size,
                self.chunk_overlap,
            )
        };
        Ok(spans_to_chunks(text, &map, &merged))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::{assert_char_offsets, merge_spans, regex_sentence_spans, word_spans};

    const TEXT: &str =
        "First sentence here. Second sentence follows! Third one now? Fourth and final.";

    /// The previous regex-and-materialise implementation, kept as the
    /// oracle for the equivalence test.
    fn chunk_reference(
        text: &str,
        chunk_size: usize,
        chunk_overlap: usize,
        respect_sentences: bool,
    ) -> Vec<Chunk> {
        let sentence_re = regex::Regex::new(r"[.!?]+\s+").expect("static regex");
        if text.trim().is_empty() {
            return Vec::new();
        }
        let map = OffsetMap::new(text);
        if map.char_len(0, text.len()) <= chunk_size {
            return spans_to_chunks(text, &map, &[(0, text.len())]);
        }
        let units = if respect_sentences {
            let sents = regex_sentence_spans(text, &sentence_re);
            if sents.len() > 1 {
                sents
            } else {
                word_spans(text)
            }
        } else {
            word_spans(text)
        };
        let merged = merge_spans(&map, &units, chunk_size, chunk_overlap);
        spans_to_chunks(text, &map, &merged)
    }

    fn assert_equivalent(text: &str, chunk_size: usize, chunk_overlap: usize, respect: bool) {
        let new = FixedSizeChunker::new(chunk_size, chunk_overlap, respect)
            .unwrap()
            .chunk(text)
            .unwrap();
        let old = chunk_reference(text, chunk_size, chunk_overlap, respect);
        assert_eq!(
            new, old,
            "divergence: size={chunk_size} overlap={chunk_overlap} respect={respect} text={text:?}"
        );
    }

    #[test]
    fn matches_reference_implementation() {
        let corpora: Vec<String> = vec![
            TEXT.to_string(),
            "word ".repeat(50),
            "  leading whitespace. And more text here! Trailing too.   ".to_string(),
            "Short. This single sentence is much longer than the limit allows obviously. End."
                .to_string(),
            "One enormous sentence without any terminal punctuation just words all the way down "
                .repeat(4),
            "Multi?! Punct... runs!! Everywhere?! Yes... Indeed!".to_string(),
            "Ends mid.word and v1.2 stays whole. Real boundary here! ok.".to_string(),
            "Caf\u{e9} na\u{ef}ve \u{1f980} crab. \u{65e5}\u{672c}\u{8a9e}\u{306e}\u{6587}. More \u{fc}nicode here! Done."
                .repeat(8),
            "Nbsp\u{a0}separated. Ideographic\u{3000}space too! And tab\tnewline\nvtab\u{b}breaks. End."
                .to_string(),
            {
                let mut s = String::new();
                for i in 0..200 {
                    let len = 3 + (i * 7) % 11;
                    for k in 0..len {
                        s.push_str(["alpha", "beta", "gamma", "delta"][(i + k) % 4]);
                        s.push(if k + 1 == len { '.' } else { ' ' });
                    }
                    s.push(' ');
                    if i % 9 == 0 {
                        s.push_str("\n\n");
                    }
                }
                s
            },
        ];
        for text in &corpora {
            for &size in &[30usize, 45, 100, 400] {
                for &overlap in &[0usize, 10, 25] {
                    if overlap >= size {
                        continue;
                    }
                    for &respect in &[true, false] {
                        assert_equivalent(text, size, overlap, respect);
                    }
                }
            }
        }
    }

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
