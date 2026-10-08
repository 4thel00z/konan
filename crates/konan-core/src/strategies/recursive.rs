use std::sync::Arc;

use crate::chunk::Chunk;
use crate::chunker::Chunker;
use crate::error::KonanError;
use crate::size::{Measure, SizeUnit};
use crate::text::{merge_spans, spans_to_chunks, stable_merge, OffsetMap};

/// Recursively splits text using a hierarchy of separators
/// (paragraph -> line -> space -> punctuation -> chars), then merges
/// pieces into chunks with overlap. LangChain-compatible semantics.
pub struct RecursiveChunker {
    chunk_size: usize,
    chunk_overlap: usize,
    separators: Vec<String>,
    size_unit: SizeUnit,
    stable_boundaries: bool,
}

impl RecursiveChunker {
    pub fn new(
        chunk_size: usize,
        chunk_overlap: usize,
        separators: Option<Vec<String>>,
    ) -> Result<Self, KonanError> {
        if chunk_size == 0 {
            return Err(KonanError::InvalidConfig("chunk_size must be > 0".into()));
        }
        if chunk_overlap >= chunk_size {
            return Err(KonanError::InvalidConfig(
                "chunk_overlap must be smaller than chunk_size".into(),
            ));
        }
        let separators = separators.unwrap_or_else(|| {
            ["\n\n", "\n", " ", ".", ",", ""]
                .iter()
                .map(|s| s.to_string())
                .collect()
        });
        Ok(Self {
            chunk_size,
            chunk_overlap,
            separators,
            size_unit: SizeUnit::Chars,
            stable_boundaries: false,
        })
    }

    /// Measures `chunk_size` and `chunk_overlap` in `size_unit` instead of chars.
    pub fn with_size_unit(mut self, size_unit: SizeUnit) -> Self {
        self.size_unit = size_unit;
        self
    }

    /// Picks cut points from chunk content instead of filling each chunk
    /// greedily, so an edit only changes the chunks near it.
    pub fn with_stable_boundaries(mut self, stable_boundaries: bool) -> Self {
        self.stable_boundaries = stable_boundaries;
        self
    }

    /// Split `span` into unit spans each at most `chunk_size` units of
    /// `measure` (or unsplittable), using the separator hierarchy starting at
    /// `sep_idx`. Each piece keeps its trailing separator so chunks reproduce
    /// the source.
    pub(crate) fn split_units(
        &self,
        text: &str,
        measure: &impl Measure,
        span: (usize, usize),
        sep_idx: usize,
        out: &mut Vec<(usize, usize)>,
    ) {
        if measure.len(span.0, span.1) <= self.chunk_size || sep_idx >= self.separators.len() {
            out.push(span);
            return;
        }
        let sep = &self.separators[sep_idx];
        if sep.is_empty() {
            measure.split_even(text, span, self.chunk_size, out);
            return;
        }
        let slice = &text[span.0..span.1];
        let mut piece_start = span.0;
        let mut search_from = 0;
        let mut found = false;
        while let Some(pos) = slice[search_from..].find(sep.as_str()) {
            found = true;
            let sep_end = span.0 + search_from + pos + sep.len();
            if sep_end > piece_start {
                self.split_units(text, measure, (piece_start, sep_end), sep_idx + 1, out);
            }
            piece_start = sep_end;
            search_from = search_from + pos + sep.len();
        }
        if !found {
            self.split_units(text, measure, span, sep_idx + 1, out);
            return;
        }
        if piece_start < span.1 {
            self.split_units(text, measure, (piece_start, span.1), sep_idx + 1, out);
        }
    }
}

impl Chunker for RecursiveChunker {
    fn chunk_shared(&self, src: &Arc<str>) -> Result<Vec<Chunk>, KonanError> {
        let text: &str = src;
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }
        let map = OffsetMap::new(text);
        let merged = match self.size_unit.token_measure(text)? {
            None => self.merged_spans(text, &map),
            Some(tokens) => self.merged_spans(text, &tokens),
        };
        Ok(spans_to_chunks(src, &map, &merged))
    }
}

impl RecursiveChunker {
    fn merged_spans(&self, text: &str, measure: &impl Measure) -> Vec<(usize, usize)> {
        let mut units = Vec::new();
        self.split_units(text, measure, (0, text.len()), 0, &mut units);
        if self.stable_boundaries {
            return stable_merge(
                measure,
                text,
                units.into_iter(),
                self.chunk_size,
                self.chunk_overlap,
            );
        }
        merge_spans(measure, &units, self.chunk_size, self.chunk_overlap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::size::Encoding;
    use crate::text::{
        assert_char_offsets, assert_token_budget, insert_mid_sentence, new_chunks, varied_prose,
    };

    #[test]
    fn prefers_paragraph_breaks() {
        let text = "Para one is here.\n\nPara two is here.\n\nPara three is here.";
        let chunks = RecursiveChunker::new(25, 0, None)
            .unwrap()
            .chunk(text)
            .unwrap();
        assert_eq!(chunks.len(), 3);
        assert!(chunks[0].text().starts_with("Para one"));
        assert!(chunks[1].text().starts_with("Para two"));
        assert_char_offsets(text, &chunks);
    }

    #[test]
    fn falls_back_to_words_then_chars() {
        let text = "abcdefghij klmnopqrst";
        let chunks = RecursiveChunker::new(5, 0, None)
            .unwrap()
            .chunk(text)
            .unwrap();
        for c in &chunks {
            assert!(c.end - c.start <= 5);
        }
        assert_char_offsets(text, &chunks);
    }

    #[test]
    fn overlap_carries_units() {
        let text = "alpha beta gamma delta epsilon zeta eta theta iota kappa";
        let chunks = RecursiveChunker::new(20, 8, None)
            .unwrap()
            .chunk(text)
            .unwrap();
        assert!(chunks.len() >= 2);
        assert!(chunks[1].start < chunks[0].end);
        assert_char_offsets(text, &chunks);
    }

    #[test]
    fn custom_separators() {
        let text = "a|b|c|d";
        let seps = Some(vec!["|".to_string(), "".to_string()]);
        let chunks = RecursiveChunker::new(3, 0, seps)
            .unwrap()
            .chunk(text)
            .unwrap();
        assert!(chunks.len() >= 2);
        assert_char_offsets(text, &chunks);
    }

    #[test]
    fn unicode_safe_char_fallback() {
        let text = "😀😁😂🤣😃😄😅😆😉😊";
        let chunks = RecursiveChunker::new(3, 0, None)
            .unwrap()
            .chunk(text)
            .unwrap();
        assert_eq!(chunks.len(), 4);
        assert_char_offsets(text, &chunks);
    }

    #[test]
    fn token_size_unit_fits_budget() {
        let text = varied_prose(600) + "😀😁😂🤣😃😄😅😆😉😊 schön müde";
        for encoding in [Encoding::Cl100kBase, Encoding::O200kBase] {
            let configs = [
                (16usize, 0usize, false),
                (64, 0, false),
                (256, 0, false),
                (16, 0, true),
                (64, 16, true),
                (256, 32, true),
            ];
            for (size, overlap, stable) in configs {
                let chunks = RecursiveChunker::new(size, overlap, None)
                    .unwrap()
                    .with_size_unit(SizeUnit::Tokens(encoding))
                    .with_stable_boundaries(stable)
                    .chunk(&text)
                    .unwrap();
                assert!(chunks.len() > 1);
                assert_char_offsets(&text, &chunks);
                assert_token_budget(encoding, &text, &chunks, size);
            }
        }
    }

    #[test]
    fn token_char_fallback_splits_long_words() {
        let text = "x".repeat(500) + " tail";
        let chunks = RecursiveChunker::new(8, 0, None)
            .unwrap()
            .with_size_unit(SizeUnit::Tokens(Encoding::Cl100kBase))
            .chunk(&text)
            .unwrap();
        assert!(chunks.len() > 3);
        assert_char_offsets(&text, &chunks);
        assert_token_budget(Encoding::Cl100kBase, &text, &chunks, 8);
    }

    #[test]
    fn stable_boundaries_survive_an_edit() {
        let text = varied_prose(1500);
        let edited = insert_mid_sentence(&text);
        let run = |stable: bool| {
            let chunker = RecursiveChunker::new(300, 0, None)
                .unwrap()
                .with_stable_boundaries(stable);
            new_chunks(
                &chunker.chunk(&text).unwrap(),
                &chunker.chunk(&edited).unwrap(),
            )
        };
        let greedy = run(false);
        let stable = run(true);
        assert!(
            greedy > 100,
            "the edit must disturb greedy packing: {greedy}"
        );
        assert!(stable <= 5, "stable boundaries changed {stable} chunks");
    }
}
