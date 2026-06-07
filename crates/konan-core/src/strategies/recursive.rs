use crate::chunk::Chunk;
use crate::chunker::Chunker;
use crate::error::KonanError;
use crate::text::{merge_spans, spans_to_chunks, OffsetMap};

/// Recursively splits text using a hierarchy of separators
/// (paragraph -> line -> space -> punctuation -> chars), then merges
/// pieces into chunks with overlap. LangChain-compatible semantics.
pub struct RecursiveChunker {
    chunk_size: usize,
    chunk_overlap: usize,
    separators: Vec<String>,
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
            ["\n\n", "\n", " ", ".", ",", ""].iter().map(|s| s.to_string()).collect()
        });
        Ok(Self { chunk_size, chunk_overlap, separators })
    }

    /// Split `span` into unit spans each at most `chunk_size` chars (or
    /// unsplittable), using the separator hierarchy starting at `sep_idx`.
    /// Each piece keeps its trailing separator so chunks reproduce the source.
    pub(crate) fn split_units(
        &self,
        text: &str,
        map: &OffsetMap,
        span: (usize, usize),
        sep_idx: usize,
        out: &mut Vec<(usize, usize)>,
    ) {
        if map.char_len(span.0, span.1) <= self.chunk_size || sep_idx >= self.separators.len() {
            out.push(span);
            return;
        }
        let sep = &self.separators[sep_idx];
        if sep.is_empty() {
            // Last resort: split into chunk_size-char pieces.
            let mut start = span.0;
            let mut count = 0;
            for (b, _) in text[span.0..span.1].char_indices() {
                let abs = span.0 + b;
                if count == self.chunk_size {
                    out.push((start, abs));
                    start = abs;
                    count = 0;
                }
                count += 1;
            }
            if start < span.1 {
                out.push((start, span.1));
            }
            return;
        }
        let slice = &text[span.0..span.1];
        if !slice.contains(sep.as_str()) {
            self.split_units(text, map, span, sep_idx + 1, out);
            return;
        }
        let mut piece_start = span.0;
        let mut search_from = 0;
        while let Some(pos) = slice[search_from..].find(sep.as_str()) {
            let sep_end = span.0 + search_from + pos + sep.len();
            if sep_end > piece_start {
                self.split_units(text, map, (piece_start, sep_end), sep_idx + 1, out);
            }
            piece_start = sep_end;
            search_from = search_from + pos + sep.len();
        }
        if piece_start < span.1 {
            self.split_units(text, map, (piece_start, span.1), sep_idx + 1, out);
        }
    }
}

impl Chunker for RecursiveChunker {
    fn chunk(&self, text: &str) -> Result<Vec<Chunk>, KonanError> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }
        let map = OffsetMap::new(text);
        let mut units = Vec::new();
        self.split_units(text, &map, (0, text.len()), 0, &mut units);
        let merged = merge_spans(&map, &units, self.chunk_size, self.chunk_overlap);
        Ok(spans_to_chunks(text, &map, &merged))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::assert_char_offsets;

    #[test]
    fn prefers_paragraph_breaks() {
        let text = "Para one is here.\n\nPara two is here.\n\nPara three is here.";
        let chunks = RecursiveChunker::new(25, 0, None).unwrap().chunk(text).unwrap();
        assert_eq!(chunks.len(), 3);
        assert!(chunks[0].text.starts_with("Para one"));
        assert!(chunks[1].text.starts_with("Para two"));
        assert_char_offsets(text, &chunks);
    }

    #[test]
    fn falls_back_to_words_then_chars() {
        let text = "abcdefghij klmnopqrst";
        let chunks = RecursiveChunker::new(5, 0, None).unwrap().chunk(text).unwrap();
        for c in &chunks {
            assert!(c.end - c.start <= 5);
        }
        assert_char_offsets(text, &chunks);
    }

    #[test]
    fn overlap_carries_units() {
        let text = "alpha beta gamma delta epsilon zeta eta theta iota kappa";
        let chunks = RecursiveChunker::new(20, 8, None).unwrap().chunk(text).unwrap();
        assert!(chunks.len() >= 2);
        assert!(chunks[1].start < chunks[0].end);
        assert_char_offsets(text, &chunks);
    }

    #[test]
    fn custom_separators() {
        let text = "a|b|c|d";
        let seps = Some(vec!["|".to_string(), "".to_string()]);
        let chunks = RecursiveChunker::new(3, 0, seps).unwrap().chunk(text).unwrap();
        assert!(chunks.len() >= 2);
        assert_char_offsets(text, &chunks);
    }

    #[test]
    fn unicode_safe_char_fallback() {
        let text = "😀😁😂🤣😃😄😅😆😉😊";
        let chunks = RecursiveChunker::new(3, 0, None).unwrap().chunk(text).unwrap();
        assert_eq!(chunks.len(), 4);
        assert_char_offsets(text, &chunks);
    }
}
