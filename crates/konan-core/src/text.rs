//! Shared span helpers. All spans are byte offsets into the source;
//! `OffsetMap` converts byte offsets to char offsets at the boundary.

use crate::chunk::Chunk;

pub(crate) struct OffsetMap {
    /// Byte offset of each char, plus a sentinel equal to text.len().
    char_starts: Vec<usize>,
}

impl OffsetMap {
    pub fn new(text: &str) -> Self {
        let mut char_starts: Vec<usize> = text.char_indices().map(|(b, _)| b).collect();
        char_starts.push(text.len());
        Self { char_starts }
    }

    /// Char index for a byte offset lying on a char boundary.
    pub fn char_idx(&self, byte: usize) -> usize {
        self.char_starts.binary_search(&byte).unwrap_or_else(|i| i - 1)
    }

    /// Char length of the byte span [start, end).
    pub fn char_len(&self, start: usize, end: usize) -> usize {
        self.char_idx(end) - self.char_idx(start)
    }
}

/// Byte spans of whitespace-separated words.
pub(crate) fn word_spans(text: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut start: Option<usize> = None;
    for (i, c) in text.char_indices() {
        if c.is_whitespace() {
            if let Some(s) = start.take() {
                spans.push((s, i));
            }
        } else if start.is_none() {
            start = Some(i);
        }
    }
    if let Some(s) = start {
        spans.push((s, text.len()));
    }
    spans
}

/// Byte spans of sentences via unicode segmentation, trimmed of whitespace.
pub(crate) fn sentence_spans(text: &str) -> Vec<(usize, usize)> {
    use unicode_segmentation::UnicodeSegmentation;
    text.split_sentence_bound_indices()
        .filter_map(|(start, s)| {
            let lead = s.len() - s.trim_start().len();
            let trimmed = s.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some((start + lead, start + lead + trimmed.len()))
            }
        })
        .collect()
}

/// Byte spans of sentences split on a punctuation regex like `[.!?]+\s+`.
/// Each sentence keeps its punctuation, drops the trailing whitespace.
pub(crate) fn regex_sentence_spans(text: &str, re: &regex::Regex) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut start = 0;
    for m in re.find_iter(text) {
        let punct_end = m.start() + m.as_str().trim_end().len();
        if punct_end > start {
            spans.push((start, punct_end));
        }
        start = m.end();
    }
    if start < text.len() {
        let tail = text[start..].trim_end();
        if !tail.is_empty() {
            spans.push((start, start + tail.len()));
        }
    }
    spans
}

/// Merge unit spans into chunk spans of at most `chunk_size` chars, carrying
/// at most `chunk_overlap` trailing chars of units into the next chunk.
/// A single unit larger than `chunk_size` becomes its own chunk.
pub(crate) fn merge_spans(
    map: &OffsetMap,
    spans: &[(usize, usize)],
    chunk_size: usize,
    chunk_overlap: usize,
) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::new();
    let mut window: Vec<(usize, usize)> = Vec::new();
    for &span in spans {
        if !window.is_empty() && map.char_len(window[0].0, span.1) > chunk_size {
            let merged = (window[0].0, window.last().unwrap().1);
            if out.last() != Some(&merged) {
                out.push(merged);
            }
            let mut keep: Vec<(usize, usize)> = Vec::new();
            let mut acc = 0;
            for &w in window.iter().rev() {
                let len = map.char_len(w.0, w.1);
                if acc + len > chunk_overlap {
                    break;
                }
                keep.insert(0, w);
                acc += len;
            }
            window = keep;
        }
        window.push(span);
    }
    if !window.is_empty() {
        let merged = (window[0].0, window.last().unwrap().1);
        if out.last() != Some(&merged) {
            out.push(merged);
        }
    }
    out
}

/// Convert chunk byte spans into `Chunk`s with char offsets.
pub(crate) fn spans_to_chunks(text: &str, map: &OffsetMap, spans: &[(usize, usize)]) -> Vec<Chunk> {
    spans
        .iter()
        .enumerate()
        .map(|(i, &(s, e))| Chunk::new(&text[s..e], map.char_idx(s), map.char_idx(e), i))
        .collect()
}

#[cfg(test)]
pub(crate) fn assert_char_offsets(text: &str, chunks: &[Chunk]) {
    for c in chunks {
        let expect: String = text.chars().skip(c.start).take(c.end - c.start).collect();
        assert_eq!(c.text, expect, "chunk {} offsets are wrong", c.index);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offset_map_handles_multibyte() {
        let map = OffsetMap::new("a😀b");
        assert_eq!(map.char_idx(0), 0);
        assert_eq!(map.char_idx(1), 1);
        assert_eq!(map.char_idx(5), 2);
        assert_eq!(map.char_idx(6), 3);
        assert_eq!(map.char_len(1, 5), 1);
    }

    #[test]
    fn word_spans_basic() {
        assert_eq!(word_spans("one  two\nthree "), vec![(0, 3), (5, 8), (9, 14)]);
        assert!(word_spans("   ").is_empty());
    }

    #[test]
    fn sentence_spans_trims() {
        let spans = sentence_spans("Hello there. Second one! ");
        let text = "Hello there. Second one! ";
        let sents: Vec<&str> = spans.iter().map(|&(s, e)| &text[s..e]).collect();
        assert_eq!(sents, vec!["Hello there.", "Second one!"]);
    }

    #[test]
    fn merge_spans_with_overlap() {
        let text = "aaa bbb ccc ddd";
        let map = OffsetMap::new(text);
        let spans = word_spans(text);
        let merged = merge_spans(&map, &spans, 7, 3);
        assert_eq!(merged, vec![(0, 7), (4, 11), (8, 15)]);
    }

    #[test]
    fn merge_spans_oversized_unit_alone() {
        let text = "tiny enormousunit tiny";
        let map = OffsetMap::new(text);
        let merged = merge_spans(&map, &word_spans(text), 6, 0);
        assert_eq!(merged, vec![(0, 4), (5, 17), (18, 22)]);
    }
}
