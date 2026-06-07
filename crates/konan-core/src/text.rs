//! Shared span helpers. All spans are byte offsets into the source;
//! `OffsetMap` converts byte offsets to char offsets at the boundary.

use crate::chunk::Chunk;

/// Byte→char conversion for one source text. For pure-ASCII text every
/// operation is O(1) byte arithmetic. For non-ASCII text spans are counted
/// directly (the per-char count below vectorizes well), which beats building
/// a global char table: chunkers only ever measure local spans and walk
/// boundaries in order.
pub(crate) struct OffsetMap<'a> {
    text: &'a str,
    ascii: bool,
}

/// Walks char offsets along (mostly) ascending byte positions without
/// recounting from the start of the text. Obtain via [`OffsetMap::cursor`].
pub(crate) struct CharCursor {
    byte: usize,
    ch: usize,
}

fn count_chars(s: &str) -> usize {
    // Counts non-continuation bytes; LLVM vectorizes this loop.
    s.as_bytes().iter().filter(|&&b| (b as i8) >= -0x40).count()
}

impl<'a> OffsetMap<'a> {
    pub fn new(text: &'a str) -> Self {
        Self {
            text,
            ascii: text.is_ascii(),
        }
    }

    /// Char length of the byte span [start, end) (char-aligned boundaries).
    pub fn char_len(&self, start: usize, end: usize) -> usize {
        if self.ascii {
            end - start
        } else {
            count_chars(&self.text[start..end])
        }
    }

    pub fn cursor(&self) -> CharCursor {
        CharCursor { byte: 0, ch: 0 }
    }

    /// Char index of `byte` (a char boundary), advancing the cursor. Small
    /// backward jumps (e.g. token-boundary snapping) are tolerated and cost
    /// only the distance moved.
    pub fn char_at(&self, cur: &mut CharCursor, byte: usize) -> usize {
        if self.ascii {
            return byte;
        }
        if byte >= cur.byte {
            cur.ch += count_chars(&self.text[cur.byte..byte]);
        } else {
            cur.ch -= count_chars(&self.text[byte..cur.byte]);
        }
        cur.byte = byte;
        cur.ch
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

/// Byte spans of sentences via UAX #29 segmentation (ICU4X), trimmed of
/// whitespace. ICU4X segments ~4x faster than the previous
/// unicode-segmentation iterator with identical boundaries; the
/// compiled-data segmenter is a zero-cost handle, so it is built per call.
pub(crate) fn sentence_spans(text: &str) -> Vec<(usize, usize)> {
    let segmenter = icu_segmenter::SentenceSegmenter::new(Default::default());
    let mut bounds = segmenter.segment_str(text);
    let mut spans = Vec::new();
    let Some(mut start) = bounds.next() else {
        return spans;
    };
    for end in bounds {
        let s = &text[start..end];
        let lead = s.len() - s.trim_start().len();
        let trimmed = s.trim();
        if !trimmed.is_empty() {
            spans.push((start + lead, start + lead + trimmed.len()));
        }
        start = end;
    }
    spans
}

/// Byte spans of sentences split on a punctuation regex like `[.!?]+\s+`.
/// Each sentence keeps its punctuation, drops the trailing whitespace.
/// Test-only since [`SentenceUnits`] took over the hot path: it remains the
/// oracle for the fixed-size equivalence matrix.
#[cfg(test)]
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

/// Lazily yields the same spans as [`regex_sentence_spans`] with the pattern
/// `[.!?]+\s+` — each sentence keeps its punctuation run, drops the trailing
/// whitespace, the final span is the trimmed tail — but finds candidate
/// boundaries with `memchr3` instead of running the regex engine, which cuts
/// the scan cost on sentence-dense prose by an order of magnitude.
pub(crate) struct SentenceUnits<'a> {
    text: &'a str,
    /// Start byte of the next unit (the first unit anchors at byte 0,
    /// matching the legacy regex split).
    start: usize,
    done: bool,
}

impl<'a> SentenceUnits<'a> {
    pub fn new(text: &'a str) -> Self {
        Self {
            text,
            start: 0,
            done: false,
        }
    }
}

impl Iterator for SentenceUnits<'_> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<(usize, usize)> {
        if self.done {
            return None;
        }
        let bytes = self.text.as_bytes();
        let mut probe = self.start;
        while let Some(found) = memchr::memchr3(b'.', b'!', b'?', &bytes[probe..]) {
            let punct_start = probe + found;
            let mut punct_end = punct_start + 1;
            while punct_end < bytes.len() && matches!(bytes[punct_end], b'.' | b'!' | b'?') {
                punct_end += 1;
            }
            let ws_len = {
                let rest = &self.text[punct_end..];
                rest.len() - rest.trim_start().len()
            };
            if ws_len > 0 {
                let span = (self.start, punct_end);
                self.start = punct_end + ws_len;
                if span.1 > span.0 {
                    return Some(span);
                }
                probe = self.start;
                continue;
            }
            probe = punct_end;
        }
        self.done = true;
        let tail = self.text[self.start..].trim_end();
        if tail.is_empty() {
            None
        } else {
            Some((self.start, self.start + tail.len()))
        }
    }
}

/// Lazily yields the same spans as [`word_spans`]: whitespace-separated
/// words, without materialising the full span vector. ASCII text takes a
/// byte-loop fast path; otherwise boundaries are char-decoded exactly like
/// the legacy scan.
pub(crate) struct WordUnits<'a> {
    text: &'a str,
    pos: usize,
    ascii: bool,
}

impl<'a> WordUnits<'a> {
    pub fn new(text: &'a str) -> Self {
        Self {
            text,
            pos: 0,
            ascii: text.is_ascii(),
        }
    }
}

impl Iterator for WordUnits<'_> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<(usize, usize)> {
        if self.ascii {
            // is_ascii_whitespace misses VT (0x0B), which char::is_whitespace
            // includes — the byte path must match the char path exactly.
            let ws = |b: u8| b.is_ascii_whitespace() || b == 0x0B;
            let bytes = self.text.as_bytes();
            let mut start = self.pos;
            while start < bytes.len() && ws(bytes[start]) {
                start += 1;
            }
            if start >= bytes.len() {
                return None;
            }
            let mut end = start + 1;
            while end < bytes.len() && !ws(bytes[end]) {
                end += 1;
            }
            self.pos = end;
            return Some((start, end));
        }
        let rest = &self.text[self.pos..];
        let lead = rest.len() - rest.trim_start().len();
        let start = self.pos + lead;
        if start >= self.text.len() {
            return None;
        }
        let word = &self.text[start..];
        let end = start
            + word
                .char_indices()
                .find(|&(_, c)| c.is_whitespace())
                .map_or(word.len(), |(i, _)| i);
        self.pos = end;
        Some((start, end))
    }
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
    merge_span_iter(map, spans.iter().copied(), chunk_size, chunk_overlap)
}

/// [`merge_spans`] over a lazy unit iterator: identical semantics, no
/// materialised unit vector.
pub(crate) fn merge_span_iter(
    map: &OffsetMap,
    spans: impl Iterator<Item = (usize, usize)>,
    chunk_size: usize,
    chunk_overlap: usize,
) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::new();
    let mut window: Vec<(usize, usize)> = Vec::new();
    // Char length of (window[0].0 .. window.last().1), maintained
    // incrementally so unicode texts never recount the whole window.
    let mut window_chars = 0usize;
    for span in spans {
        if !window.is_empty()
            && window_chars + map.char_len(window.last().unwrap().1, span.1) > chunk_size
        {
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
                keep.push(w);
                acc += len;
            }
            keep.reverse();
            window = keep;
            window_chars = match (window.first(), window.last()) {
                (Some(&(s, _)), Some(&(_, e))) => map.char_len(s, e),
                _ => 0,
            };
        }
        window_chars = match window.last() {
            Some(&(_, last_end)) => window_chars + map.char_len(last_end, span.1),
            None => map.char_len(span.0, span.1),
        };
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

/// Convert chunk byte spans into `Chunk`s with char offsets. Span starts and
/// ends must each be ascending (overlap between chunks is fine).
pub(crate) fn spans_to_chunks(text: &str, map: &OffsetMap, spans: &[(usize, usize)]) -> Vec<Chunk> {
    let mut start_cur = map.cursor();
    let mut end_cur = map.cursor();
    spans
        .iter()
        .enumerate()
        .map(|(i, &(s, e))| {
            let cs = map.char_at(&mut start_cur, s);
            let ce = map.char_at(&mut end_cur, e);
            Chunk::new(&text[s..e], cs, ce, i)
        })
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
        let mut cur = map.cursor();
        assert_eq!(map.char_at(&mut cur, 0), 0);
        assert_eq!(map.char_at(&mut cur, 1), 1);
        assert_eq!(map.char_at(&mut cur, 5), 2);
        assert_eq!(map.char_at(&mut cur, 6), 3);
        // Backward jump (token-boundary snapping pattern).
        assert_eq!(map.char_at(&mut cur, 1), 1);
        assert_eq!(map.char_len(1, 5), 1);
    }

    #[test]
    fn offset_map_ascii_fast_path() {
        let map = OffsetMap::new("plain ascii text");
        assert!(map.ascii, "ASCII text must take the byte==char path");
        let mut cur = map.cursor();
        assert_eq!(map.char_at(&mut cur, 7), 7);
        assert_eq!(map.char_at(&mut cur, 16), 16);
        assert_eq!(map.char_len(6, 11), 5);
    }

    #[test]
    fn word_spans_basic() {
        assert_eq!(
            word_spans("one  two\nthree "),
            vec![(0, 3), (5, 8), (9, 14)]
        );
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
