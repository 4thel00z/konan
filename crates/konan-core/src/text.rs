//! Shared span helpers. All spans are byte offsets into the source;
//! `OffsetMap` converts byte offsets to char offsets at the boundary.

use std::sync::Arc;

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

/// Lanes of a `u64` viewed as eight bytes: low bit of each lane, and the high
/// bit of each lane.
const SWAR_LO: u64 = 0x0101_0101_0101_0101;
const SWAR_HI: u64 = 0x8080_8080_8080_8080;

/// Marks the high bit of every lane whose byte is `>= k`, for `k <= 0x80`.
///
/// Setting each lane's high bit before subtracting keeps the borrow inside its
/// own lane, so `0x80 + v - k` leaves the high bit set exactly when `v >= k`.
/// Lanes whose byte is itself `>= 0x80` answer "yes" to every `k`, so callers
/// that care must mask them out — every use below does, via `!x & SWAR_HI`.
#[inline(always)]
fn swar_ge(x: u64, k: u64) -> u64 {
    ((x | SWAR_HI).wrapping_sub(SWAR_LO * k)) & SWAR_HI
}

/// Counts UTF-8 char starts (non-continuation bytes) eight at a time.
///
/// A continuation byte is exactly `0b10xx_xxxx` — high bit set, next bit clear —
/// so `x & !(x << 1)` leaves a lane's high bit set precisely for continuations
/// and `count_ones` totals them. Only within-lane bits are tested, so the
/// native-endian load is correct on both endiannesses. This runs ~7x the scalar
/// filter loop it replaced, which only reached ~4 GB/s even auto-vectorised.
fn count_chars(s: &str) -> usize {
    let bytes = s.as_bytes();
    let mut chunks = bytes.chunks_exact(8);
    let mut cont = 0u32;
    for c in chunks.by_ref() {
        let x = u64::from_ne_bytes(c.try_into().expect("chunks_exact(8) yields 8 bytes"));
        cont += (x & !(x << 1) & SWAR_HI).count_ones();
    }
    // 0..7 trailing bytes: 0x80..=0xBF is the continuation range.
    cont += chunks
        .remainder()
        .iter()
        .filter(|&&b| (b as i8) < -0x40)
        .count() as u32;
    bytes.len() - cont as usize
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

/// The ASCII subset of `char::is_whitespace`: space plus `\t \n \v \f \r`.
/// Note `u8::is_ascii_whitespace` omits VT (0x0B), so it cannot be used here —
/// the byte path has to agree with the char path exactly.
#[inline(always)]
fn is_ascii_ws(b: u8) -> bool {
    b == 0x20 || (0x09..=0x0D).contains(&b)
}

/// Sets the high bit of every lane holding an ASCII whitespace byte.
/// Lanes with the high bit already set (UTF-8 leads/continuations) are cleared,
/// so this only ever fires on real ASCII whitespace.
#[inline(always)]
fn ws_mask(x: u64) -> u64 {
    let space = swar_ge(x, 0x20) & !swar_ge(x, 0x21);
    let ctl = swar_ge(x, 0x09) & !swar_ge(x, 0x0E);
    (space | ctl) & !x & SWAR_HI
}

/// Loads 8 bytes little-endian so lane `j` always occupies bits `8j..8j+8`
/// regardless of host endianness, which keeps the bit↔byte index arithmetic
/// below portable.
#[inline(always)]
fn load8(bytes: &[u8], i: usize) -> u64 {
    u64::from_le_bytes(bytes[i..i + 8].try_into().expect("8-byte window"))
}

/// First index `>= from` whose byte is not ASCII whitespace, else `bytes.len()`.
fn skip_ascii_ws(bytes: &[u8], mut i: usize) -> usize {
    while i + 8 <= bytes.len() {
        let non_ws = !ws_mask(load8(bytes, i)) & SWAR_HI;
        if non_ws != 0 {
            return i + (non_ws.trailing_zeros() as usize >> 3);
        }
        i += 8;
    }
    while i < bytes.len() && is_ascii_ws(bytes[i]) {
        i += 1;
    }
    i
}

/// First index `>= from` whose byte is ASCII whitespace, else `bytes.len()`.
fn find_ascii_ws(bytes: &[u8], mut i: usize) -> usize {
    while i + 8 <= bytes.len() {
        let m = ws_mask(load8(bytes, i));
        if m != 0 {
            return i + (m.trailing_zeros() as usize >> 3);
        }
        i += 8;
    }
    while i < bytes.len() && !is_ascii_ws(bytes[i]) {
        i += 1;
    }
    i
}

/// Byte spans of whitespace-separated words.
///
/// ASCII text takes a structural-indexing path: a whitespace bitmask is built
/// eight bytes at a time and word boundaries fall out of the mask's transition
/// bits, so the cost is roughly one branch per *word* instead of per *byte*.
/// Non-ASCII text keeps the char-decoding scan, because `char::is_whitespace`
/// also matches NBSP and ideographic space.
pub(crate) fn word_spans(text: &str) -> Vec<(usize, usize)> {
    if !text.is_ascii() {
        return word_spans_unicode(text);
    }
    let bytes = text.as_bytes();
    let n = bytes.len();
    // Prose averages ~6 bytes per word; over-reserving slightly beats regrowing.
    let mut spans: Vec<(usize, usize)> = Vec::with_capacity(n / 6 + 8);
    let mut prev_ws = true;
    let mut open = 0usize;
    let mut i = 0usize;
    while i + 8 <= n {
        let mask = ws_mask(load8(bytes, i));
        // Lane j's predecessor is lane j-1; lane 0's comes from the last block.
        let shifted = (mask << 8) | ((prev_ws as u64) << 7);
        let mut trans = mask ^ shifted;
        while trans != 0 {
            let bit = trans.trailing_zeros() as usize;
            let pos = i + (bit >> 3);
            if (mask >> bit) & 1 == 1 {
                spans.push((open, pos));
            } else {
                open = pos;
            }
            trans &= trans - 1;
        }
        prev_ws = (mask >> 63) & 1 == 1;
        i += 8;
    }
    // 0..7 trailing bytes, continuing the same state machine.
    while i < n {
        let w = is_ascii_ws(bytes[i]);
        if w != prev_ws {
            if w {
                spans.push((open, i));
            } else {
                open = i;
            }
            prev_ws = w;
        }
        i += 1;
    }
    if !prev_ws {
        spans.push((open, n));
    }
    spans
}

/// `word_spans` for text containing non-ASCII bytes.
fn word_spans_unicode(text: &str) -> Vec<(usize, usize)> {
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

/// Byte offsets a text can be split at without changing how the pieces
/// tokenize or segment: the first byte after a whitespace run containing a
/// newline.
///
/// Two independent guarantees rest on this shape. UAX #29 treats a paragraph
/// separator as a mandatory sentence boundary, so ICU segments the pieces the
/// same way it would segment the whole. And the cl100k/o200k pretokenizers never
/// emit a token spanning such a boundary, so BPE-encoding the pieces and
/// concatenating reproduces the single-threaded token stream. Both are pinned by
/// tests rather than assumed.
///
/// Always starts at 0 and ends at `text.len()`; `want` is a hint for how many
/// pieces to aim for. Every cut lands on a char boundary because it follows an
/// ASCII whitespace byte.
pub(crate) fn paragraph_cuts(text: &str, want: usize) -> Vec<usize> {
    let bytes = text.as_bytes();
    let mut cuts = vec![0usize];
    if want > 1 && !bytes.is_empty() {
        let stride = (text.len() / want).max(1);
        let mut target = stride;
        let mut i = 0usize;
        while i < bytes.len() {
            if bytes[i] == b'\n' {
                let mut j = i + 1;
                while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                    j += 1;
                }
                if j >= target && j < bytes.len() && j > *cuts.last().expect("seeded with 0") {
                    cuts.push(j);
                    target = j + stride;
                }
                i = j.max(i + 1);
            } else {
                i += 1;
            }
        }
    }
    cuts.push(text.len());
    cuts
}

/// Byte spans of sentences via UAX #29 segmentation (ICU4X), trimmed of
/// whitespace, with every offset shifted by `base`.
///
/// The compiled-data segmenter is a zero-cost handle, so it is built per call
/// (measured: unmeasurably cheap).
fn sentence_spans_at(text: &str, base: usize) -> Vec<(usize, usize)> {
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
            spans.push((base + start + lead, base + start + lead + trimmed.len()));
        }
        start = end;
    }
    spans
}

/// Byte spans of sentences via UAX #29 segmentation (ICU4X), trimmed of
/// whitespace.
///
/// Segmentation is essentially the entire cost of the sentence and semantic
/// strategies, so large inputs are split at paragraph boundaries and segmented
/// across cores. [`paragraph_cuts`] explains why that is boundary-preserving;
/// `sentence_spans_parallel_matches_sequential` proves it.
pub(crate) fn sentence_spans(text: &str) -> Vec<(usize, usize)> {
    /// Below this the fan-out costs more than the segmentation saves.
    const PARALLEL_MIN_BYTES: usize = 128 * 1024;

    if text.len() < PARALLEL_MIN_BYTES {
        return sentence_spans_at(text, 0);
    }
    let cuts = paragraph_cuts(text, rayon::current_num_threads() * 4);
    if cuts.len() <= 2 {
        return sentence_spans_at(text, 0);
    }
    use rayon::prelude::*;
    let windows: Vec<(usize, usize)> = cuts.windows(2).map(|w| (w[0], w[1])).collect();
    // Indexed collect keeps piece order, so the spans stay ascending.
    let pieces: Vec<Vec<(usize, usize)>> = windows
        .into_par_iter()
        .map(|(s, e)| sentence_spans_at(&text[s..e], s))
        .collect();
    let mut out = Vec::with_capacity(pieces.iter().map(Vec::len).sum());
    for piece in &pieces {
        out.extend_from_slice(piece);
    }
    out
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
            let bytes = self.text.as_bytes();
            let start = skip_ascii_ws(bytes, self.pos);
            if start >= bytes.len() {
                return None;
            }
            let end = find_ascii_ws(bytes, start + 1);
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
    // Units land in one append-only arena and the live window is the index range
    // `[lo, arena.len())`. Carrying overlap into the next chunk is then just an
    // assignment to `lo`, where the previous implementation allocated and
    // reversed a fresh `keep` Vec at every chunk boundary.
    let mut arena: Vec<(usize, usize)> = Vec::new();
    let mut lo = 0usize;
    // Char length of (arena[lo].0 .. arena.last().1), maintained incrementally
    // so unicode texts never recount the whole window.
    let mut window_chars = 0usize;
    for span in spans {
        // Once the dead prefix dominates, drop it so the arena stays proportional
        // to the live window rather than to the whole input. Amortised O(1).
        if lo >= 4096 && lo * 2 >= arena.len() {
            arena.drain(..lo);
            lo = 0;
        }
        let hi = arena.len();
        if lo < hi && window_chars + map.char_len(arena[hi - 1].1, span.1) > chunk_size {
            let merged = (arena[lo].0, arena[hi - 1].1);
            if out.last() != Some(&merged) {
                out.push(merged);
            }
            // Walk back from the window end to the furthest unit that still fits
            // in the overlap budget.
            let mut acc = 0usize;
            let mut new_lo = hi;
            for k in (lo..hi).rev() {
                let len = map.char_len(arena[k].0, arena[k].1);
                if acc + len > chunk_overlap {
                    break;
                }
                acc += len;
                new_lo = k;
            }
            lo = new_lo;
            window_chars = if lo < hi {
                map.char_len(arena[lo].0, arena[hi - 1].1)
            } else {
                0
            };
        }
        window_chars = if lo < arena.len() {
            window_chars + map.char_len(arena[arena.len() - 1].1, span.1)
        } else {
            map.char_len(span.0, span.1)
        };
        arena.push(span);
    }
    if lo < arena.len() {
        let merged = (arena[lo].0, arena[arena.len() - 1].1);
        if out.last() != Some(&merged) {
            out.push(merged);
        }
    }
    out
}

/// Convert chunk byte spans into `Chunk`s with char offsets. Span starts and
/// ends must each be ascending (overlap between chunks is fine).
///
/// The chunks borrow `src`, so no chunk text is copied.
pub(crate) fn spans_to_chunks(
    src: &Arc<str>,
    map: &OffsetMap,
    spans: &[(usize, usize)],
) -> Vec<Chunk> {
    let mut start_cur = map.cursor();
    let mut end_cur = map.cursor();
    spans
        .iter()
        .enumerate()
        .map(|(i, &(s, e))| {
            let cs = map.char_at(&mut start_cur, s);
            let ce = map.char_at(&mut end_cur, e);
            Chunk::shared(src, s, e, cs, ce, i)
        })
        .collect()
}

#[cfg(test)]
pub(crate) fn assert_char_offsets(text: &str, chunks: &[Chunk]) {
    for c in chunks {
        let expect: String = text.chars().skip(c.start).take(c.end - c.start).collect();
        assert_eq!(c.text(), expect, "chunk {} offsets are wrong", c.index);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Corpora that between them cover ASCII, 2/3/4-byte sequences, every
    /// alignment of the SWAR block loop, and lengths either side of a block.
    fn char_count_corpora() -> Vec<String> {
        let mut v = vec![
            String::new(),
            "a".to_string(),
            "ascii only".to_string(),
            "café".to_string(),
            "日本語の文書です".to_string(),
            "🦀🦀🦀".to_string(),
            "aé日🦀 mixed \u{a0}\u{3000}".to_string(),
        ];
        // Every length 0..40 of a repeating multibyte pattern, so the block loop
        // and its 0..7-byte remainder are both exercised at each offset.
        for n in 0..40 {
            v.push("aé日🦀".chars().cycle().take(n).collect());
            v.push("x".repeat(n));
        }
        v
    }

    #[test]
    fn count_chars_matches_std_char_count() {
        for s in char_count_corpora() {
            assert_eq!(
                count_chars(&s),
                s.chars().count(),
                "wrong char count for {s:?} ({} bytes)",
                s.len()
            );
        }
    }

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

    /// Every ASCII byte, at every offset within a SWAR block and in the 0..7
    /// byte remainder, must split exactly like the char-decoding scan. This is
    /// the test that catches lane/bit-index and tail mistakes.
    #[test]
    fn word_spans_swar_matches_char_scan_for_every_ascii_byte() {
        for b in 0u8..=0x7F {
            let c = b as char;
            for pad in 0..13 {
                let mut s = "ab ".repeat(pad);
                s.push(c);
                s.push_str("cd ef");
                assert_eq!(
                    word_spans(&s),
                    word_spans_unicode(&s),
                    "byte {b:#04x} at pad {pad} in {s:?}"
                );
                // Same byte as a trailing character, to vary the remainder length.
                let mut t = "xy ".repeat(pad);
                t.push_str("word");
                t.push(c);
                assert_eq!(
                    word_spans(&t),
                    word_spans_unicode(&t),
                    "trailing byte {b:#04x} at pad {pad} in {t:?}"
                );
            }
        }
    }

    /// Runs longer than one block, of both whitespace and non-whitespace, so the
    /// no-transition fast path and the cross-block `prev_ws` carry get covered.
    #[test]
    fn word_spans_swar_matches_char_scan_across_long_runs() {
        let cases = [
            " ".repeat(40),
            "\t\n\r\u{b}\u{c} ".repeat(9),
            "averyveryverylongwordwithnobreaksatallhere".to_string(),
            format!("{}{}", " ".repeat(17), "w".repeat(23)),
            format!("{}{}", "w".repeat(23), " ".repeat(17)),
            String::new(),
            "\n".to_string(),
            "a".to_string(),
        ];
        for s in cases {
            assert_eq!(word_spans(&s), word_spans_unicode(&s), "for {s:?}");
        }
    }

    /// Non-ASCII text must keep using `char::is_whitespace`, which treats NBSP
    /// and ideographic space as whitespace where the ASCII mask would not.
    #[test]
    fn word_spans_unicode_whitespace_still_splits() {
        let text = "one\u{a0}two\u{3000}three caf\u{e9}";
        let spans = word_spans(text);
        let words: Vec<&str> = spans.iter().map(|&(s, e)| &text[s..e]).collect();
        assert_eq!(words, vec!["one", "two", "three", "caf\u{e9}"]);
        assert_eq!(spans, word_spans_unicode(text));
    }

    /// The lazy iterator and the eager scan are used by different strategies and
    /// must agree, or `naive` and `fixed_size` would disagree on word fallback.
    #[test]
    fn word_units_agrees_with_word_spans() {
        let cases = [
            "one  two\nthree ".to_string(),
            "   ".to_string(),
            String::new(),
            "word ".repeat(30),
            "\t\n\r\u{b}\u{c} mixed \u{b}breaks\t".to_string(),
            "one\u{a0}two\u{3000}three caf\u{e9}".to_string(),
            "a".repeat(100),
        ];
        for s in cases {
            let lazy: Vec<(usize, usize)> = WordUnits::new(&s).collect();
            assert_eq!(lazy, word_spans(&s), "for {s:?}");
        }
    }

    /// Multi-paragraph prose comfortably over the parallel threshold.
    fn big_paragraphed_text() -> String {
        let mut s = String::with_capacity(400_000);
        let mut n = 0usize;
        while s.len() < 300_000 {
            n += 1;
            s.push_str("The quick brown fox jumps over the lazy dog. ");
            s.push_str("Seventy wizards briskly mix jugs of liquid oxygen! ");
            s.push_str("Was it under pale moonlight? Indeed. ");
            if n.is_multiple_of(4) {
                s.push_str("\n\n");
            }
            if n.is_multiple_of(31) {
                s.push_str("Zw\u{f6}lf Boxk\u{e4}mpfer... \u{1f980}!? \u{65e5}\u{672c}\u{8a9e}\u{306e}\u{6587}\u{7ae0}\u{3002}\n\n");
            }
        }
        s
    }

    /// Splitting segmentation across paragraph cuts must not move a single
    /// sentence boundary.
    #[test]
    fn sentence_spans_parallel_matches_sequential() {
        let text = big_paragraphed_text();
        assert!(text.len() >= 128 * 1024, "must exercise the parallel path");
        let cuts = paragraph_cuts(&text, rayon::current_num_threads() * 4);
        assert!(cuts.len() > 2, "corpus must produce usable cuts");
        assert_eq!(
            sentence_spans(&text),
            sentence_spans_at(&text, 0),
            "parallel segmentation diverges from sequential"
        );
    }

    /// A large text with no newline has no safe cut and must fall back.
    #[test]
    fn sentence_spans_without_newlines_falls_back() {
        let text = "This is a sentence. ".repeat(9_000);
        assert!(text.len() >= 128 * 1024);
        assert_eq!(paragraph_cuts(&text, 32), vec![0, text.len()]);
        assert_eq!(sentence_spans(&text), sentence_spans_at(&text, 0));
    }

    #[test]
    fn paragraph_cuts_are_char_boundaries_and_strictly_ascend() {
        let text = big_paragraphed_text();
        for want in [2usize, 8, 32, 256] {
            let cuts = paragraph_cuts(&text, want);
            assert_eq!(cuts.first(), Some(&0));
            assert_eq!(cuts.last(), Some(&text.len()));
            for w in cuts.windows(2) {
                assert!(w[0] < w[1], "cuts must strictly ascend for want={want}");
            }
            for &c in &cuts {
                assert!(text.is_char_boundary(c), "cut {c} splits a char");
            }
        }
    }

    #[test]
    fn paragraph_cuts_handles_degenerate_inputs() {
        assert_eq!(paragraph_cuts("", 8), vec![0, 0]);
        assert_eq!(paragraph_cuts("no newline", 8), vec![0, 10]);
        // want <= 1 asks for no split at all.
        assert_eq!(paragraph_cuts("a\n\nb\n\nc", 1), vec![0, 7]);
        // A trailing newline must not produce a cut at len.
        let t = "a\n\nb\n\n";
        let cuts = paragraph_cuts(t, 8);
        assert_eq!(cuts.last(), Some(&t.len()));
        assert_eq!(cuts.iter().filter(|&&c| c == t.len()).count(), 1);
    }

    #[test]
    fn sentence_spans_trims() {
        let spans = sentence_spans("Hello there. Second one! ");
        let text = "Hello there. Second one! ";
        let sents: Vec<&str> = spans.iter().map(|&(s, e)| &text[s..e]).collect();
        assert_eq!(sents, vec!["Hello there.", "Second one!"]);
    }

    /// The window-as-a-Vec implementation the arena version replaced. Kept as
    /// the oracle: the rewrite is a pure performance change and any divergence
    /// is a bug.
    fn merge_span_iter_reference(
        map: &OffsetMap,
        spans: impl Iterator<Item = (usize, usize)>,
        chunk_size: usize,
        chunk_overlap: usize,
    ) -> Vec<(usize, usize)> {
        let mut out: Vec<(usize, usize)> = Vec::new();
        let mut window: Vec<(usize, usize)> = Vec::new();
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

    #[test]
    fn merge_arena_matches_window_reference() {
        // Includes a corpus with well over 4096 units so arena compaction runs,
        // and unicode so the incremental char_len bookkeeping is exercised.
        let long_words = "alpha beta gamma delta epsilon ".repeat(2000);
        let unicode =
            "Caf\u{e9} na\u{ef}ve \u{1f980} crab. \u{65e5}\u{672c}\u{8a9e}. More \u{fc}nicode! "
                .repeat(200);
        let corpora = [
            "aaa bbb ccc ddd".to_string(),
            "tiny enormousunit tiny".to_string(),
            "word ".repeat(50),
            long_words,
            unicode,
            String::new(),
        ];
        for text in &corpora {
            let map = OffsetMap::new(text);
            let words = word_spans(text);
            for &size in &[1usize, 7, 30, 100, 1000] {
                for &overlap in &[0usize, 1, 10, 200] {
                    if overlap >= size {
                        continue;
                    }
                    let got = merge_span_iter(&map, words.iter().copied(), size, overlap);
                    let want =
                        merge_span_iter_reference(&map, words.iter().copied(), size, overlap);
                    assert_eq!(
                        got,
                        want,
                        "divergence at size={size} overlap={overlap} for {} units",
                        words.len()
                    );
                }
            }
        }
    }

    /// Sentence units are longer and less uniform than words, which exercises
    /// different overlap walk-back paths than the word corpus above.
    #[test]
    fn merge_arena_matches_reference_on_sentence_units() {
        let text = "First one here. Second follows! Third now? A much longer fourth sentence that runs well past any small limit. Fifth. ".repeat(120);
        let map = OffsetMap::new(&text);
        for &size in &[20usize, 50, 200, 1000] {
            for &overlap in &[0usize, 5, 19, 150] {
                if overlap >= size {
                    continue;
                }
                let got = merge_span_iter(&map, SentenceUnits::new(&text), size, overlap);
                let want =
                    merge_span_iter_reference(&map, SentenceUnits::new(&text), size, overlap);
                assert_eq!(got, want, "divergence at size={size} overlap={overlap}");
            }
        }
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
