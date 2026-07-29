use std::sync::Arc;

use crate::chunk::Chunk;
use crate::chunker::Chunker;
use crate::error::KonanError;
use crate::text::{paragraph_cuts, OffsetMap};
use bpe_openai::Tokenizer;

/// Token-exact chunker via bpe-openai (tiktoken-equivalent tokenization,
/// substantially faster encoder). `chunk_size`/`chunk_overlap` are in tokens.
/// Supported encodings: "cl100k_base", "o200k_base".
pub struct TokenChunker {
    tokenizer: &'static Tokenizer,
    chunk_size: usize,
    chunk_overlap: usize,
}

impl TokenChunker {
    pub fn new(
        chunk_size: usize,
        chunk_overlap: usize,
        encoding: &str,
    ) -> Result<Self, KonanError> {
        if chunk_size == 0 {
            return Err(KonanError::InvalidConfig("chunk_size must be > 0".into()));
        }
        if chunk_overlap >= chunk_size {
            return Err(KonanError::InvalidConfig(
                "chunk_overlap must be smaller than chunk_size".into(),
            ));
        }
        // Both supported encodings skip NFC normalization, so token byte
        // lengths map directly onto the input text — required for offsets.
        let tokenizer = match encoding {
            "cl100k_base" => bpe_openai::cl100k_base(),
            "o200k_base" => bpe_openai::o200k_base(),
            other => return Err(KonanError::Tokenizer(format!("unknown encoding: {other}"))),
        };
        Ok(Self {
            tokenizer,
            chunk_size,
            chunk_overlap,
        })
    }

    /// Byte offset where each token starts, plus a sentinel holding the total
    /// byte length. `token_len` is an O(1) dictionary lookup of the token's raw
    /// byte length, so the offsets tile the source exactly.
    ///
    /// `base` is added to every offset, letting a caller compute offsets for a
    /// slice of a larger text.
    fn token_offsets_seq(&self, text: &str, base: usize) -> Vec<usize> {
        let tokens = self.tokenizer.encode(text);
        let mut offsets = Vec::with_capacity(tokens.len() + 1);
        let mut pos = base;
        for &tok in &tokens {
            offsets.push(pos);
            pos += self.tokenizer.bpe.token_len(tok);
        }
        offsets.push(pos);
        offsets
    }

    /// [`Self::token_offsets_seq`] over the whole text, spreading the BPE pass
    /// across cores for inputs big enough to pay for the split.
    ///
    /// BPE encoding dominates this strategy by roughly 30x over every other
    /// chunker, and it parallelises exactly because the pretokenizer never emits
    /// a token spanning a paragraph break — see [`paragraph_cuts`].
    fn token_offsets(&self, text: &str) -> Vec<usize> {
        /// Below this the join and the cut scan cost more than they save.
        const PARALLEL_MIN_BYTES: usize = 256 * 1024;

        if text.len() < PARALLEL_MIN_BYTES {
            return self.token_offsets_seq(text, 0);
        }
        // Over-subscribing keeps threads fed when paragraph sizes are uneven.
        let cuts = paragraph_cuts(text, rayon::current_num_threads() * 8);
        if cuts.len() <= 2 {
            // No usable paragraph break (e.g. one enormous line).
            return self.token_offsets_seq(text, 0);
        }
        use rayon::prelude::*;
        let windows: Vec<(usize, usize)> = cuts.windows(2).map(|w| (w[0], w[1])).collect();
        let last = windows.len() - 1;
        let mut pieces: Vec<Vec<usize>> = windows
            .into_par_iter()
            .map(|(s, e)| self.token_offsets_seq(&text[s..e], s))
            .collect();
        // Every piece ends with a sentinel equal to the next piece's first
        // offset, so all but the final sentinel are redundant.
        let total: usize = pieces.iter().map(Vec::len).sum();
        let mut out = Vec::with_capacity(total);
        for (k, piece) in pieces.iter_mut().enumerate() {
            if k != last {
                piece.pop();
            }
            out.append(piece);
        }
        out
    }
}

impl Chunker for TokenChunker {
    fn chunk_shared(&self, src: &Arc<str>) -> Result<Vec<Chunk>, KonanError> {
        let text: &str = src;
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let offsets = self.token_offsets(text);
        if offsets.len() < 2 {
            return Ok(Vec::new());
        }
        let tokens_len = offsets.len() - 1;

        let map = OffsetMap::new(text);
        // Chunk starts and ends each ascend (snapping moves boundaries by at
        // most a few bytes), so cursors track char offsets in one pass.
        let mut start_cur = map.cursor();
        let mut end_cur = map.cursor();
        let step = self.chunk_size - self.chunk_overlap;
        let mut chunks = Vec::new();
        let mut i = 0;
        while i < tokens_len {
            let j = (i + self.chunk_size).min(tokens_len);
            let mut start_b = offsets[i];
            let mut end_b = offsets[j];
            // Tokens can split multi-byte chars: snap to char boundaries.
            while start_b > 0 && !text.is_char_boundary(start_b) {
                start_b -= 1;
            }
            while end_b < text.len() && !text.is_char_boundary(end_b) {
                end_b += 1;
            }
            let index = chunks.len();
            chunks.push(Chunk::shared(
                src,
                start_b,
                end_b,
                map.char_at(&mut start_cur, start_b),
                map.char_at(&mut end_cur, end_b),
                index,
            ));
            if j == tokens_len {
                break;
            }
            i += step;
        }
        Ok(chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::assert_char_offsets;

    /// Multi-paragraph text over the parallel threshold, so `token_offsets`
    /// actually takes the rayon path.
    fn big_paragraphed_text() -> String {
        let mut s = String::with_capacity(400_000);
        let mut n = 0usize;
        while s.len() < 350_000 {
            n += 1;
            s.push_str("The quick brown fox jumps over the lazy dog while seventy wizards ");
            s.push_str("briskly mix jugs of liquid oxygen under pale moonlight. ");
            if n.is_multiple_of(5) {
                s.push_str("\n\n");
            }
            if n.is_multiple_of(37) {
                // Multibyte and punctuation runs near cut points.
                s.push_str("Zw\u{f6}lf Boxk\u{e4}mpfer... \u{1f980}!?\n\n");
            }
        }
        s
    }

    /// The whole point of the parallel BPE pass: it must be indistinguishable
    /// from the sequential one.
    #[test]
    fn parallel_offsets_match_sequential() {
        let text = big_paragraphed_text();
        for enc in ["cl100k_base", "o200k_base"] {
            let chunker = TokenChunker::new(512, 64, enc).unwrap();
            let cuts = paragraph_cuts(&text, rayon::current_num_threads() * 8);
            assert!(
                cuts.len() > 2,
                "{enc}: corpus must exercise the parallel path"
            );
            assert_eq!(
                chunker.token_offsets(&text),
                chunker.token_offsets_seq(&text, 0),
                "{enc}: parallel offsets diverge from sequential"
            );
            // Stronger claim: the token stream itself is identical, not just the
            // byte boundaries it implies.
            let whole = chunker.tokenizer.encode(&text);
            let joined: Vec<u32> = cuts
                .windows(2)
                .flat_map(|w| chunker.tokenizer.encode(&text[w[0]..w[1]]))
                .collect();
            assert_eq!(whole, joined, "{enc}: token stream diverges across cuts");
        }
    }

    /// Chunk output, not just offsets, must be identical either side of the
    /// parallel threshold.
    #[test]
    fn parallel_chunks_match_sequential_chunks() {
        let text = big_paragraphed_text();
        let chunker = TokenChunker::new(512, 64, "cl100k_base").unwrap();
        let parallel = chunker.chunk(&text).unwrap();
        // Reproduce the sequential path by chunking with sequential offsets.
        let seq_offsets = chunker.token_offsets_seq(&text, 0);
        assert_eq!(chunker.token_offsets(&text), seq_offsets);
        assert!(parallel.len() > 1);
        assert_char_offsets(&text, &parallel);
    }

    /// A text with no newline at all has no safe cut, and must fall back rather
    /// than split somewhere unsafe.
    #[test]
    fn single_line_text_falls_back_to_sequential() {
        let text = "word ".repeat(80_000); // > threshold, zero newlines
        assert!(text.len() > 256 * 1024);
        let cuts = paragraph_cuts(&text, 16);
        assert_eq!(cuts, vec![0, text.len()], "no newline means no cut");
        let chunker = TokenChunker::new(512, 64, "cl100k_base").unwrap();
        assert_eq!(
            chunker.token_offsets(&text),
            chunker.token_offsets_seq(&text, 0)
        );
    }

    #[test]
    fn token_counts_respected() {
        let chunker = TokenChunker::new(8, 0, "cl100k_base").unwrap();
        let text = "the quick brown fox jumps over the lazy dog ".repeat(5);
        let chunks = chunker.chunk(&text).unwrap();
        assert!(chunks.len() > 1);
        for c in &chunks {
            let n = chunker.tokenizer.count(c.text());
            assert!(n <= 9, "chunk has {n} tokens"); // +1 slack for boundary snapping
        }
        assert_char_offsets(&text, &chunks);
    }

    #[test]
    fn overlap_in_tokens() {
        let chunker = TokenChunker::new(8, 4, "cl100k_base").unwrap();
        let text = "one two three four five six seven eight nine ten eleven twelve";
        let chunks = chunker.chunk(text).unwrap();
        assert!(chunks.len() >= 2);
        assert!(chunks[1].start < chunks[0].end);
        assert_char_offsets(text, &chunks);
    }

    #[test]
    fn unknown_encoding_rejected() {
        assert!(matches!(
            TokenChunker::new(8, 0, "nope"),
            Err(KonanError::Tokenizer(_))
        ));
    }

    #[test]
    fn unicode_boundary_snapping() {
        let chunker = TokenChunker::new(2, 0, "cl100k_base").unwrap();
        let text = "😀😁😂🤣😃 schön müde";
        let chunks = chunker.chunk(text).unwrap();
        assert_char_offsets(text, &chunks);
    }

    #[test]
    fn token_offsets_cover_source_exactly() {
        // The per-token byte lengths must tile the source text: with zero
        // overlap, chunks must be contiguous and cover the whole input.
        let chunker = TokenChunker::new(16, 0, "o200k_base").unwrap();
        let text = "Zwölf Boxkämpfer jagen Viktor quer über den großen Sylter Deich. 😀 done.";
        let chunks = chunker.chunk(text).unwrap();
        assert_eq!(chunks[0].start, 0);
        assert_eq!(chunks.last().unwrap().end, text.chars().count());
        for pair in chunks.windows(2) {
            assert_eq!(pair[0].end, pair[1].start);
        }
        assert_char_offsets(text, &chunks);
    }
}
