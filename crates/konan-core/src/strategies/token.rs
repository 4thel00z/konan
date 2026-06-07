use crate::chunk::Chunk;
use crate::chunker::Chunker;
use crate::error::KonanError;
use crate::text::OffsetMap;
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
}

impl Chunker for TokenChunker {
    fn chunk(&self, text: &str) -> Result<Vec<Chunk>, KonanError> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let tokens = self.tokenizer.encode(text);
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        // Byte offset where each token starts, plus a sentinel. `token_len`
        // is an O(1) dictionary lookup of the token's raw byte length.
        let mut offsets = Vec::with_capacity(tokens.len() + 1);
        let mut pos = 0usize;
        for &tok in &tokens {
            offsets.push(pos);
            pos += self.tokenizer.bpe.token_len(tok);
        }
        offsets.push(pos);

        let map = OffsetMap::new(text);
        let step = self.chunk_size - self.chunk_overlap;
        let mut chunks = Vec::new();
        let mut i = 0;
        while i < tokens.len() {
            let j = (i + self.chunk_size).min(tokens.len());
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
            chunks.push(Chunk::new(
                &text[start_b..end_b],
                map.char_idx(start_b),
                map.char_idx(end_b),
                index,
            ));
            if j == tokens.len() {
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

    #[test]
    fn token_counts_respected() {
        let chunker = TokenChunker::new(8, 0, "cl100k_base").unwrap();
        let text = "the quick brown fox jumps over the lazy dog ".repeat(5);
        let chunks = chunker.chunk(&text).unwrap();
        assert!(chunks.len() > 1);
        for c in &chunks {
            let n = chunker.tokenizer.count(c.text.as_str());
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
