//! Units that chunk budgets are measured in: chars (the default), tokens of
//! an OpenAI encoding, or tokens of any [`TokenEnds`] implementation.

use std::fmt;
use std::sync::Arc;

use bpe_openai::Tokenizer;

use crate::error::KonanError;
use crate::text::{paragraph_cuts, OffsetMap};

/// A tokenizer that chunk budgets can count in.
pub trait TokenEnds: Send + Sync {
    /// Byte offset just past each token of `text`, in ascending order. Offsets
    /// must refer to `text` itself, not to a normalized copy of it.
    fn token_ends(&self, text: &str) -> Result<Vec<usize>, KonanError>;
}

/// An OpenAI BPE encoding supported for token budgets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Encoding {
    Cl100kBase,
    O200kBase,
}

impl Encoding {
    /// Parses `"cl100k_base"` or `"o200k_base"`.
    pub fn from_name(name: &str) -> Result<Self, KonanError> {
        match name {
            "cl100k_base" => Ok(Self::Cl100kBase),
            "o200k_base" => Ok(Self::O200kBase),
            other => Err(KonanError::Tokenizer(format!("unknown encoding: {other}"))),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Cl100kBase => "cl100k_base",
            Self::O200kBase => "o200k_base",
        }
    }

    /// Both encodings skip NFC normalization, so token byte lengths map
    /// directly onto the input text, which offsets rely on.
    pub(crate) fn tokenizer(self) -> &'static Tokenizer {
        match self {
            Self::Cl100kBase => bpe_openai::cl100k_base(),
            Self::O200kBase => bpe_openai::o200k_base(),
        }
    }
}

impl TokenEnds for Encoding {
    fn token_ends(&self, text: &str) -> Result<Vec<usize>, KonanError> {
        let mut ends = token_offsets(self.tokenizer(), text);
        ends.remove(0);
        Ok(ends)
    }
}

/// What `chunk_size` and `chunk_overlap` count. Token units are counted over
/// one tokenization of the whole document.
#[derive(Clone, Default)]
pub enum SizeUnit {
    #[default]
    Chars,
    Tokens(Encoding),
    Tokenizer(Arc<dyn TokenEnds>),
}

impl SizeUnit {
    /// Token counts for `text`, or `None` when the unit is chars.
    pub(crate) fn token_measure(&self, text: &str) -> Result<Option<TokenMeasure>, KonanError> {
        let Some(ends) = self.token_ends(text)? else {
            return Ok(None);
        };
        Ok(Some(TokenMeasure {
            ends,
            unit: self.clone(),
        }))
    }

    fn token_ends(&self, text: &str) -> Result<Option<Vec<usize>>, KonanError> {
        let mut ends = match self {
            Self::Chars => return Ok(None),
            Self::Tokens(encoding) => encoding.token_ends(text)?,
            Self::Tokenizer(tokenizer) => tokenizer.token_ends(text)?,
        };
        if !ends.is_sorted() {
            ends.sort_unstable();
        }
        Ok(Some(ends))
    }
}

impl fmt::Debug for SizeUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Chars => f.write_str("Chars"),
            Self::Tokens(encoding) => f.debug_tuple("Tokens").field(encoding).finish(),
            Self::Tokenizer(_) => f.write_str("Tokenizer(..)"),
        }
    }
}

/// Size of byte spans of one source text, in some unit.
///
/// `len` must be additive: `len(a, b) + len(b, c) == len(a, c)`. The merge
/// loops grow window sizes incrementally and depend on it.
pub(crate) trait Measure {
    /// Whether merges must drop carried overlap until the next unit fits, so
    /// no chunk exceeds the budget. Char budgets keep their original merge,
    /// which can overshoot by up to the overlap.
    const STRICT_BUDGET: bool = false;

    fn len(&self, start: usize, end: usize) -> usize;

    /// Budget consumed by `prefix`, text prepended to chunks that is not part
    /// of the source.
    fn reserve(&self, prefix: &str) -> Result<usize, KonanError>;

    /// Splits `span` into consecutive pieces of at most `size` units, each on
    /// char boundaries.
    fn split_even(
        &self,
        text: &str,
        span: (usize, usize),
        size: usize,
        out: &mut Vec<(usize, usize)>,
    );
}

impl Measure for OffsetMap<'_> {
    #[inline]
    fn len(&self, start: usize, end: usize) -> usize {
        self.char_len(start, end)
    }

    /// Char budgets have never counted prefixes.
    fn reserve(&self, _: &str) -> Result<usize, KonanError> {
        Ok(0)
    }

    fn split_even(
        &self,
        text: &str,
        span: (usize, usize),
        size: usize,
        out: &mut Vec<(usize, usize)>,
    ) {
        let mut start = span.0;
        let mut count = 0;
        for (b, _) in text[span.0..span.1].char_indices() {
            let abs = span.0 + b;
            if count == size {
                out.push((start, abs));
                start = abs;
                count = 0;
            }
            count += 1;
        }
        if start < span.1 {
            out.push((start, span.1));
        }
    }
}

/// Token counts from one tokenization of the whole document.
///
/// A span's size is the number of tokens that end inside it, which is
/// additive and gives every word its token: `" word"` starts on the space
/// before the word but ends with it. Re-tokenizing a chunk on its own can
/// differ by a token at each edge, where document tokens straddle it.
pub(crate) struct TokenMeasure {
    ends: Vec<usize>,
    unit: SizeUnit,
}

impl TokenMeasure {
    /// Number of tokens ending at or before `byte`.
    fn rank(&self, byte: usize) -> usize {
        self.ends.partition_point(|&o| o <= byte)
    }
}

impl Measure for TokenMeasure {
    const STRICT_BUDGET: bool = true;

    #[inline]
    fn len(&self, start: usize, end: usize) -> usize {
        self.rank(end) - self.rank(start)
    }

    fn reserve(&self, prefix: &str) -> Result<usize, KonanError> {
        Ok(self.unit.token_ends(prefix)?.map_or(0, |ends| ends.len()))
    }

    fn split_even(
        &self,
        text: &str,
        span: (usize, usize),
        size: usize,
        out: &mut Vec<(usize, usize)>,
    ) {
        let first = self.rank(span.0);
        let last = self.rank(span.1);
        let mut start = span.0;
        for k in (first + size - 1..last).step_by(size) {
            let mut cut = self.ends[k].min(span.1);
            while !text.is_char_boundary(cut) {
                cut -= 1;
            }
            if cut <= start || cut >= span.1 {
                continue;
            }
            out.push((start, cut));
            start = cut;
        }
        if start < span.1 {
            out.push((start, span.1));
        }
    }
}

/// Byte offset where each token starts, plus a sentinel holding the end
/// offset, all shifted by `base`. `token_len` is an O(1) lookup of the token's
/// raw byte length, so the offsets tile the source exactly.
pub(crate) fn token_offsets_at(tokenizer: &Tokenizer, text: &str, base: usize) -> Vec<usize> {
    let tokens = tokenizer.encode(text);
    let mut offsets = Vec::with_capacity(tokens.len() + 1);
    let mut pos = base;
    for &tok in &tokens {
        offsets.push(pos);
        pos += tokenizer.bpe.token_len(tok);
    }
    offsets.push(pos);
    offsets
}

/// [`token_offsets_at`] over the whole text, spreading the BPE pass across
/// cores for inputs big enough to pay for the split.
///
/// It parallelises exactly because the pretokenizer never emits a token
/// spanning a paragraph break; see [`paragraph_cuts`].
pub(crate) fn token_offsets(tokenizer: &Tokenizer, text: &str) -> Vec<usize> {
    /// Below this the join and the cut scan cost more than they save.
    const PARALLEL_MIN_BYTES: usize = 256 * 1024;

    if text.len() < PARALLEL_MIN_BYTES {
        return token_offsets_at(tokenizer, text, 0);
    }
    // Over-subscribing keeps threads fed when paragraph sizes are uneven.
    let cuts = paragraph_cuts(text, rayon::current_num_threads() * 8);
    if cuts.len() <= 2 {
        return token_offsets_at(tokenizer, text, 0);
    }
    use rayon::prelude::*;
    let windows: Vec<(usize, usize)> = cuts.windows(2).map(|w| (w[0], w[1])).collect();
    let last = windows.len() - 1;
    let mut pieces: Vec<Vec<usize>> = windows
        .into_par_iter()
        .map(|(s, e)| token_offsets_at(tokenizer, &text[s..e], s))
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

#[cfg(test)]
mod tests {
    use super::*;

    fn measure(encoding: Encoding, text: &str) -> TokenMeasure {
        SizeUnit::Tokens(encoding)
            .token_measure(text)
            .unwrap()
            .expect("token unit")
    }

    const TEXT: &str = "Zwölf Boxkämpfer jagen Viktor quer über den großen Sylter Deich. 😀 done.\n\nNext paragraph here.";

    #[test]
    fn token_len_is_additive() {
        let m = measure(Encoding::Cl100kBase, TEXT);
        let bounds: Vec<usize> = TEXT
            .char_indices()
            .map(|(i, _)| i)
            .chain([TEXT.len()])
            .collect();
        for &a in bounds.iter().step_by(3) {
            for &b in bounds.iter().filter(|&&b| b >= a).step_by(5) {
                for &c in bounds.iter().filter(|&&c| c >= b).step_by(7) {
                    assert_eq!(m.len(a, b) + m.len(b, c), m.len(a, c));
                }
            }
        }
        assert_eq!(
            m.len(0, TEXT.len()),
            Encoding::Cl100kBase.tokenizer().count(TEXT)
        );
    }

    #[test]
    fn token_split_even_tiles_span_on_char_boundaries() {
        let m = measure(Encoding::O200kBase, TEXT);
        for size in [1usize, 2, 3, 7] {
            let mut out = Vec::new();
            m.split_even(TEXT, (4, TEXT.len()), size, &mut out);
            assert_eq!(out.first().map(|p| p.0), Some(4));
            assert_eq!(out.last().map(|p| p.1), Some(TEXT.len()));
            for w in out.windows(2) {
                assert_eq!(w[0].1, w[1].0);
            }
            for &(s, e) in &out {
                assert!(TEXT.is_char_boundary(s) && TEXT.is_char_boundary(e) && s < e);
                assert!(m.len(s, e) <= size, "size={size} piece={:?}", &TEXT[s..e]);
            }
        }
    }

    #[test]
    fn unknown_encoding_rejected() {
        assert!(matches!(
            Encoding::from_name("nope"),
            Err(KonanError::Tokenizer(_))
        ));
        assert_eq!(
            Encoding::from_name("o200k_base").map(Encoding::name).ok(),
            Some("o200k_base")
        );
    }
}
