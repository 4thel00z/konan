//! [`TokenEnds`] for Hugging Face `tokenizer.json` tokenizers.

use std::path::Path;

use tokenizers::Tokenizer;

use crate::error::KonanError;
use crate::size::TokenEnds;

/// A Hugging Face tokenizer with truncation and padding switched off, since
/// both would drop or invent tokens. Special tokens are never added.
pub struct HfTokenizer {
    tokenizer: Tokenizer,
}

impl HfTokenizer {
    pub fn new(mut tokenizer: Tokenizer) -> Result<Self, KonanError> {
        tokenizer.with_truncation(None).map_err(tokenizer_error)?;
        tokenizer.with_padding(None);
        Ok(Self { tokenizer })
    }

    /// Loads a `tokenizer.json` file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, KonanError> {
        Self::new(Tokenizer::from_file(path).map_err(tokenizer_error)?)
    }

    /// Parses the contents of a `tokenizer.json` file.
    pub fn from_json(json: &str) -> Result<Self, KonanError> {
        Self::new(json.parse().map_err(tokenizer_error)?)
    }
}

impl TokenEnds for HfTokenizer {
    fn token_ends(&self, text: &str) -> Result<Vec<usize>, KonanError> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(tokenizer_error)?;
        Ok(encoding.get_offsets().iter().map(|&(_, end)| end).collect())
    }
}

fn tokenizer_error(err: tokenizers::Error) -> KonanError {
    KonanError::Tokenizer(err.to_string())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::size::SizeUnit;
    use crate::text::{assert_char_offsets, varied_prose};
    use crate::{Chunker, RecursiveChunker};

    /// One token per word and per punctuation run, lowercased first, with a
    /// truncation setting that must not take effect.
    const TOKENIZER_JSON: &str = r#"{
        "version": "1.0",
        "truncation": {"direction": "Right", "max_length": 4, "strategy": "LongestFirst", "stride": 0},
        "padding": null,
        "added_tokens": [],
        "normalizer": {"type": "Lowercase"},
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": null,
        "decoder": null,
        "model": {"type": "WordLevel", "vocab": {"[UNK]": 0, "river": 1}, "unk_token": "[UNK]"}
    }"#;

    #[test]
    fn ends_map_to_original_bytes_despite_normalization() {
        let tokenizer = HfTokenizer::from_json(TOKENIZER_JSON).unwrap();
        let text = "GRÜSSE, schöne Welt! River";
        let expected: Vec<usize> = ["GRÜSSE", ",", "schöne", "Welt", "!", "River"]
            .iter()
            .map(|word| text.find(word).unwrap() + word.len())
            .collect();
        assert_eq!(tokenizer.token_ends(text).unwrap(), expected);
    }

    #[test]
    fn invalid_json_is_a_tokenizer_error() {
        assert!(matches!(
            HfTokenizer::from_json("{}"),
            Err(KonanError::Tokenizer(_))
        ));
    }

    #[test]
    fn recursive_chunks_fit_the_tokenizer_budget() {
        let tokenizer = Arc::new(HfTokenizer::from_json(TOKENIZER_JSON).unwrap());
        let text = varied_prose(400) + " Schöne Grüße, 😀 Welt!";
        for (size, overlap, stable) in [(16usize, 0usize, false), (64, 0, false), (64, 16, true)] {
            let chunks = RecursiveChunker::new(size, overlap, None)
                .unwrap()
                .with_size_unit(SizeUnit::Tokenizer(tokenizer.clone()))
                .with_stable_boundaries(stable)
                .chunk(&text)
                .unwrap();
            assert!(chunks.len() > 5);
            assert_char_offsets(&text, &chunks);
            for c in &chunks {
                let n = tokenizer.token_ends(c.text()).unwrap().len();
                assert!(n <= size, "chunk {} has {n} tokens", c.index);
            }
        }
    }
}
