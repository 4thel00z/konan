use std::sync::Arc;

use xxhash_rust::xxh3::xxh3_64;

/// Where a chunk's text lives.
///
/// Nearly every chunk is a verbatim slice of its source, so the common case
/// borrows from one shared buffer instead of copying. `MarkdownChunker` is the
/// exception: it prefixes a heading breadcrumb, so its text is not a slice of
/// the source and has to be owned.
#[derive(Debug, Clone)]
enum ChunkText {
    /// Byte range into a source shared by every chunk of one document.
    Shared {
        src: Arc<str>,
        start: u32,
        end: u32,
    },
    Owned(Box<str>),
}

impl ChunkText {
    #[inline]
    fn as_str(&self) -> &str {
        match self {
            // The range came from a char-boundary-checked span at construction.
            ChunkText::Shared { src, start, end } => &src[*start as usize..*end as usize],
            ChunkText::Owned(s) => s,
        }
    }
}

/// A chunk of text with char offsets into its source.
///
/// Chunks produced from one document share a single reference-counted copy of
/// that document's text rather than each owning their own `String`. With the
/// default overlap that removes ~1.25 copies of the input plus two allocations
/// per chunk. The trade-off is that holding on to one chunk keeps the whole
/// source alive; clone `text()` into an owned `String` if you need to drop the
/// rest.
#[derive(Clone)]
pub struct Chunk {
    text: ChunkText,
    /// Char offset (not bytes) of the chunk content in the source text.
    pub start: usize,
    /// Char offset (exclusive) of the chunk content in the source text.
    pub end: usize,
    /// 0-based chunk index.
    pub index: usize,
    /// xxh3-64 digest of `text()`. Use [`Chunk::hash_hex`] for the hex digest —
    /// formatting it eagerly cost ~14% of total chunking time.
    pub hash: u64,
}

impl Chunk {
    /// A chunk borrowing `src[byte_start..byte_end]`.
    ///
    /// `byte_start`/`byte_end` are byte offsets into `src` and must sit on char
    /// boundaries; `start`/`end` are the corresponding char offsets.
    pub(crate) fn shared(
        src: &Arc<str>,
        byte_start: usize,
        byte_end: usize,
        start: usize,
        end: usize,
        index: usize,
    ) -> Self {
        debug_assert!(src.is_char_boundary(byte_start) && src.is_char_boundary(byte_end));
        let hash = xxh3_64(src[byte_start..byte_end].as_bytes());
        Self {
            text: ChunkText::Shared {
                src: Arc::clone(src),
                start: byte_start as u32,
                end: byte_end as u32,
            },
            start,
            end,
            index,
            hash,
        }
    }

    /// A chunk owning its text, for text that is not a verbatim slice of the
    /// source.
    pub fn new(text: impl Into<String>, start: usize, end: usize, index: usize) -> Self {
        let text = text.into();
        let hash = xxh3_64(text.as_bytes());
        Self {
            text: ChunkText::Owned(text.into_boxed_str()),
            start,
            end,
            index,
            hash,
        }
    }

    /// The chunk's text.
    #[inline]
    pub fn text(&self) -> &str {
        self.text.as_str()
    }

    /// Lowercase hex form of [`Chunk::hash`], zero-padded to 16 digits.
    pub fn hash_hex(&self) -> String {
        format!("{:016x}", self.hash)
    }
}

/// Compares content, not which buffer it came from: two chunks with the same
/// text and offsets are equal whether one borrows and the other owns.
impl PartialEq for Chunk {
    fn eq(&self, other: &Self) -> bool {
        self.start == other.start
            && self.end == other.end
            && self.index == other.index
            && self.hash == other.hash
            && self.text() == other.text()
    }
}

impl Eq for Chunk {}

impl std::hash::Hash for Chunk {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.text().hash(state);
        self.start.hash(state);
        self.end.hash(state);
        self.index.hash(state);
        self.hash.hash(state);
    }
}

/// Prints as a flat struct so the `Shared`/`Owned` split stays an implementation
/// detail in assertion output.
impl std::fmt::Debug for Chunk {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Chunk")
            .field("text", &self.text())
            .field("start", &self.start)
            .field("end", &self.end)
            .field("index", &self.index)
            .field("hash", &self.hash_hex())
            .finish()
    }
}

/// Serialises to the same shape as before the arena change — in particular
/// `hash` stays a hex string, so stored output and wire formats are unaffected.
impl serde::Serialize for Chunk {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut st = s.serialize_struct("Chunk", 5)?;
        st.serialize_field("text", self.text())?;
        st.serialize_field("start", &self.start)?;
        st.serialize_field("end", &self.end)?;
        st.serialize_field("index", &self.index)?;
        st.serialize_field("hash", &self.hash_hex())?;
        st.end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_chunk_borrows_the_right_slice() {
        let src: Arc<str> = Arc::from("hello wonderful world");
        let c = Chunk::shared(&src, 6, 15, 6, 15, 1);
        assert_eq!(c.text(), "wonderful");
        assert_eq!((c.start, c.end, c.index), (6, 15, 1));
    }

    #[test]
    fn hash_is_xxh3_of_the_text() {
        let src: Arc<str> = Arc::from("hello world");
        let c = Chunk::shared(&src, 0, 5, 0, 5, 0);
        assert_eq!(c.hash, xxh3_64(b"hello"));
    }

    /// The digest users see must be byte-identical to the eagerly formatted hex
    /// string this replaced.
    #[test]
    fn hash_hex_matches_the_previous_format() {
        let src: Arc<str> = Arc::from("some chunk text");
        let c = Chunk::shared(&src, 0, 15, 0, 15, 0);
        assert_eq!(
            c.hash_hex(),
            format!("{:016x}", xxh3_64(b"some chunk text"))
        );
        assert_eq!(c.hash_hex().len(), 16, "digest must stay zero-padded");
    }

    #[test]
    fn small_hash_stays_zero_padded() {
        // A digest with leading zero nibbles must still be 16 chars wide.
        let c = Chunk {
            text: ChunkText::Owned("x".into()),
            start: 0,
            end: 1,
            index: 0,
            hash: 0xab,
        };
        assert_eq!(c.hash_hex(), "00000000000000ab");
    }

    #[test]
    fn owned_and_shared_with_equal_content_are_equal() {
        let src: Arc<str> = Arc::from("abc");
        let shared = Chunk::shared(&src, 0, 3, 0, 3, 0);
        let owned = Chunk::new("abc", 0, 3, 0);
        assert_eq!(shared, owned);

        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let h = |c: &Chunk| {
            let mut s = DefaultHasher::new();
            c.hash(&mut s);
            s.finish()
        };
        assert_eq!(h(&shared), h(&owned), "Eq chunks must hash alike");
    }

    #[test]
    fn different_content_is_not_equal() {
        let src: Arc<str> = Arc::from("abcdef");
        let a = Chunk::shared(&src, 0, 3, 0, 3, 0);
        let b = Chunk::shared(&src, 3, 6, 3, 6, 0);
        assert_ne!(a, b);
    }

    #[test]
    fn serialises_with_a_hex_hash_field() {
        let src: Arc<str> = Arc::from("hi");
        let c = Chunk::shared(&src, 0, 2, 0, 2, 3);
        let json = serde_json::to_string(&c).expect("serialisable");
        let expect = format!(
            r#"{{"text":"hi","start":0,"end":2,"index":3,"hash":"{:016x}"}}"#,
            xxh3_64(b"hi")
        );
        assert_eq!(json, expect);
    }

    #[test]
    fn multibyte_slices_are_preserved() {
        let text = "caf\u{e9} \u{1f980} \u{65e5}\u{672c}";
        let src: Arc<str> = Arc::from(text);
        let crab_start = text.find('\u{1f980}').expect("crab present");
        let c = Chunk::shared(
            &src,
            crab_start,
            crab_start + '\u{1f980}'.len_utf8(),
            5,
            6,
            0,
        );
        assert_eq!(c.text(), "\u{1f980}");
    }
}
