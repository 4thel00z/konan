use unicode_general_category::{get_general_category, GeneralCategory};

/// Bounded evidence about the non-whitespace content in a text.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TextMetrics {
    pub content_characters: usize,
    pub alphanumeric_characters: usize,
    pub exceeded_limit: bool,
}

fn is_python_whitespace(character: char) -> bool {
    matches!(
        character,
        '\u{0009}'..='\u{000d}'
            | '\u{001c}'..='\u{0020}'
            | '\u{0085}'
            | '\u{00a0}'
            | '\u{1680}'
            | '\u{2000}'..='\u{200a}'
            | '\u{2028}'
            | '\u{2029}'
            | '\u{202f}'
            | '\u{205f}'
            | '\u{3000}'
    )
}

fn is_python_alphanumeric(character: char) -> bool {
    matches!(
        get_general_category(character),
        GeneralCategory::UppercaseLetter
            | GeneralCategory::LowercaseLetter
            | GeneralCategory::TitlecaseLetter
            | GeneralCategory::ModifierLetter
            | GeneralCategory::OtherLetter
            | GeneralCategory::DecimalNumber
            | GeneralCategory::LetterNumber
            | GeneralCategory::OtherNumber
    )
}

/// Measure non-whitespace characters without constructing a compacted copy.
///
/// NUL characters are ignored because document extractors commonly emit them
/// as layout noise. The scan stops before classifying the source character
/// after `max_source_characters`, bounding work even when oversized text is
/// entirely whitespace or NULs.
///
/// Classification uses Unicode 15.0 category data to match Python 3.12's
/// `str.isspace()` and `str.isalnum()` for Unicode scalar values. Lone
/// surrogates are not Unicode scalar values and cannot be represented by Rust
/// `str`.
pub fn measure_text(text: &str, max_source_characters: usize) -> TextMetrics {
    let mut content_characters = 0;
    let mut alphanumeric_characters = 0;

    for (source_index, character) in text.chars().enumerate() {
        if source_index >= max_source_characters {
            return TextMetrics {
                content_characters,
                alphanumeric_characters,
                exceeded_limit: true,
            };
        }
        if character == '\0' || is_python_whitespace(character) {
            continue;
        }
        content_characters += 1;
        if is_python_alphanumeric(character) {
            alphanumeric_characters += 1;
        }
    }

    TextMetrics {
        content_characters,
        alphanumeric_characters,
        exceeded_limit: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measures_unicode_content_and_ignores_layout_noise() {
        let metrics = measure_text(" A\u{a0}é日\0! ", 10);

        assert_eq!(
            metrics,
            TextMetrics {
                content_characters: 4,
                alphanumeric_characters: 3,
                exceeded_limit: false,
            }
        );
    }

    #[test]
    fn stops_after_source_limit() {
        let metrics = measure_text("a \n b c trailing content", 2);

        assert_eq!(
            metrics,
            TextMetrics {
                content_characters: 1,
                alphanumeric_characters: 1,
                exceeded_limit: true,
            }
        );
    }

    #[test]
    fn exact_limit_is_not_exceeded() {
        let metrics = measure_text("a b", 3);

        assert_eq!(
            metrics,
            TextMetrics {
                content_characters: 2,
                alphanumeric_characters: 2,
                exceeded_limit: false,
            }
        );
    }

    #[test]
    fn whitespace_and_nul_prefixes_are_source_bounded() {
        for text in [" ".repeat(100), "\0".repeat(100)] {
            assert_eq!(
                measure_text(&text, 2),
                TextMetrics {
                    content_characters: 0,
                    alphanumeric_characters: 0,
                    exceeded_limit: true,
                }
            );
        }
    }

    #[test]
    fn zero_limit_rejects_non_empty_source_without_counting_it() {
        assert_eq!(
            measure_text("a", 0),
            TextMetrics {
                content_characters: 0,
                alphanumeric_characters: 0,
                exceeded_limit: true,
            }
        );
        assert!(!measure_text("", 0).exceeded_limit);
    }

    #[test]
    fn matches_python_specific_unicode_classification() {
        let metrics = measure_text("\u{1c}\u{345}\u{5b0}A1", 5);
        assert_eq!(metrics.content_characters, 4);
        assert_eq!(metrics.alphanumeric_characters, 2);
    }
}
