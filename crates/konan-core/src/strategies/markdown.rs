use std::fmt::Write as _;
use std::sync::Arc;

use crate::chunk::Chunk;
use crate::chunker::Chunker;
use crate::error::KonanError;
use crate::size::{Measure, SizeUnit};
use crate::strategies::recursive::RecursiveChunker;
use crate::text::{merge_spans, stable_merge, OffsetMap};

/// Markdown-structure-aware chunker. Splits along top-level blocks, never
/// splits fenced code blocks, and prefixes each chunk with its heading
/// breadcrumb (e.g. "# A > ## B"). For breadcrumbed chunks,
/// `start`/`end` refer to the source content span (the breadcrumb prefix is
/// not part of the source).
pub struct MarkdownChunker {
    chunk_size: usize,
    chunk_overlap: usize,
    splitter: RecursiveChunker,
    size_unit: SizeUnit,
    stable_boundaries: bool,
}

struct Block {
    span: (usize, usize),
    is_code: bool,
    heading: Option<u32>,
}

fn heading_level(level: pulldown_cmark::HeadingLevel) -> u32 {
    use pulldown_cmark::HeadingLevel::*;
    match level {
        H1 => 1,
        H2 => 2,
        H3 => 3,
        H4 => 4,
        H5 => 5,
        H6 => 6,
    }
}

fn parse_blocks(text: &str) -> Vec<Block> {
    use pulldown_cmark::{Event, Options, Parser, Tag};
    let mut blocks = Vec::new();
    let mut depth = 0usize;
    for (event, range) in Parser::new_ext(text, Options::empty()).into_offset_iter() {
        match event {
            Event::Start(tag) => {
                if depth == 0 {
                    let heading = match &tag {
                        Tag::Heading { level, .. } => Some(heading_level(*level)),
                        _ => None,
                    };
                    blocks.push(Block {
                        span: (range.start, range.end),
                        is_code: matches!(tag, Tag::CodeBlock(_)),
                        heading,
                    });
                }
                depth += 1;
            }
            Event::End(_) => depth -= 1,
            Event::Rule | Event::Html(_) if depth == 0 => {
                blocks.push(Block {
                    span: (range.start, range.end),
                    is_code: false,
                    heading: None,
                });
            }
            _ => {}
        }
    }
    blocks
}

impl MarkdownChunker {
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Result<Self, KonanError> {
        if chunk_size == 0 {
            return Err(KonanError::InvalidConfig("chunk_size must be > 0".into()));
        }
        if chunk_overlap >= chunk_size {
            return Err(KonanError::InvalidConfig(
                "chunk_overlap must be smaller than chunk_size".into(),
            ));
        }
        let splitter = RecursiveChunker::new(chunk_size, chunk_overlap, None)?;
        Ok(Self {
            chunk_size,
            chunk_overlap,
            splitter,
            size_unit: SizeUnit::Chars,
            stable_boundaries: false,
        })
    }

    /// Measures `chunk_size` and `chunk_overlap` in `size_unit` instead of chars.
    pub fn with_size_unit(mut self, size_unit: SizeUnit) -> Self {
        self.size_unit = size_unit;
        self
    }

    /// Picks cut points from chunk content instead of filling each chunk
    /// greedily, so an edit only changes the chunks near it.
    pub fn with_stable_boundaries(mut self, stable_boundaries: bool) -> Self {
        self.stable_boundaries = stable_boundaries;
        self
    }

    #[allow(clippy::too_many_arguments)]
    fn flush_section(
        &self,
        src: &Arc<str>,
        map: &OffsetMap,
        measure: &impl Measure,
        cursor: &mut crate::text::CharCursor,
        section: &[(usize, usize, bool)],
        breadcrumb: &str,
        chunks: &mut Vec<Chunk>,
    ) -> Result<(), KonanError> {
        if section.is_empty() {
            return Ok(());
        }
        let text: &str = src;
        let reserved = if breadcrumb.is_empty() {
            0
        } else {
            measure.reserve(breadcrumb)? + measure.reserve("\n\n")?
        };
        let size = self.chunk_size.saturating_sub(reserved).max(1);
        let overlap = self.chunk_overlap.min(size - 1);
        // Oversized non-code blocks get recursively split; code blocks stay atomic.
        let mut units: Vec<(usize, usize)> = Vec::new();
        for &(s, e, is_code) in section {
            if !is_code && measure.len(s, e) > size {
                self.splitter
                    .split_units(text, measure, (s, e), size, 0, &mut units);
            } else {
                units.push((s, e));
            }
        }
        let merged = if self.stable_boundaries {
            stable_merge(measure, text, units.into_iter(), size, overlap)
        } else {
            merge_spans(measure, &units, size, overlap)
        };
        for (s, e) in merged {
            let content = text[s..e].trim_end();
            let content_end = s + content.len();
            let index = chunks.len();
            let start = map.char_at(cursor, s);
            // char_len takes the ASCII fast path; chars().count() never could.
            let end = start + map.char_len(s, content_end);
            if breadcrumb.is_empty() {
                // Verbatim source, so the chunk can borrow instead of copying.
                chunks.push(Chunk::shared(src, s, content_end, start, end, index));
            } else {
                let mut chunk_text = String::with_capacity(breadcrumb.len() + 2 + content.len());
                chunk_text.push_str(breadcrumb);
                chunk_text.push_str("\n\n");
                chunk_text.push_str(content);
                chunks.push(Chunk::new(chunk_text, start, end, index));
            }
        }
        Ok(())
    }
}

impl Chunker for MarkdownChunker {
    fn chunk_shared(&self, src: &Arc<str>) -> Result<Vec<Chunk>, KonanError> {
        let text: &str = src;
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }
        let map = OffsetMap::new(text);
        match self.size_unit.token_measure(text)? {
            None => self.sections(src, &map, &map),
            Some(tokens) => self.sections(src, &map, &tokens),
        }
    }
}

impl MarkdownChunker {
    fn sections(
        &self,
        src: &Arc<str>,
        map: &OffsetMap,
        measure: &impl Measure,
    ) -> Result<Vec<Chunk>, KonanError> {
        let text: &str = src;
        // Merged span starts ascend across sections, so one cursor suffices.
        let mut cursor = map.cursor();
        let mut chunks: Vec<Chunk> = Vec::new();
        // Heading stack, each entry recording where its segment starts in
        // `breadcrumb` so popping a level is a truncate rather than a rebuild.
        let mut stack: Vec<(u32, usize)> = Vec::new();
        let mut section: Vec<(usize, usize, bool)> = Vec::new();
        let mut breadcrumb = String::new();

        for block in parse_blocks(text) {
            if let Some(level) = block.heading {
                self.flush_section(
                    src,
                    map,
                    measure,
                    &mut cursor,
                    &section,
                    &breadcrumb,
                    &mut chunks,
                )?;
                section.clear();
                let title = text[block.span.0..block.span.1]
                    .trim()
                    .lines()
                    .next()
                    .unwrap_or_default()
                    .trim_start_matches('#')
                    .trim();
                // Drop sibling and deeper headings, then append this one. The
                // breadcrumb is grown in place instead of being re-joined from a
                // Vec<String> at every heading.
                while let Some(&(l, at)) = stack.last() {
                    if l < level {
                        break;
                    }
                    breadcrumb.truncate(at);
                    stack.pop();
                }
                let at = breadcrumb.len();
                if at > 0 {
                    breadcrumb.push_str(" > ");
                }
                for _ in 0..level {
                    breadcrumb.push('#');
                }
                let _ = write!(breadcrumb, " {title}");
                stack.push((level, at));
            } else {
                section.push((block.span.0, block.span.1, block.is_code));
            }
        }
        self.flush_section(
            src,
            map,
            measure,
            &mut cursor,
            &section,
            &breadcrumb,
            &mut chunks,
        )?;
        Ok(chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::size::Encoding;
    use crate::text::{new_chunks, varied_prose};

    const MD: &str = "# Guide\n\nIntro paragraph about the guide.\n\n## Install\n\nRun the installer.\n\n```bash\npip install konan\n```\n";

    fn source_slice(text: &str, c: &Chunk) -> String {
        text.chars().skip(c.start).take(c.end - c.start).collect()
    }

    #[test]
    fn breadcrumbs_prefix_chunks() {
        let chunks = MarkdownChunker::new(200, 0).unwrap().chunk(MD).unwrap();
        assert!(chunks[0].text().starts_with("# Guide\n\nIntro paragraph"));
        assert!(chunks
            .iter()
            .any(|c| c.text().starts_with("# Guide > ## Install")));
        for c in &chunks {
            assert!(c.text().ends_with(&source_slice(MD, c)));
        }
    }

    #[test]
    fn code_fences_never_split() {
        let chunks = MarkdownChunker::new(10, 0).unwrap().chunk(MD).unwrap();
        let code: Vec<_> = chunks
            .iter()
            .filter(|c| c.text().contains("```bash"))
            .collect();
        assert_eq!(code.len(), 1);
        assert!(code[0].text().contains("pip install konan"));
    }

    #[test]
    fn plain_text_without_headings() {
        let text = "Just a paragraph.\n\nAnother paragraph.";
        let chunks = MarkdownChunker::new(200, 0).unwrap().chunk(text).unwrap();
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text().starts_with("Just a paragraph."));
    }

    #[test]
    fn sibling_heading_replaces_breadcrumb() {
        let md = "## A\n\none\n\n## B\n\ntwo\n";
        let chunks = MarkdownChunker::new(200, 0).unwrap().chunk(md).unwrap();
        assert!(chunks[0].text().starts_with("## A"));
        assert!(chunks[1].text().starts_with("## B"));
        assert!(!chunks[1].text().contains("## A"));
    }

    #[test]
    fn setext_heading_breadcrumb() {
        let md = "Title\n=====\n\nBody text here.\n\nSub\n---\n\nMore body.\n";
        let chunks = MarkdownChunker::new(200, 0).unwrap().chunk(md).unwrap();
        // No chunk should have an underline leaked into its breadcrumb
        assert!(
            !chunks.iter().any(|c| c.text().contains("=====")),
            "underline leaked into breadcrumb"
        );
        assert!(
            !chunks.iter().any(|c| c.text().contains("---\n")),
            "setext underline leaked into breadcrumb"
        );
        // The H1 section chunk must start with the clean breadcrumb
        assert!(
            chunks[0].text().starts_with("# Title\n\n"),
            "got: {:?}",
            chunks[0].text()
        );
        // The H2 section chunk must contain the nested breadcrumb without any underlines
        let sub_chunk = chunks
            .iter()
            .find(|c| c.text().contains("## Sub"))
            .expect("no Sub chunk");
        assert!(
            sub_chunk.text().starts_with("# Title > ## Sub\n\n"),
            "got: {:?}",
            sub_chunk.text()
        );
    }

    fn long_markdown() -> String {
        let prose = varied_prose(900);
        let paragraphs: Vec<&str> = prose.split("\n\n").collect();
        let mut md = String::new();
        for (i, p) in paragraphs.iter().enumerate() {
            if i % 15 == 0 {
                md.push_str(&format!("# Handbook volume {i} on rivers and stones\n\n"));
            }
            if i % 5 == 0 {
                md.push_str(&format!("## Part {i}: the long winter harbor chapter\n\n"));
            }
            if i % 5 == 2 {
                md.push_str("### Copper lanterns under pale moonlight\n\n");
            }
            md.push_str(p);
            md.push_str("\n\n");
        }
        md
    }

    #[test]
    fn token_size_unit_fits_budget() {
        let md = long_markdown();
        for (overlap, stable) in [(0, false), (16, false), (16, true)] {
            let chunks = MarkdownChunker::new(64, overlap)
                .unwrap()
                .with_size_unit(SizeUnit::Tokens(Encoding::O200kBase))
                .with_stable_boundaries(stable)
                .chunk(&md)
                .unwrap();
            assert!(chunks.len() > 10);
            for c in &chunks {
                assert!(c.text().ends_with(&source_slice(&md, c)));
                let n = Encoding::O200kBase.tokenizer().count(c.text());
                assert!(
                    n <= 64 + 3,
                    "chunk {} has {n} tokens with its breadcrumb",
                    c.index
                );
            }
        }
    }

    #[test]
    fn stable_boundaries_survive_an_edit() {
        let md = long_markdown();
        let mid = md.len() / 2;
        let at = mid + md[mid..].find(". ").unwrap() + 2;
        let edited = format!(
            "{}Inserted sentence about nothing much. {}",
            &md[..at],
            &md[at..]
        );
        let run = |stable: bool| {
            let chunker = MarkdownChunker::new(120, 0)
                .unwrap()
                .with_stable_boundaries(stable);
            new_chunks(
                &chunker.chunk(&md).unwrap(),
                &chunker.chunk(&edited).unwrap(),
            )
        };
        assert!(
            run(true) <= 3,
            "stable boundaries changed {} chunks",
            run(true)
        );
    }
}
