use crate::chunk::Chunk;
use crate::chunker::Chunker;
use crate::error::KonanError;
use crate::strategies::recursive::RecursiveChunker;
use crate::text::{merge_spans, OffsetMap};

/// Markdown-structure-aware chunker. Splits along top-level blocks, never
/// splits fenced code blocks, and prefixes each chunk with its heading
/// breadcrumb (e.g. "# A > ## B"). For breadcrumbed chunks,
/// `start`/`end` refer to the source content span (the breadcrumb prefix is
/// not part of the source).
pub struct MarkdownChunker {
    chunk_size: usize,
    chunk_overlap: usize,
    splitter: RecursiveChunker,
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
                blocks.push(Block { span: (range.start, range.end), is_code: false, heading: None });
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
        Ok(Self { chunk_size, chunk_overlap, splitter })
    }

    fn flush_section(
        &self,
        text: &str,
        map: &OffsetMap,
        section: &[(usize, usize, bool)],
        breadcrumb: &str,
        chunks: &mut Vec<Chunk>,
    ) {
        if section.is_empty() {
            return;
        }
        // Oversized non-code blocks get recursively split; code blocks stay atomic.
        let mut units: Vec<(usize, usize)> = Vec::new();
        for &(s, e, is_code) in section {
            if !is_code && map.char_len(s, e) > self.chunk_size {
                self.splitter.split_units(text, map, (s, e), 0, &mut units);
            } else {
                units.push((s, e));
            }
        }
        for (s, e) in merge_spans(map, &units, self.chunk_size, self.chunk_overlap) {
            let content = text[s..e].trim_end();
            let chunk_text = if breadcrumb.is_empty() {
                content.to_string()
            } else {
                format!("{breadcrumb}\n\n{content}")
            };
            let index = chunks.len();
            let end = map.char_idx(s) + content.chars().count();
            chunks.push(Chunk::new(chunk_text, map.char_idx(s), end, index));
        }
    }
}

impl Chunker for MarkdownChunker {
    fn chunk(&self, text: &str) -> Result<Vec<Chunk>, KonanError> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }
        let map = OffsetMap::new(text);
        let mut chunks: Vec<Chunk> = Vec::new();
        let mut stack: Vec<(u32, String)> = Vec::new();
        let mut section: Vec<(usize, usize, bool)> = Vec::new();
        let mut breadcrumb = String::new();

        for block in parse_blocks(text) {
            if let Some(level) = block.heading {
                self.flush_section(text, &map, &section, &breadcrumb, &mut chunks);
                section.clear();
                let title = text[block.span.0..block.span.1]
                    .trim()
                    .lines()
                    .next()
                    .unwrap_or_default()
                    .trim_start_matches('#')
                    .trim()
                    .to_string();
                while stack.last().is_some_and(|(l, _)| *l >= level) {
                    stack.pop();
                }
                stack.push((level, title));
                breadcrumb = stack
                    .iter()
                    .map(|(l, t)| format!("{} {}", "#".repeat(*l as usize), t))
                    .collect::<Vec<_>>()
                    .join(" > ");
            } else {
                section.push((block.span.0, block.span.1, block.is_code));
            }
        }
        self.flush_section(text, &map, &section, &breadcrumb, &mut chunks);
        Ok(chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MD: &str = "# Guide\n\nIntro paragraph about the guide.\n\n## Install\n\nRun the installer.\n\n```bash\npip install konan\n```\n";

    fn source_slice(text: &str, c: &Chunk) -> String {
        text.chars().skip(c.start).take(c.end - c.start).collect()
    }

    #[test]
    fn breadcrumbs_prefix_chunks() {
        let chunks = MarkdownChunker::new(200, 0).unwrap().chunk(MD).unwrap();
        assert!(chunks[0].text.starts_with("# Guide\n\nIntro paragraph"));
        assert!(chunks.iter().any(|c| c.text.starts_with("# Guide > ## Install")));
        for c in &chunks {
            assert!(c.text.ends_with(&source_slice(MD, c)));
        }
    }

    #[test]
    fn code_fences_never_split() {
        let chunks = MarkdownChunker::new(10, 0).unwrap().chunk(MD).unwrap();
        let code: Vec<_> = chunks.iter().filter(|c| c.text.contains("```bash")).collect();
        assert_eq!(code.len(), 1);
        assert!(code[0].text.contains("pip install konan"));
    }

    #[test]
    fn plain_text_without_headings() {
        let text = "Just a paragraph.\n\nAnother paragraph.";
        let chunks = MarkdownChunker::new(200, 0).unwrap().chunk(text).unwrap();
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.starts_with("Just a paragraph."));
    }

    #[test]
    fn sibling_heading_replaces_breadcrumb() {
        let md = "## A\n\none\n\n## B\n\ntwo\n";
        let chunks = MarkdownChunker::new(200, 0).unwrap().chunk(md).unwrap();
        assert!(chunks[0].text.starts_with("## A"));
        assert!(chunks[1].text.starts_with("## B"));
        assert!(!chunks[1].text.contains("## A"));
    }

    #[test]
    fn setext_heading_breadcrumb() {
        let md = "Title\n=====\n\nBody text here.\n\nSub\n---\n\nMore body.\n";
        let chunks = MarkdownChunker::new(200, 0).unwrap().chunk(md).unwrap();
        // No chunk should have an underline leaked into its breadcrumb
        assert!(!chunks.iter().any(|c| c.text.contains("=====")), "underline leaked into breadcrumb");
        assert!(!chunks.iter().any(|c| c.text.contains("---\n")), "setext underline leaked into breadcrumb");
        // The H1 section chunk must start with the clean breadcrumb
        assert!(chunks[0].text.starts_with("# Title\n\n"), "got: {:?}", chunks[0].text);
        // The H2 section chunk must contain the nested breadcrumb without any underlines
        let sub_chunk = chunks.iter().find(|c| c.text.contains("## Sub")).expect("no Sub chunk");
        assert!(sub_chunk.text.starts_with("# Title > ## Sub\n\n"), "got: {:?}", sub_chunk.text);
    }
}
