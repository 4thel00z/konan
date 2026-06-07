use crate::chunk::Chunk;
use crate::embedder::Embedder;
use crate::error::KonanError;
use crate::text::{merge_spans, sentence_spans, spans_to_chunks, OffsetMap};

/// Splits where embedding similarity between adjacent sentences drops.
/// Break rule: `threshold` (absolute cosine similarity) if set, otherwise the
/// `percentile`-th percentile of adjacent distances (default p95).
pub struct SemanticChunker<E: Embedder> {
    embedder: E,
    threshold: Option<f32>,
    percentile: f32,
    min_chunk_size: usize,
    max_chunk_size: Option<usize>,
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

impl<E: Embedder> SemanticChunker<E> {
    pub fn new(
        embedder: E,
        threshold: Option<f32>,
        percentile: f32,
        min_chunk_size: usize,
        max_chunk_size: Option<usize>,
    ) -> Result<Self, KonanError> {
        if let Some(t) = threshold {
            if !(-1.0..=1.0).contains(&t) {
                return Err(KonanError::InvalidConfig(
                    "threshold must be in [-1, 1]".into(),
                ));
            }
        }
        if !(0.0..=100.0).contains(&percentile) {
            return Err(KonanError::InvalidConfig(
                "percentile must be in [0, 100]".into(),
            ));
        }
        if let Some(m) = max_chunk_size {
            if m == 0 || m < min_chunk_size {
                return Err(KonanError::InvalidConfig(
                    "max_chunk_size must be > 0 and >= min_chunk_size".into(),
                ));
            }
        }
        Ok(Self {
            embedder,
            threshold,
            percentile,
            min_chunk_size,
            max_chunk_size,
        })
    }

    pub async fn chunk(&self, text: &str) -> Result<Vec<Chunk>, KonanError> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }
        let map = OffsetMap::new(text);
        let sents = sentence_spans(text);
        if sents.len() <= 1 {
            return Ok(spans_to_chunks(text, &map, &sents));
        }
        let sentence_texts: Vec<String> =
            sents.iter().map(|&(s, e)| text[s..e].to_string()).collect();
        let embeddings = self.embedder.embed(&sentence_texts).await?;
        if embeddings.len() != sents.len() {
            return Err(KonanError::Embedding(format!(
                "embedder returned {} embeddings for {} sentences",
                embeddings.len(),
                sents.len()
            )));
        }
        let sims: Vec<f32> = embeddings
            .windows(2)
            .map(|w| cosine_similarity(&w[0], &w[1]))
            .collect();
        if sims.iter().any(|s| s.is_nan()) {
            return Err(KonanError::Embedding(
                "embedder returned NaN values; embeddings must be finite".into(),
            ));
        }
        let cutoff = match self.threshold {
            Some(t) => t,
            None => {
                let mut distances: Vec<f32> = sims.iter().map(|s| 1.0 - s).collect();
                distances.sort_by(|a, b| a.partial_cmp(b).expect("NaN sims rejected above"));
                let rank =
                    ((self.percentile / 100.0) * (distances.len() - 1) as f32).round() as usize;
                1.0 - distances[rank.min(distances.len() - 1)]
            }
        };

        // Group sentences, breaking where similarity drops below the cutoff.
        let mut groups: Vec<Vec<(usize, usize)>> = vec![vec![sents[0]]];
        for (i, &sim) in sims.iter().enumerate() {
            if sim < cutoff {
                groups.push(Vec::new());
            }
            groups.last_mut().unwrap().push(sents[i + 1]);
        }

        // Enforce min_chunk_size: groups still too small absorb the next group,
        // and a trailing small group is absorbed back into its predecessor.
        if self.min_chunk_size > 0 {
            let group_len = |g: &Vec<(usize, usize)>| map.char_len(g[0].0, g.last().unwrap().1);
            let mut merged: Vec<Vec<(usize, usize)>> = Vec::new();
            for g in groups {
                let prev_small = merged
                    .last()
                    .is_some_and(|p| group_len(p) < self.min_chunk_size);
                if prev_small {
                    merged.last_mut().unwrap().extend(g);
                } else {
                    merged.push(g);
                }
            }
            if merged.len() >= 2 && group_len(merged.last().unwrap()) < self.min_chunk_size {
                let last = merged.pop().unwrap();
                merged.last_mut().unwrap().extend(last);
            }
            groups = merged;
        }

        // Enforce max_chunk_size: oversized groups re-split by sentence.
        if let Some(maxs) = self.max_chunk_size {
            let mut split: Vec<Vec<(usize, usize)>> = Vec::new();
            for g in groups {
                if map.char_len(g[0].0, g.last().unwrap().1) <= maxs {
                    split.push(g);
                } else {
                    for span in merge_spans(&map, &g, maxs, 0) {
                        split.push(vec![span]);
                    }
                }
            }
            groups = split;
        }

        let spans: Vec<(usize, usize)> = groups
            .iter()
            .filter(|g| !g.is_empty())
            .map(|g| (g[0].0, g.last().unwrap().1))
            .collect();
        Ok(spans_to_chunks(text, &map, &spans))
    }

    pub async fn chunk_many(&self, texts: &[String]) -> Result<Vec<Vec<Chunk>>, KonanError> {
        futures::future::try_join_all(texts.iter().map(|t| self.chunk(t))).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    struct FakeEmbedder;

    #[async_trait]
    impl Embedder for FakeEmbedder {
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, KonanError> {
            Ok(texts
                .iter()
                .map(|t| {
                    if t.to_lowercase().contains("cat") {
                        vec![1.0, 0.0]
                    } else {
                        vec![0.0, 1.0]
                    }
                })
                .collect())
        }
    }

    const TEXT: &str = "Cats purr softly. My cat naps all day. Quantum entanglement defies intuition. Quantum computers exploit superposition.";

    #[tokio::test]
    async fn splits_on_topic_shift_with_threshold() {
        let c = SemanticChunker::new(FakeEmbedder, Some(0.5), 95.0, 0, None).unwrap();
        let chunks = c.chunk(TEXT).await.unwrap();
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].text.contains("cat naps"));
        assert!(chunks[1].text.starts_with("Quantum"));
        crate::text::assert_char_offsets(TEXT, &chunks);
    }

    #[tokio::test]
    async fn percentile_mode_splits_largest_gap() {
        let c = SemanticChunker::new(FakeEmbedder, None, 50.0, 0, None).unwrap();
        let chunks = c.chunk(TEXT).await.unwrap();
        assert!(chunks.len() >= 2);
    }

    #[tokio::test]
    async fn single_sentence_is_single_chunk() {
        let c = SemanticChunker::new(FakeEmbedder, Some(0.5), 95.0, 0, None).unwrap();
        let chunks = c.chunk("Just one sentence here.").await.unwrap();
        assert_eq!(chunks.len(), 1);
    }

    #[tokio::test]
    async fn max_chunk_size_resplits() {
        let c = SemanticChunker::new(FakeEmbedder, Some(0.5), 95.0, 0, Some(25)).unwrap();
        let chunks = c.chunk(TEXT).await.unwrap();
        assert!(chunks.len() > 2);
    }

    #[test]
    fn validates_config() {
        assert!(SemanticChunker::new(FakeEmbedder, Some(2.0), 95.0, 0, None).is_err());
        assert!(SemanticChunker::new(FakeEmbedder, None, 200.0, 0, None).is_err());
        assert!(SemanticChunker::new(FakeEmbedder, None, 95.0, 100, Some(50)).is_err());
    }

    struct NanEmbedder;

    #[async_trait]
    impl Embedder for NanEmbedder {
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, KonanError> {
            Ok(texts.iter().map(|_| vec![f32::NAN, 1.0]).collect())
        }
    }

    #[tokio::test]
    async fn nan_embeddings_error_instead_of_panic() {
        let c = SemanticChunker::new(NanEmbedder, None, 95.0, 0, None).unwrap();
        let err = c.chunk(TEXT).await.unwrap_err();
        assert!(matches!(err, KonanError::Embedding(_)));
    }

    #[tokio::test]
    async fn trailing_small_group_absorbed_into_previous() {
        // Topic shift right before a tiny final sentence: without trailing
        // absorption the "Quantum!" group (8 chars) stays below min_chunk_size.
        let text = "Cats purr softly. My cat naps all day. Quantum!";
        let c = SemanticChunker::new(FakeEmbedder, Some(0.5), 95.0, 20, None).unwrap();
        let chunks = c.chunk(text).await.unwrap();
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.ends_with("Quantum!"));
        crate::text::assert_char_offsets(text, &chunks);
    }
}
