use crate::error::KonanError;
use async_trait::async_trait;

/// Port: an embedding backend. Adapters: OpenAIEmbedder (HTTP), or anything
/// injected from the outside (e.g. a Python async callable in konan-py).
#[async_trait]
pub trait Embedder: Send + Sync {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, KonanError>;
}

#[async_trait]
impl<T: Embedder + ?Sized> Embedder for std::sync::Arc<T> {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, KonanError> {
        (**self).embed(texts).await
    }
}

/// Adapter: any OpenAI-compatible `/embeddings` endpoint.
pub struct OpenAIEmbedder {
    client: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
    model: String,
    batch_size: usize,
}

impl OpenAIEmbedder {
    pub fn new(
        base_url: impl Into<String>,
        model: impl Into<String>,
        api_key: Option<String>,
        batch_size: usize,
    ) -> Result<Self, KonanError> {
        if batch_size == 0 {
            return Err(KonanError::InvalidConfig("batch_size must be > 0".into()));
        }
        Ok(Self {
            client: reqwest::Client::new(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key,
            model: model.into(),
            batch_size,
        })
    }
}

#[derive(serde::Deserialize)]
struct EmbeddingsResponse {
    data: Vec<EmbeddingItem>,
}

#[derive(serde::Deserialize)]
struct EmbeddingItem {
    index: usize,
    embedding: Vec<f32>,
}

#[async_trait]
impl Embedder for OpenAIEmbedder {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, KonanError> {
        let url = format!("{}/embeddings", self.base_url);
        let mut out: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
        for batch in texts.chunks(self.batch_size) {
            let mut req = self.client.post(&url).json(&serde_json::json!({
                "model": self.model,
                "input": batch,
            }));
            if let Some(key) = &self.api_key {
                req = req.bearer_auth(key);
            }
            let resp = req.send().await.map_err(|e| KonanError::Embedding(e.to_string()))?;
            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                return Err(KonanError::Embedding(format!("HTTP {status}: {body}")));
            }
            let mut parsed: EmbeddingsResponse =
                resp.json().await.map_err(|e| KonanError::Embedding(e.to_string()))?;
            if parsed.data.len() != batch.len() {
                return Err(KonanError::Embedding(format!(
                    "expected {} embeddings, got {}",
                    batch.len(),
                    parsed.data.len()
                )));
            }
            parsed.data.sort_by_key(|d| d.index);
            out.extend(parsed.data.into_iter().map(|d| d.embedding));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_openai_response_shape() {
        let json = r#"{"object":"list","model":"m","data":[
            {"object":"embedding","index":1,"embedding":[0.0,1.0]},
            {"object":"embedding","index":0,"embedding":[1.0,0.0]}
        ]}"#;
        let mut parsed: EmbeddingsResponse = serde_json::from_str(json).unwrap();
        parsed.data.sort_by_key(|d| d.index);
        assert_eq!(parsed.data[0].embedding, vec![1.0, 0.0]);
        assert_eq!(parsed.data[1].embedding, vec![0.0, 1.0]);
    }

    #[tokio::test]
    async fn unreachable_endpoint_is_embedding_error() {
        let e = OpenAIEmbedder::new("http://127.0.0.1:9", "m", None, 16).unwrap();
        let err = e.embed(&["x".to_string()]).await.unwrap_err();
        assert!(matches!(err, KonanError::Embedding(_)));
    }
}
