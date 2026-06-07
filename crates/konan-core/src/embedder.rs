use std::time::Duration;

use async_openai::config::OpenAIConfig;
use async_openai::middleware::retry::OpenAIRetryLayer;
use async_openai::middleware::ReqwestService;
use async_openai::types::embeddings::{CreateEmbeddingRequest, EmbeddingInput};
use async_openai::Client;
use async_trait::async_trait;
use tower::Layer;

use crate::error::KonanError;

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

#[async_trait]
impl<T: Embedder + ?Sized> Embedder for Box<T> {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, KonanError> {
        (**self).embed(texts).await
    }
}

/// Adapter: any OpenAI-compatible `/embeddings` endpoint, backed by the
/// async-openai client. Requests are batched, bounded by a request timeout,
/// and retried by the client's retry layer (exponential backoff, honoring
/// Retry-After) on 429s, server errors and connection failures.
pub struct OpenAIEmbedder {
    client: Client<OpenAIConfig>,
    model: String,
    batch_size: usize,
    dimensions: Option<u32>,
}

impl OpenAIEmbedder {
    pub fn new(
        base_url: impl Into<String>,
        model: impl Into<String>,
        api_key: Option<String>,
        batch_size: usize,
        timeout: f64,
        max_retries: u32,
        dimensions: Option<u32>,
    ) -> Result<Self, KonanError> {
        if batch_size == 0 {
            return Err(KonanError::InvalidConfig("batch_size must be > 0".into()));
        }
        if !timeout.is_finite() || timeout <= 0.0 {
            return Err(KonanError::InvalidConfig("timeout must be > 0".into()));
        }
        if dimensions == Some(0) {
            return Err(KonanError::InvalidConfig("dimensions must be > 0".into()));
        }
        let base_url = base_url.into().trim_end_matches('/').to_string();
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs_f64(timeout))
            .build()
            .map_err(|e| KonanError::Embedding(e.to_string()))?;
        let mut config = OpenAIConfig::new().with_api_base(base_url);
        if let Some(key) = api_key {
            config = config.with_api_key(key);
        }
        // with_http_client installs the timeout-bearing client for request
        // building; with_http_service replaces the executor (and its default
        // 3-retry layer) with one honoring our max_retries.
        let retry_service =
            OpenAIRetryLayer::new(max_retries as usize).layer(ReqwestService::new(http_client.clone()));
        let client = Client::with_config(config)
            .with_http_client(http_client)
            .with_http_service(retry_service);
        Ok(Self {
            client,
            model: model.into(),
            batch_size,
            dimensions,
        })
    }

    async fn embed_batch(&self, batch: &[String]) -> Result<Vec<Vec<f32>>, KonanError> {
        let request = CreateEmbeddingRequest {
            model: self.model.clone(),
            input: EmbeddingInput::StringArray(batch.to_vec()),
            encoding_format: None,
            user: None,
            dimensions: self.dimensions,
        };
        let response = self
            .client
            .embeddings()
            .create(request)
            .await
            .map_err(|e| KonanError::Embedding(e.to_string()))?;
        if response.data.len() != batch.len() {
            return Err(KonanError::Embedding(format!(
                "expected {} embeddings, got {}",
                batch.len(),
                response.data.len()
            )));
        }
        let mut data = response.data;
        data.sort_by_key(|d| d.index);
        Ok(data.into_iter().map(|d| d.embedding).collect())
    }
}

#[async_trait]
impl Embedder for OpenAIEmbedder {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, KonanError> {
        let mut out: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
        for batch in texts.chunks(self.batch_size) {
            out.extend(self.embed_batch(batch).await?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn embedder(base_url: &str, max_retries: u32) -> Result<OpenAIEmbedder, KonanError> {
        OpenAIEmbedder::new(base_url, "m", None, 16, 1.0, max_retries, None)
    }

    #[test]
    fn validates_config() {
        assert!(OpenAIEmbedder::new("http://x", "m", None, 0, 1.0, 0, None).is_err());
        assert!(OpenAIEmbedder::new("http://x", "m", None, 16, 0.0, 0, None).is_err());
        assert!(OpenAIEmbedder::new("http://x", "m", None, 16, f64::NAN, 0, None).is_err());
        assert!(OpenAIEmbedder::new("http://x", "m", None, 16, 1.0, 0, Some(0)).is_err());
        assert!(embedder("http://x", 0).is_ok());
    }

    #[tokio::test]
    async fn unreachable_endpoint_is_embedding_error() {
        let e = embedder("http://127.0.0.1:9", 0).unwrap();
        let err = e.embed(&["x".to_string()]).await.unwrap_err();
        assert!(matches!(err, KonanError::Embedding(_)));
    }

    #[tokio::test]
    async fn retries_are_bounded() {
        // Connection refused is transient; with max_retries=1 the call must
        // still fail after the bounded retry rather than loop forever.
        let e = embedder("http://127.0.0.1:9", 1).unwrap();
        let err = e.embed(&["x".to_string()]).await.unwrap_err();
        assert!(matches!(err, KonanError::Embedding(_)));
    }
}
