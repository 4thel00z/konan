#[derive(Debug, thiserror::Error)]
pub enum KonanError {
    #[error("invalid config: {0}")]
    InvalidConfig(String),
    #[error("embedding error: {0}")]
    Embedding(String),
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
}
