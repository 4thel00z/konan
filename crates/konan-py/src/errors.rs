use konan_core::KonanError;
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyValueError};

create_exception!(
    konan,
    EmbeddingError,
    PyException,
    "Embedding backend failure."
);

pub(crate) fn to_py_err(err: KonanError) -> pyo3::PyErr {
    match err {
        KonanError::InvalidConfig(msg) | KonanError::Tokenizer(msg) => PyValueError::new_err(msg),
        KonanError::Embedding(msg) => EmbeddingError::new_err(msg),
    }
}
