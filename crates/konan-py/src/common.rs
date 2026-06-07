use std::sync::Arc;

use konan_core::{chunk_many, Chunker};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::chunk::PyChunk;
use crate::errors::to_py_err;

pub(crate) fn do_chunk(
    py: Python<'_>,
    chunker: Arc<dyn Chunker>,
    text: String,
) -> PyResult<Vec<PyChunk>> {
    let chunks = py
        .allow_threads(move || chunker.chunk(&text))
        .map_err(to_py_err)?;
    Ok(chunks.into_iter().map(PyChunk::from).collect())
}

pub(crate) fn do_chunk_many(
    py: Python<'_>,
    chunker: Arc<dyn Chunker>,
    texts: Vec<String>,
) -> PyResult<Vec<Vec<PyChunk>>> {
    let results = py
        .allow_threads(move || chunk_many(&*chunker, &texts))
        .map_err(to_py_err)?;
    Ok(results
        .into_iter()
        .map(|cs| cs.into_iter().map(PyChunk::from).collect())
        .collect())
}

pub(crate) fn do_chunk_async<'p>(
    py: Python<'p>,
    chunker: Arc<dyn Chunker>,
    text: String,
) -> PyResult<Bound<'p, PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let chunks = tokio::task::spawn_blocking(move || chunker.chunk(&text))
            .await
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .map_err(to_py_err)?;
        Ok(chunks.into_iter().map(PyChunk::from).collect::<Vec<_>>())
    })
}

pub(crate) fn do_chunk_many_async<'p>(
    py: Python<'p>,
    chunker: Arc<dyn Chunker>,
    texts: Vec<String>,
) -> PyResult<Bound<'p, PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let results = tokio::task::spawn_blocking(move || chunk_many(&*chunker, &texts))
            .await
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .map_err(to_py_err)?;
        Ok(results
            .into_iter()
            .map(|cs| cs.into_iter().map(PyChunk::from).collect::<Vec<PyChunk>>())
            .collect::<Vec<_>>())
    })
}
