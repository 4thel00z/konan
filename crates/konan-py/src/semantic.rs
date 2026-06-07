use std::sync::Arc;

use async_trait::async_trait;
use konan_core::{Embedder, KonanError, OpenAIEmbedder, SemanticChunker};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::chunk::PyChunk;
use crate::errors::to_py_err;

/// Python-style display of an `Option`: `None` or the value's repr-ish form.
pub(crate) fn fmt_opt<T: std::fmt::Display>(opt: &Option<T>) -> String {
    match opt {
        Some(v) => v.to_string(),
        None => "None".to_string(),
    }
}

#[pyclass(name = "OpenAIEmbedder", module = "konan", frozen)]
#[derive(Clone)]
pub struct PyOpenAIEmbedder {
    pub(crate) inner: Arc<OpenAIEmbedder>,
    pub(crate) repr: String,
}

#[pymethods]
impl PyOpenAIEmbedder {
    #[new]
    #[pyo3(signature = (base_url, model, api_key=None, batch_size=128, timeout=30.0, max_retries=2, dimensions=None))]
    fn new(
        base_url: String,
        model: String,
        api_key: Option<String>,
        batch_size: usize,
        timeout: f64,
        max_retries: u32,
        dimensions: Option<u32>,
    ) -> PyResult<Self> {
        let repr = format!(
            "OpenAIEmbedder(base_url={base_url:?}, model={model:?}, api_key={}, \
             batch_size={batch_size}, timeout={timeout:?}, max_retries={max_retries}, \
             dimensions={})",
            if api_key.is_some() { "\"***\"" } else { "None" },
            fmt_opt(&dimensions),
        );
        Ok(Self {
            inner: Arc::new(
                OpenAIEmbedder::new(base_url, model, api_key, batch_size, timeout, max_retries, dimensions)
                    .map_err(to_py_err)?,
            ),
            repr,
        })
    }

    /// Embed texts directly — handy for verifying the endpoint.
    fn embed_async<'p>(&self, py: Python<'p>, texts: Vec<String>) -> PyResult<Bound<'p, PyAny>> {
        let inner = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner.embed(&texts).await.map_err(to_py_err)
        })
    }

    fn __repr__(&self) -> &str {
        &self.repr
    }
}

/// Adapter: a Python `async def (list[str]) -> list[list[float]]` callable
/// plugged into the core Embedder port. Async-only — sync chunk() cannot
/// drive a Python coroutine without a running asyncio loop.
struct PyCallableEmbedder {
    callable: Py<PyAny>,
}

#[async_trait]
impl Embedder for PyCallableEmbedder {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, KonanError> {
        let texts = texts.to_vec();
        let fut = Python::with_gil(|py| {
            let coro = self
                .callable
                .bind(py)
                .call1((texts,))
                .map_err(|e| KonanError::Embedding(e.to_string()))?;
            pyo3_async_runtimes::tokio::into_future(coro)
                .map_err(|e| KonanError::Embedding(e.to_string()))
        })?;
        let result = fut.await.map_err(|e| KonanError::Embedding(e.to_string()))?;
        Python::with_gil(|py| {
            result
                .bind(py)
                .extract::<Vec<Vec<f32>>>()
                .map_err(|e| KonanError::Embedding(e.to_string()))
        })
    }
}

#[pyclass(name = "SemanticChunker", module = "konan", frozen)]
pub struct PySemanticChunker {
    inner: Arc<SemanticChunker<Arc<dyn Embedder>>>,
    repr: String,
    /// Python-callable embedders need a running asyncio loop, so the sync
    /// chunk()/chunk_many() paths must be rejected with a clear error.
    embedder_is_py_callable: bool,
}

#[pymethods]
impl PySemanticChunker {
    #[new]
    #[pyo3(signature = (embedder, threshold=None, percentile=95.0, min_chunk_size=0, max_chunk_size=None))]
    fn new(
        embedder: Bound<'_, PyAny>,
        threshold: Option<f32>,
        percentile: f32,
        min_chunk_size: usize,
        max_chunk_size: Option<usize>,
    ) -> PyResult<Self> {
        let (port, embedder_repr, embedder_is_py_callable): (Arc<dyn Embedder>, String, bool) =
            if let Ok(native) = embedder.extract::<PyOpenAIEmbedder>() {
                let repr = native.repr.clone();
                (Arc::clone(&native.inner) as Arc<dyn Embedder>, repr, false)
            } else if embedder.is_callable() {
                (
                    Arc::new(PyCallableEmbedder { callable: embedder.unbind() }),
                    "<async callable>".to_string(),
                    true,
                )
            } else {
                return Err(PyValueError::new_err(
                    "embedder must be an OpenAIEmbedder or an async callable",
                ));
            };
        let repr = format!(
            "SemanticChunker(embedder={embedder_repr}, threshold={}, percentile={percentile:?}, \
             min_chunk_size={min_chunk_size}, max_chunk_size={})",
            match threshold {
                Some(t) => format!("{t:?}"),
                None => "None".to_string(),
            },
            fmt_opt(&max_chunk_size),
        );
        let inner = SemanticChunker::new(port, threshold, percentile, min_chunk_size, max_chunk_size)
            .map_err(to_py_err)?;
        Ok(Self { inner: Arc::new(inner), repr, embedder_is_py_callable })
    }

    fn __repr__(&self) -> &str {
        &self.repr
    }

    /// Python-callable embedders can only run inside an asyncio loop.
    fn reject_py_callable_in_sync(&self, method: &str) -> PyResult<()> {
        if self.embedder_is_py_callable {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "SemanticChunker.{method}() does not support Python async-callable \
                 embedders; use {method}_async() instead (or pass an OpenAIEmbedder)",
            )));
        }
        Ok(())
    }

    fn chunk(&self, py: Python<'_>, text: String) -> PyResult<Vec<PyChunk>> {
        self.reject_py_callable_in_sync("chunk")?;
        let inner = Arc::clone(&self.inner);
        let chunks = py
            .allow_threads(move || {
                pyo3_async_runtimes::tokio::get_runtime().block_on(inner.chunk(&text))
            })
            .map_err(to_py_err)?;
        Ok(chunks.into_iter().map(PyChunk::from).collect())
    }

    fn chunk_many(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<Vec<Vec<PyChunk>>> {
        self.reject_py_callable_in_sync("chunk_many")?;
        let inner = Arc::clone(&self.inner);
        let results = py
            .allow_threads(move || {
                pyo3_async_runtimes::tokio::get_runtime().block_on(inner.chunk_many(&texts))
            })
            .map_err(to_py_err)?;
        Ok(results
            .into_iter()
            .map(|cs| cs.into_iter().map(PyChunk::from).collect())
            .collect())
    }

    fn chunk_async<'p>(&self, py: Python<'p>, text: String) -> PyResult<Bound<'p, PyAny>> {
        let inner = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let chunks = inner.chunk(&text).await.map_err(to_py_err)?;
            Ok(chunks.into_iter().map(PyChunk::from).collect::<Vec<_>>())
        })
    }

    fn chunk_many_async<'p>(&self, py: Python<'p>, texts: Vec<String>) -> PyResult<Bound<'p, PyAny>> {
        let inner = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let results = inner.chunk_many(&texts).await.map_err(to_py_err)?;
            Ok(results
                .into_iter()
                .map(|cs| cs.into_iter().map(PyChunk::from).collect::<Vec<PyChunk>>())
                .collect::<Vec<_>>())
        })
    }
}
