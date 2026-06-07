use std::sync::Arc;

use konan_core::{
    Chunker, FixedSizeChunker, MarkdownChunker, NaiveChunker, RecursiveChunker, SentenceChunker,
    TokenChunker,
};
use pyo3::prelude::*;

use crate::chunk::PyChunk;
use crate::common::{do_chunk, do_chunk_async, do_chunk_many, do_chunk_many_async};
use crate::errors::to_py_err;

/// Python-style bool literal for reprs.
fn py_bool(b: bool) -> &'static str {
    if b {
        "True"
    } else {
        "False"
    }
}

#[pyclass(name = "NaiveChunker", module = "konan", frozen)]
pub struct PyNaiveChunker {
    inner: Arc<dyn Chunker>,
    repr: String,
}

#[pymethods]
impl PyNaiveChunker {
    #[new]
    #[pyo3(signature = (chunk_size=200))]
    fn new(chunk_size: usize) -> PyResult<Self> {
        Ok(Self {
            inner: Arc::new(NaiveChunker::new(chunk_size).map_err(to_py_err)?),
            repr: format!("NaiveChunker(chunk_size={chunk_size})"),
        })
    }
    fn __repr__(&self) -> &str {
        &self.repr
    }
    fn chunk(&self, py: Python<'_>, text: String) -> PyResult<Vec<PyChunk>> {
        do_chunk(py, Arc::clone(&self.inner), text)
    }
    fn chunk_many(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<Vec<Vec<PyChunk>>> {
        do_chunk_many(py, Arc::clone(&self.inner), texts)
    }
    fn chunk_async<'p>(&self, py: Python<'p>, text: String) -> PyResult<Bound<'p, PyAny>> {
        do_chunk_async(py, Arc::clone(&self.inner), text)
    }
    fn chunk_many_async<'p>(
        &self,
        py: Python<'p>,
        texts: Vec<String>,
    ) -> PyResult<Bound<'p, PyAny>> {
        do_chunk_many_async(py, Arc::clone(&self.inner), texts)
    }
}

#[pyclass(name = "FixedSizeChunker", module = "konan", frozen)]
pub struct PyFixedSizeChunker {
    inner: Arc<dyn Chunker>,
    repr: String,
}

#[pymethods]
impl PyFixedSizeChunker {
    #[new]
    #[pyo3(signature = (chunk_size=1000, chunk_overlap=200, respect_sentences=true))]
    fn new(chunk_size: usize, chunk_overlap: usize, respect_sentences: bool) -> PyResult<Self> {
        Ok(Self {
            inner: Arc::new(
                FixedSizeChunker::new(chunk_size, chunk_overlap, respect_sentences)
                    .map_err(to_py_err)?,
            ),
            repr: format!(
                "FixedSizeChunker(chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, \
                 respect_sentences={})",
                py_bool(respect_sentences)
            ),
        })
    }
    fn __repr__(&self) -> &str {
        &self.repr
    }
    fn chunk(&self, py: Python<'_>, text: String) -> PyResult<Vec<PyChunk>> {
        do_chunk(py, Arc::clone(&self.inner), text)
    }
    fn chunk_many(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<Vec<Vec<PyChunk>>> {
        do_chunk_many(py, Arc::clone(&self.inner), texts)
    }
    fn chunk_async<'p>(&self, py: Python<'p>, text: String) -> PyResult<Bound<'p, PyAny>> {
        do_chunk_async(py, Arc::clone(&self.inner), text)
    }
    fn chunk_many_async<'p>(
        &self,
        py: Python<'p>,
        texts: Vec<String>,
    ) -> PyResult<Bound<'p, PyAny>> {
        do_chunk_many_async(py, Arc::clone(&self.inner), texts)
    }
}

#[pyclass(name = "RecursiveChunker", module = "konan", frozen)]
pub struct PyRecursiveChunker {
    inner: Arc<dyn Chunker>,
    repr: String,
}

#[pymethods]
impl PyRecursiveChunker {
    #[new]
    #[pyo3(signature = (chunk_size=1000, chunk_overlap=200, separators=None))]
    fn new(
        chunk_size: usize,
        chunk_overlap: usize,
        separators: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let separators_repr = match &separators {
            None => "None".to_string(),
            Some(seps) => format!(
                "[{}]",
                seps.iter()
                    .map(|s| format!("{s:?}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        };
        Ok(Self {
            inner: Arc::new(
                RecursiveChunker::new(chunk_size, chunk_overlap, separators).map_err(to_py_err)?,
            ),
            repr: format!(
                "RecursiveChunker(chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, \
                 separators={separators_repr})"
            ),
        })
    }
    fn __repr__(&self) -> &str {
        &self.repr
    }
    fn chunk(&self, py: Python<'_>, text: String) -> PyResult<Vec<PyChunk>> {
        do_chunk(py, Arc::clone(&self.inner), text)
    }
    fn chunk_many(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<Vec<Vec<PyChunk>>> {
        do_chunk_many(py, Arc::clone(&self.inner), texts)
    }
    fn chunk_async<'p>(&self, py: Python<'p>, text: String) -> PyResult<Bound<'p, PyAny>> {
        do_chunk_async(py, Arc::clone(&self.inner), text)
    }
    fn chunk_many_async<'p>(
        &self,
        py: Python<'p>,
        texts: Vec<String>,
    ) -> PyResult<Bound<'p, PyAny>> {
        do_chunk_many_async(py, Arc::clone(&self.inner), texts)
    }
}

#[pyclass(name = "SentenceChunker", module = "konan", frozen)]
pub struct PySentenceChunker {
    inner: Arc<dyn Chunker>,
    repr: String,
}

#[pymethods]
impl PySentenceChunker {
    #[new]
    #[pyo3(signature = (max_chars=1000, overlap_sentences=1))]
    fn new(max_chars: usize, overlap_sentences: usize) -> PyResult<Self> {
        Ok(Self {
            inner: Arc::new(SentenceChunker::new(max_chars, overlap_sentences).map_err(to_py_err)?),
            repr: format!(
                "SentenceChunker(max_chars={max_chars}, overlap_sentences={overlap_sentences})"
            ),
        })
    }
    fn __repr__(&self) -> &str {
        &self.repr
    }
    fn chunk(&self, py: Python<'_>, text: String) -> PyResult<Vec<PyChunk>> {
        do_chunk(py, Arc::clone(&self.inner), text)
    }
    fn chunk_many(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<Vec<Vec<PyChunk>>> {
        do_chunk_many(py, Arc::clone(&self.inner), texts)
    }
    fn chunk_async<'p>(&self, py: Python<'p>, text: String) -> PyResult<Bound<'p, PyAny>> {
        do_chunk_async(py, Arc::clone(&self.inner), text)
    }
    fn chunk_many_async<'p>(
        &self,
        py: Python<'p>,
        texts: Vec<String>,
    ) -> PyResult<Bound<'p, PyAny>> {
        do_chunk_many_async(py, Arc::clone(&self.inner), texts)
    }
}

#[pyclass(name = "MarkdownChunker", module = "konan", frozen)]
pub struct PyMarkdownChunker {
    inner: Arc<dyn Chunker>,
    repr: String,
}

#[pymethods]
impl PyMarkdownChunker {
    #[new]
    #[pyo3(signature = (chunk_size=1000, chunk_overlap=200))]
    fn new(chunk_size: usize, chunk_overlap: usize) -> PyResult<Self> {
        Ok(Self {
            inner: Arc::new(MarkdownChunker::new(chunk_size, chunk_overlap).map_err(to_py_err)?),
            repr: format!(
                "MarkdownChunker(chunk_size={chunk_size}, chunk_overlap={chunk_overlap})"
            ),
        })
    }
    fn __repr__(&self) -> &str {
        &self.repr
    }
    fn chunk(&self, py: Python<'_>, text: String) -> PyResult<Vec<PyChunk>> {
        do_chunk(py, Arc::clone(&self.inner), text)
    }
    fn chunk_many(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<Vec<Vec<PyChunk>>> {
        do_chunk_many(py, Arc::clone(&self.inner), texts)
    }
    fn chunk_async<'p>(&self, py: Python<'p>, text: String) -> PyResult<Bound<'p, PyAny>> {
        do_chunk_async(py, Arc::clone(&self.inner), text)
    }
    fn chunk_many_async<'p>(
        &self,
        py: Python<'p>,
        texts: Vec<String>,
    ) -> PyResult<Bound<'p, PyAny>> {
        do_chunk_many_async(py, Arc::clone(&self.inner), texts)
    }
}

#[pyclass(name = "TokenChunker", module = "konan", frozen)]
pub struct PyTokenChunker {
    inner: Arc<dyn Chunker>,
    repr: String,
}

#[pymethods]
impl PyTokenChunker {
    #[new]
    #[pyo3(signature = (chunk_size=512, chunk_overlap=64, encoding="cl100k_base"))]
    fn new(chunk_size: usize, chunk_overlap: usize, encoding: &str) -> PyResult<Self> {
        Ok(Self {
            inner: Arc::new(
                TokenChunker::new(chunk_size, chunk_overlap, encoding).map_err(to_py_err)?,
            ),
            repr: format!(
                "TokenChunker(chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, \
                 encoding={encoding:?})"
            ),
        })
    }
    fn __repr__(&self) -> &str {
        &self.repr
    }
    fn chunk(&self, py: Python<'_>, text: String) -> PyResult<Vec<PyChunk>> {
        do_chunk(py, Arc::clone(&self.inner), text)
    }
    fn chunk_many(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<Vec<Vec<PyChunk>>> {
        do_chunk_many(py, Arc::clone(&self.inner), texts)
    }
    fn chunk_async<'p>(&self, py: Python<'p>, text: String) -> PyResult<Bound<'p, PyAny>> {
        do_chunk_async(py, Arc::clone(&self.inner), text)
    }
    fn chunk_many_async<'p>(
        &self,
        py: Python<'p>,
        texts: Vec<String>,
    ) -> PyResult<Bound<'p, PyAny>> {
        do_chunk_many_async(py, Arc::clone(&self.inner), texts)
    }
}
