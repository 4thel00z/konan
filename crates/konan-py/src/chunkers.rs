use std::path::PathBuf;
use std::sync::Arc;

use konan_core::{
    Chunker, Encoding, FixedSizeChunker, HfTokenizer, MarkdownChunker, NaiveChunker,
    RecursiveChunker, SentenceChunker, SizeUnit, TokenChunker,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyString;

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

/// The unit an `encoding` argument selects, with its repr: chars when absent,
/// else a built-in encoding name, a Hugging Face `tokenizers.Tokenizer`, or a
/// path to a `tokenizer.json`.
fn size_unit(encoding: Option<&Bound<'_, PyAny>>) -> PyResult<(SizeUnit, Option<String>)> {
    let Some(encoding) = encoding else {
        return Ok((SizeUnit::Chars, None));
    };
    if let Ok(name) = encoding.downcast::<PyString>() {
        let name = name.to_str()?;
        if name.ends_with(".json") {
            return Err(PyValueError::new_err(format!(
                "unknown encoding: {name}; pass tokenizer files as a pathlib.Path"
            )));
        }
        let unit = SizeUnit::Tokens(Encoding::from_name(name).map_err(to_py_err)?);
        return Ok((unit, Some(format!("{name:?}"))));
    }
    if encoding.hasattr("to_str")? {
        let json: String = encoding.call_method0("to_str")?.extract()?;
        let tokenizer = HfTokenizer::from_json(&json).map_err(to_py_err)?;
        let repr = "<tokenizers.Tokenizer>".to_string();
        return Ok((SizeUnit::Tokenizer(Arc::new(tokenizer)), Some(repr)));
    }
    if encoding.hasattr("__fspath__")? {
        let path: PathBuf = encoding.extract()?;
        let tokenizer = HfTokenizer::from_file(&path).map_err(to_py_err)?;
        let repr = encoding.repr()?.to_string();
        return Ok((SizeUnit::Tokenizer(Arc::new(tokenizer)), Some(repr)));
    }
    Err(PyTypeError::new_err(
        "encoding must be an encoding name, a tokenizers.Tokenizer, or a path to a tokenizer.json",
    ))
}

/// Repr suffix for the opt-in size options, empty at their defaults so
/// existing reprs are unchanged.
fn options_repr(encoding: Option<&str>, stable_boundaries: bool) -> String {
    let mut out = String::new();
    if let Some(encoding) = encoding {
        out.push_str(&format!(", encoding={encoding}"));
    }
    if stable_boundaries {
        out.push_str(", stable_boundaries=True");
    }
    out
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
    #[pyo3(signature = (
        chunk_size=1000,
        chunk_overlap=200,
        respect_sentences=true,
        encoding=None,
        stable_boundaries=false,
    ))]
    fn new(
        chunk_size: usize,
        chunk_overlap: usize,
        respect_sentences: bool,
        encoding: Option<&Bound<'_, PyAny>>,
        stable_boundaries: bool,
    ) -> PyResult<Self> {
        let (unit, encoding_repr) = size_unit(encoding)?;
        Ok(Self {
            inner: Arc::new(
                FixedSizeChunker::new(chunk_size, chunk_overlap, respect_sentences)
                    .map_err(to_py_err)?
                    .with_size_unit(unit)
                    .with_stable_boundaries(stable_boundaries),
            ),
            repr: format!(
                "FixedSizeChunker(chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, \
                 respect_sentences={}{})",
                py_bool(respect_sentences),
                options_repr(encoding_repr.as_deref(), stable_boundaries)
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
    #[pyo3(signature = (
        chunk_size=1000,
        chunk_overlap=200,
        separators=None,
        encoding=None,
        stable_boundaries=false,
    ))]
    fn new(
        chunk_size: usize,
        chunk_overlap: usize,
        separators: Option<Vec<String>>,
        encoding: Option<&Bound<'_, PyAny>>,
        stable_boundaries: bool,
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
        let (unit, encoding_repr) = size_unit(encoding)?;
        Ok(Self {
            inner: Arc::new(
                RecursiveChunker::new(chunk_size, chunk_overlap, separators)
                    .map_err(to_py_err)?
                    .with_size_unit(unit)
                    .with_stable_boundaries(stable_boundaries),
            ),
            repr: format!(
                "RecursiveChunker(chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, \
                 separators={separators_repr}{})",
                options_repr(encoding_repr.as_deref(), stable_boundaries)
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
    #[pyo3(signature = (max_chars=1000, overlap_sentences=1, encoding=None))]
    fn new(
        max_chars: usize,
        overlap_sentences: usize,
        encoding: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let (unit, encoding_repr) = size_unit(encoding)?;
        Ok(Self {
            inner: Arc::new(
                SentenceChunker::new(max_chars, overlap_sentences)
                    .map_err(to_py_err)?
                    .with_size_unit(unit),
            ),
            repr: format!(
                "SentenceChunker(max_chars={max_chars}, overlap_sentences={overlap_sentences}{})",
                options_repr(encoding_repr.as_deref(), false)
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
    #[pyo3(signature = (
        chunk_size=1000,
        chunk_overlap=200,
        encoding=None,
        stable_boundaries=false,
    ))]
    fn new(
        chunk_size: usize,
        chunk_overlap: usize,
        encoding: Option<&Bound<'_, PyAny>>,
        stable_boundaries: bool,
    ) -> PyResult<Self> {
        let (unit, encoding_repr) = size_unit(encoding)?;
        Ok(Self {
            inner: Arc::new(
                MarkdownChunker::new(chunk_size, chunk_overlap)
                    .map_err(to_py_err)?
                    .with_size_unit(unit)
                    .with_stable_boundaries(stable_boundaries),
            ),
            repr: format!(
                "MarkdownChunker(chunk_size={chunk_size}, chunk_overlap={chunk_overlap}{})",
                options_repr(encoding_repr.as_deref(), stable_boundaries)
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
