use pyo3::prelude::*;

#[pyclass(name = "Chunk", module = "konan", frozen)]
#[derive(Clone)]
pub struct PyChunk {
    pub(crate) inner: konan_core::Chunk,
}

impl From<konan_core::Chunk> for PyChunk {
    fn from(inner: konan_core::Chunk) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyChunk {
    #[getter]
    fn text(&self) -> &str {
        self.inner.text()
    }
    #[getter]
    fn start(&self) -> usize {
        self.inner.start
    }
    #[getter]
    fn end(&self) -> usize {
        self.inner.end
    }
    #[getter]
    fn index(&self) -> usize {
        self.inner.index
    }
    /// Hex digest, as before the core switched to storing the raw u64 — the
    /// formatting just happens on access now instead of for every chunk.
    #[getter]
    fn hash(&self) -> String {
        self.inner.hash_hex()
    }
    /// The raw xxh3-64 digest, for callers who would only parse `hash` back.
    #[getter]
    fn hash_int(&self) -> u64 {
        self.inner.hash
    }
    fn __len__(&self) -> usize {
        self.inner.text().chars().count()
    }
    fn __repr__(&self) -> String {
        let text = self.inner.text();
        let mut preview: String = text.chars().take(40).collect();
        if text.chars().count() > 40 {
            preview.push('…');
        }
        format!(
            "Chunk(index={}, start={}, end={}, text={:?})",
            self.inner.index, self.inner.start, self.inner.end, preview
        )
    }
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
    fn __hash__(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        self.inner.hash(&mut h);
        h.finish()
    }
}
