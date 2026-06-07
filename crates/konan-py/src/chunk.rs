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
        &self.inner.text
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
    #[getter]
    fn hash(&self) -> &str {
        &self.inner.hash
    }
    fn __len__(&self) -> usize {
        self.inner.text.chars().count()
    }
    fn __repr__(&self) -> String {
        let mut preview: String = self.inner.text.chars().take(40).collect();
        if self.inner.text.chars().count() > 40 {
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
