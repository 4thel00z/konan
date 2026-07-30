use pyo3::prelude::*;

#[pyclass(name = "TextMetrics", module = "konan", frozen, eq)]
#[derive(PartialEq)]
pub(crate) struct PyTextMetrics {
    #[pyo3(get)]
    content_characters: usize,
    #[pyo3(get)]
    alphanumeric_characters: usize,
    #[pyo3(get)]
    exceeded_limit: bool,
}

impl From<konan_core::TextMetrics> for PyTextMetrics {
    fn from(metrics: konan_core::TextMetrics) -> Self {
        Self {
            content_characters: metrics.content_characters,
            alphanumeric_characters: metrics.alphanumeric_characters,
            exceeded_limit: metrics.exceeded_limit,
        }
    }
}

#[pymethods]
impl PyTextMetrics {
    fn __repr__(&self) -> String {
        format!(
            "TextMetrics(content_characters={}, alphanumeric_characters={}, exceeded_limit={})",
            self.content_characters,
            self.alphanumeric_characters,
            if self.exceeded_limit { "True" } else { "False" }
        )
    }
}

#[pyfunction]
#[pyo3(signature = (text, max_source_characters))]
pub(crate) fn measure_text(text: &str, max_source_characters: usize) -> PyTextMetrics {
    konan_core::measure_text(text, max_source_characters).into()
}
