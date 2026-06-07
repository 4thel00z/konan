use pyo3::prelude::*;

mod chunk;
mod chunkers;
mod common;
mod errors;
mod semantic;

#[pymodule]
fn _konan(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<chunk::PyChunk>()?;
    m.add_class::<chunkers::PyNaiveChunker>()?;
    m.add_class::<chunkers::PyFixedSizeChunker>()?;
    m.add_class::<chunkers::PyRecursiveChunker>()?;
    m.add_class::<chunkers::PySentenceChunker>()?;
    m.add_class::<chunkers::PyMarkdownChunker>()?;
    m.add_class::<chunkers::PyTokenChunker>()?;
    m.add("EmbeddingError", py.get_type::<errors::EmbeddingError>())?;
    m.add_class::<semantic::PyOpenAIEmbedder>()?;
    m.add_class::<semantic::PySemanticChunker>()?;
    Ok(())
}
