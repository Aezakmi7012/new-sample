"""Retrieval pipeline package.

Provides modular components for document loading, splitting,
vector storage, retrieval, and LangGraph-based query routing.
"""

from retrieval_pipeline.config import PipelineConfig
from retrieval_pipeline.display import ResultsDisplay
from retrieval_pipeline.llm_chain import answer, build_rag_chain
from retrieval_pipeline.loaders import DocumentLoader
from retrieval_pipeline.logging_config import setup_logging
from retrieval_pipeline.pipeline import RetrievalPipeline
from retrieval_pipeline.splitters import DocumentSplitter
from retrieval_pipeline.vectorstore import VectorStoreBuilder


def __getattr__(name: str) -> object:
    """Lazy-load graph symbols to avoid circular imports."""
    if name in ("build_graph", "GraphState"):
        from retrieval_pipeline import graph as _graph

        return getattr(_graph, name)

    raise AttributeError(f"module 'retrieval_pipeline' has no attribute {name!r}")


__all__ = [
    "DocumentLoader",
    "DocumentSplitter",
    "GraphState",
    "PipelineConfig",
    "ResultsDisplay",
    "RetrievalPipeline",
    "VectorStoreBuilder",
    "answer",
    "build_graph",
    "build_rag_chain",
    "setup_logging",
]
