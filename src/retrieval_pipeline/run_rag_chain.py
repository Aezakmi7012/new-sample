"""Script: run the RAG chain directly (no routing).

Usage
-----
    uv run python scripts/run_rag_chain.py

Reads source path and query from environment / .env or falls back to
the hard-coded defaults below.
"""

from __future__ import annotations

import os

from retrieval_pipeline import PipelineConfig, build_rag_chain, run_pipeline
from retrieval_pipeline.logging_config import setup_logging

_SOURCE: str = os.getenv("PIPELINE_SOURCE", "dataset/data.pdf")
_QUERY: str = os.getenv("PIPELINE_QUERY", "What is the EBITDA margin?")


def main() -> None:
    """Load documents, retrieve chunks, and answer via Groq RAG chain."""
    setup_logging()

    cfg = PipelineConfig()
    pipeline = run_pipeline(source=_SOURCE, queries=[], config=cfg)

    docs = pipeline.compression_retriever.invoke(_QUERY)
    chain = build_rag_chain(cfg)
    answer = chain.invoke({"question": _QUERY, "context": docs})

    print(f"\nQuestion : {_QUERY}")
    print(f"Answer   : {answer}")


if __name__ == "__main__":
    main()