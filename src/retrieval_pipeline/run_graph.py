"""Script: run the LangGraph routing pipeline.

Classifies each query as ml or general, then either
retrieves → generates (ml) or answers directly (general).

Usage
-----
    uv run python scripts/run_graph.py

Set PIPELINE_SOURCE in .env or the shell to point at your document.
"""

from __future__ import annotations

import os

from retrieval_pipeline import PipelineConfig, build_graph, run_pipeline
from retrieval_pipeline.logging_config import setup_logging

_SOURCE: str = os.getenv("PIPELINE_SOURCE", "dataset/data.pdf")

_QUERIES: list[dict[str, str]] = [
    {"question": "How does gradient descent work?"},        # ml path
    {"question": "What is the capital of France?"},         # general path
]


def main() -> None:
    """Build the graph and run both ML and general queries."""
    setup_logging()

    cfg = PipelineConfig()
    pipeline = run_pipeline(source=_SOURCE, queries=[], config=cfg)
    app = build_graph(pipeline, cfg)

    for query in _QUERIES:
        result = app.invoke(query)
        print(f"\nQuestion  : {query['question']}")
        print(f"Type      : {result['query_type']}")
        print(f"Answer    : {result['answer']}")


if __name__ == "__main__":
    main()