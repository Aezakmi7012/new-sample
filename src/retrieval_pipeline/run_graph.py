"""Script: run the LangGraph routing pipeline."""

from __future__ import annotations

import os

from loguru import logger

from retrieval_pipeline.config import PipelineConfig
from retrieval_pipeline.graph import build_graph
from retrieval_pipeline.logging_config import setup_logging
from retrieval_pipeline.pipeline import run_pipeline

_SOURCE: str = os.getenv("PIPELINE_SOURCE", "dataset/data.pdf")

_QUERIES: list[dict[str, str]] = [
    {"question": "How does gradient descent work?"},
    {"question": "What is the capital of France?"},
]


def main() -> None:
    """Build the graph and run both ML and general queries."""
    setup_logging()

    cfg = PipelineConfig()
    pipeline = run_pipeline(source=_SOURCE, queries=[], config=cfg)
    app = build_graph(pipeline, cfg)

    for query in _QUERIES:
        result = app.invoke(query)

        logger.info("\nQuestion  : {}", query["question"])
        logger.info("Type      : {}", result["query_type"])
        logger.info("Answer    : {}", result["answer"])


if __name__ == "__main__":
    main()
