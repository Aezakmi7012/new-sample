"""Main orchestrator for the retrieval and reranking pipeline."""

from __future__ import annotations

import sys

from langchain_community.vectorstores.utils import filter_complex_metadata
from loguru import logger

from retrieval_pipeline.config import PipelineConfig
from retrieval_pipeline.display import ResultsDisplay
from retrieval_pipeline.loaders import DocumentLoader
from retrieval_pipeline.logging_config import setup_logging
from retrieval_pipeline.pipeline import RetrievalPipeline
from retrieval_pipeline.splitters import DocumentSplitter
from retrieval_pipeline.vectorstore import VectorStoreBuilder


def run_pipeline(
    source: object,
    queries: list[str],
    config: PipelineConfig | None = None,
    is_directory: bool = False,
    extensions: list[str] | None = None,
    json_jq_schema: str = ".",
    sql_query: str = "SELECT * FROM documents",
    show: str = "both",
) -> RetrievalPipeline:
    """Execute the end-to-end document loading, chunking, and querying process."""
    cfg = config or PipelineConfig()

    logger.info("Step 1: Loading documents")
    loader = DocumentLoader(json_jq_schema=json_jq_schema, sql_query=sql_query)
    docs = (
        loader.load_directory(source, extensions=extensions)
        if is_directory
        else loader.load(source)
    )

    logger.info("Step 2: Splitting")
    splitter = DocumentSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )
    chunks = splitter.split(docs)

    logger.info("Step 3: Embedding & vector store")
    clean_chunks = filter_complex_metadata(chunks)
    vs_builder = VectorStoreBuilder(
        embedding_model=cfg.embedding_model,
        chroma_dir=cfg.chroma_dir,
        collection_name=cfg.collection_name,
        device=cfg.device,
    )
    vectorstore, _ = vs_builder.build(clean_chunks)

    logger.info("Step 4: Building retrieval pipeline")
    pipeline = RetrievalPipeline(
        vectorstore=vectorstore,
        reranker_model=cfg.reranker_model,
        top_k=cfg.top_k,
        top_n=cfg.top_n,
        device=cfg.device,
    )

    logger.info("Step 5: Querying")
    display = ResultsDisplay()
    for q in queries:
        if show == "retriever":
            display.show_retriever(
                q,
                pipeline.base_retriever,
                pipeline.cross_encoder,
                cfg.top_k,
            )
        elif show == "reranker":
            display.show_reranker(
                q,
                pipeline.compression_retriever,
                pipeline.cross_encoder,
                cfg.top_n,
            )
        else:
            display.compare(
                q,
                pipeline.base_retriever,
                pipeline.compression_retriever,
                pipeline.cross_encoder,
                cfg.top_k,
                cfg.top_n,
            )

    return pipeline


if __name__ == "__main__":
    setup_logging()
    logger.info("Pipeline ready.")

    mode = sys.argv[1] if len(sys.argv) > 1 else "retriever"

    if mode == "chain":
        from retrieval_pipeline.llm_chain import answer as llm_answer

        pipeline = run_pipeline(
            source="dataset/data.pdf",
            queries=[],
        )

        docs = pipeline.compression_retriever.invoke("What is the EBITDA margin?")

        logger.info(
            "Answer: {}",
            llm_answer("What is the EBITDA margin?", docs),
        )

    elif mode == "graph":
        from retrieval_pipeline.graph import build_graph

        cfg = PipelineConfig()

        pipeline = run_pipeline(
            source="/workspaces/new-sample/dataset/samplee.pdf",
            queries=[],
            config=cfg,
        )

        app = build_graph(pipeline, cfg)

        queries = ["Types of Machine Learning?", "Hi"]

        for q in queries:
            result = app.invoke({"question": q})

            logger.info("\nQuestion: {}", q)
            logger.info("Type    : {}", result["query_type"])
            logger.info("Answer  : {}", result["answer"])

    else:
        run_pipeline(
            source="dataset/data.pdf",
            queries=["What is the invisible man name"],
        )
