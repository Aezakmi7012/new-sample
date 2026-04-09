"""Main orchestrator for the retrieval and reranking pipeline.

Entry point that wires together all pipeline components:
:class:`~retrieval_pipeline.loaders.DocumentLoader`,
:class:`~retrieval_pipeline.splitters.DocumentSplitter`,
:class:`~retrieval_pipeline.vectorstore.VectorStoreBuilder`,
:class:`~retrieval_pipeline.pipeline.RetrievalPipeline`, and
:class:`~retrieval_pipeline.display.ResultsDisplay`.
"""

from __future__ import annotations

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
    """Execute the end-to-end document loading, chunking, and querying process.

    Parameters
    ----------
    source : object
        A file path, directory path, HTTP(S) URL, ``pd.DataFrame``, or list.
    queries : list[str]
        One or more search queries to run against the pipeline.
    config : PipelineConfig | None
        Pipeline configuration. Defaults to :class:`~retrieval_pipeline.config.PipelineConfig`
        with values read from environment variables.
    is_directory : bool
        When ``True``, *source* is treated as a directory and scanned
        recursively.
    extensions : list[str] | None
        File extensions to include when *is_directory* is ``True``.
        ``None`` loads all supported extensions.
    json_jq_schema : str
        ``jq`` schema applied to ``.json`` source files.
    sql_query : str
        SQL statement executed against ``.db`` / ``.sqlite`` source files.
    show : str
        Controls which results are displayed:
        ``"retriever"``, ``"reranker"``, or ``"both"``.

    Returns
    -------
    RetrievalPipeline
        The assembled pipeline (retriever + reranker + cross-encoder).
    """
    cfg = config or PipelineConfig()

    # Step 1 - Load
    logger.info("Step 1: Loading documents")
    loader = DocumentLoader(json_jq_schema=json_jq_schema, sql_query=sql_query)
    docs = (
        loader.load_directory(source, extensions=extensions)
        if is_directory
        else loader.load(source)
    )

    # Step 2 - Split
    logger.info("Step 2: Splitting")
    splitter = DocumentSplitter(chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)
    chunks = splitter.split(docs)

    # Step 3 - Embed & store
    logger.info("Step 3: Embedding & vector store")
    clean_chunks = filter_complex_metadata(chunks)
    vs_builder = VectorStoreBuilder(
        embedding_model=cfg.embedding_model,
        chroma_dir=cfg.chroma_dir,
        collection_name=cfg.collection_name,
        device=cfg.device,
    )
    vectorstore, _ = vs_builder.build(clean_chunks)

    # Step 4 - Build retrieval pipeline
    logger.info("Step 4: Building retrieval pipeline")
    pipeline = RetrievalPipeline(
        vectorstore=vectorstore,
        reranker_model=cfg.reranker_model,
        top_k=cfg.top_k,
        top_n=cfg.top_n,
        device=cfg.device,
    )

    # Step 5 - Query
    logger.info("Step 5: Querying")
    display = ResultsDisplay()
    for q in queries:
        if show == "retriever":
            display.show_retriever(q, pipeline.base_retriever, pipeline.cross_encoder, cfg.top_k)
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
    run_pipeline(
        source="/workspaces/new-sample/dataset/data.pdf", queries=["What is the invicible man name"]
    )
