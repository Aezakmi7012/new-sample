"""Retrieval and reranking pipeline definition module.

Provides :class:`RetrievalPipeline` which assembles a bi-encoder retriever
and a cross-encoder reranker on top of a Chroma vector store.
"""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from loguru import logger

from retrieval_pipeline.config import DEVICE, RERANKER_MODEL, TOP_K, TOP_N


class RetrievalPipeline:
    """Bi-encoder retriever combined with a cross-encoder reranker.

    Parameters
    ----------
    vectorstore : Chroma
        Populated Chroma vector store to search over.
    reranker_model : str
        HuggingFace model identifier for the cross-encoder.
    top_k : int
        Number of candidates the bi-encoder retrieves.
    top_n : int
        Number of results retained after cross-encoder reranking.
    device : str
        PyTorch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
    """

    def __init__(
        self,
        vectorstore: Chroma,
        reranker_model: str = RERANKER_MODEL,
        top_k: int = TOP_K,
        top_n: int = TOP_N,
        device: str = DEVICE,
    ) -> None:
        """Build the retrieval pipeline from *vectorstore*."""
        self.top_k = top_k
        self.top_n = top_n

        self.base_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k},
        )

        self.cross_encoder = HuggingFaceCrossEncoder(
            model_name=reranker_model,
            model_kwargs={"device": device},
        )

        compressor = CrossEncoderReranker(model=self.cross_encoder, top_n=top_n)
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.base_retriever,
        )

        logger.info(
            f"Pipeline: bi-encoder top-{top_k} -> cross-encoder rerank -> top-{top_n}",
        )
