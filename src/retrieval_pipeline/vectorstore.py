"""Vector store management module.

Provides :class:`VectorStoreBuilder` which wraps Chroma and HuggingFace
embeddings into a single buildable unit.
"""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger

from retrieval_pipeline.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DEVICE,
    EMBEDDING_MODEL,
)


class VectorStoreBuilder:
    """Build or reload a Chroma vector store from document chunks.

    Parameters
    ----------
    embedding_model : str
        HuggingFace model identifier for the bi-encoder.
    chroma_dir : str
        Filesystem path where Chroma persists its data.
    collection_name : str
        Name of the Chroma collection to create or reuse.
    device : str
        PyTorch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
    """

    def __init__(
        self,
        embedding_model: str = EMBEDDING_MODEL,
        chroma_dir: str = CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
        device: str = DEVICE,
    ) -> None:
        """Initialise the builder with model and storage settings."""
        self.embedding_model = embedding_model
        self.chroma_dir = chroma_dir
        self.collection_name = collection_name
        self.device = device

    def build(self, chunks: list[Document]) -> tuple[Chroma, HuggingFaceEmbeddings]:
        """Embed *chunks* and store them in the Chroma vector database.

        Parameters
        ----------
        chunks : list[Document]
            Pre-split document chunks to embed.

        Returns
        -------
        tuple[Chroma, HuggingFaceEmbeddings]
            The populated vector store and the embedding model instance.
        """
        embeddings = self._build_embeddings()
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=self.collection_name,
            persist_directory=self.chroma_dir,
        )
        count = vectorstore.get()["ids"]
        logger.info(f"Vector store ready: {len(count)} vectors stored.")
        return vectorstore, embeddings

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_embeddings(self) -> HuggingFaceEmbeddings:
        """Instantiate and log the HuggingFace embedding model."""
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
        )
        dim = len(embeddings.embed_query("test"))
        logger.info(f"Embedding model: {self.embedding_model} | dim: {dim}")
        return embeddings
