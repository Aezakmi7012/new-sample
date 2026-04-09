"""Document splitting and chunking module.

Provides :class:`DocumentSplitter` which routes documents to the correct
LangChain text splitter based on their detected type (Markdown vs. plain).
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
)
from loguru import logger

from retrieval_pipeline.config import CHUNK_OVERLAP, CHUNK_SIZE


class DocumentSplitter:
    """Split documents into chunks for downstream embedding.

    Markdown documents are split with a Markdown-aware splitter;
    all other documents use the recursive character splitter.

    Parameters
    ----------
    chunk_size : int
        Maximum character count per chunk.
    chunk_overlap : int
        Overlap in characters between consecutive chunks.
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> None:
        """Initialise the splitter with chunk size and overlap settings."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, docs: list[Document]) -> list[Document]:
        """Split *docs* into smaller chunks.

        Parameters
        ----------
        docs : list[Document]
            Source documents to split.

        Returns
        -------
        list[Document]
            All chunks produced from *docs*.
        """
        md_docs = [d for d in docs if d.metadata.get("source", "").endswith(".md")]
        other_docs = [d for d in docs if not d.metadata.get("source", "").endswith(".md")]

        chunks: list[Document] = []

        if md_docs:
            md_splitter = MarkdownTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            chunks.extend(md_splitter.split_documents(md_docs))

        if other_docs:
            generic_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            chunks.extend(generic_splitter.split_documents(other_docs))

        logger.info(f"Total chunks after splitting: {len(chunks)}")
        return chunks
