"""LLM chain module using Groq as the provider.

Build a RAG chain via the pipe (``|``) operator:

    prompt | llm | output_parser

The chain is assembled in :func:`build_rag_chain` and returns a plain
``str`` answer given a *question* and a list of retrieved
:class:`~langchain_core.documents.Document` objects.

Usage example
-------------
>>> from retrieval_pipeline.config import PipelineConfig
>>> from retrieval_pipeline.llm_chain import build_rag_chain
>>> chain = build_rag_chain(PipelineConfig())
>>> answer = chain.invoke({"question": "What is EBITDA?", "context": docs})
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq
from loguru import logger

from retrieval_pipeline.config import PipelineConfig

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_RAG_SYSTEM_PROMPT: str = (
    "You are a helpful assistant. Answer the user's question using ONLY the "
    "provided context excerpts. If the context does not contain enough "
    "information to answer, say so clearly — do not fabricate facts.\n\n"
    "Context:\n{context}"
)

_RAG_HUMAN_PROMPT: str = "{question}"


def _format_docs(docs: list[Document]) -> str:
    """Concatenate document page content into a single context string.

    Parameters
    ----------
    docs : list[Document]
        Retrieved document chunks.

    Returns
    -------
    str
        Newline-separated page contents.
    """
    return "\n\n---\n\n".join(d.page_content for d in docs)


def build_rag_chain(config: PipelineConfig | None = None) -> Runnable:
    """Build and return the RAG chain using the pipe operator.

    Chain shape::

        ChatPromptTemplate | ChatGroq | StrOutputParser

    Parameters
    ----------
    config : PipelineConfig | None
        Pipeline configuration. If ``None``, a default instance is used.

    Returns
    -------
    Runnable
        A LangChain runnable that accepts:
        - "question": str
        - "context": str

    Raises
    ------
    ValueError
        If GROQ_API_KEY is not set.
    """
    cfg = config or PipelineConfig()

    if not cfg.groq_api_key:
        msg = (
            "GROQ_API_KEY is not set. "
            "Add it to your .env file or export it as an environment variable."
        )
        raise ValueError(msg)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _RAG_SYSTEM_PROMPT),
            ("human", _RAG_HUMAN_PROMPT),
        ]
    )

    llm = ChatGroq(
        model=cfg.groq_model,
        api_key=cfg.groq_api_key,
        max_tokens=cfg.llm_max_tokens,
        temperature=cfg.llm_temperature,
    )

    chain: Runnable = prompt | llm | StrOutputParser()

    logger.info(
        "RAG chain built: prompt | ChatGroq(%s) | StrOutputParser",
        cfg.groq_model,
    )

    return chain


def answer(
    question: str,
    docs: list[Document],
    config: PipelineConfig | None = None,
) -> str:
    """Retrieve documents and return an answer in one call.

    Parameters
    ----------
    question : str
        The user's question.
    docs : list[Document]
        Pre-retrieved document chunks.
    config : PipelineConfig | None
        Optional config override.

    Returns
    -------
    str
        The LLM-generated answer.
    """
    chain = build_rag_chain(config)
    context_str = _format_docs(docs)
    return chain.invoke({"question": question, "context": context_str})
