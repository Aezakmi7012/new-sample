"""LangGraph-based query routing and conditional RAG pipeline.

Graph topology
--------------
::

    [classify_query]
          |
    ──────┴──────────────────
    |                       |
  "ml"                 "general"
    |                       |
 [retrieve]           [answer_general]
    |
 [generate]

Nodes
~~~~~
* **classify_query** — classifies the incoming prompt as
  ``"ml"`` or ``"general"`` using the Groq LLM.
* **retrieve** — runs the full reranking retriever (ML path only).
* **generate** — calls the RAG chain with retrieved chunks.
* **answer_general** — answers directly without retrieval.

Usage example
-------------
>>> from retrieval_pipeline.config import PipelineConfig
>>> from retrieval_pipeline.graph import build_graph
>>> from retrieval_pipeline.pipeline import RetrievalPipeline
>>> app = build_graph(pipeline, config)
>>> result = app.invoke({"question": "What is gradient descent?"})
>>> print(result["answer"])
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Annotated

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from loguru import logger
from pydantic import BaseModel, Field

from retrieval_pipeline.config import PipelineConfig
from retrieval_pipeline.llm_chain import _format_docs, build_rag_chain
from retrieval_pipeline.pipeline import RetrievalPipeline

if TYPE_CHECKING:
    from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class GraphState(BaseModel):
    """Shared mutable state passed between graph nodes.

    Pydantic validates every field on assignment, so invalid state
    is caught immediately rather than surfacing deep inside a node.

    Parameters
    ----------
    question : str
        The original user question.
    query_type : str
        Classification result — ``"ml"`` or ``"general"``.
        Empty string before classification runs.
    docs : list[Document]
        Retrieved document chunks (populated only on the ML path).
    answer : str
        Final answer string produced by whichever terminal node runs.
        Empty string until a terminal node writes it.
    """

    model_config = {"arbitrary_types_allowed": True}

    question: Annotated[str, Field(min_length=1, description="User question.")]
    query_type: str = Field(
        default="",
        description="'ml' or 'general'; set by classify_query node.",
    )
    docs: list[Document] = Field(
        default_factory=list,
        description="Retrieved chunks; populated only on the ML path.",
    )
    answer: str = Field(
        default="",
        description="Final LLM answer; populated by a terminal node.",
    )


# ---------------------------------------------------------------------------
# Classifier prompt
# ---------------------------------------------------------------------------

_CLASSIFY_SYSTEM: str = (
    "You are a query classifier. Classify the user's question as EXACTLY one "
    "of two categories:\n"
    "  - ml       (questions about machine learning, deep learning, neural "
    "networks, NLP, computer vision, reinforcement learning, model training, "
    "optimisation algorithms, datasets, evaluation metrics, transformers, "
    "embeddings, or any related AI/ML topic)\n"
    "  - general  (everything else)\n\n"
    "Reply with ONLY the single word: ml  OR  general. "
    "No punctuation, no explanation."
)

_CLASSIFY_HUMAN: str = "{question}"


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------


def _make_classifier(cfg: PipelineConfig) -> ChatGroq:
    """Return a lightweight Groq LLM configured for classification.

    Parameters
    ----------
    cfg : PipelineConfig
        Pipeline configuration.

    Returns
    -------
    ChatGroq
        A Groq LLM instance with minimal token budget (single-word reply).
    """
    return ChatGroq(
        model=cfg.groq_model,
        api_key=cfg.groq_api_key,
        max_tokens=5,
        temperature=0.0,
    )


def classify_query(state: GraphState, *, cfg: PipelineConfig) -> GraphState:
    """Classify the question as ``'ml'`` or ``'general'``.

    Parameters
    ----------
    state : GraphState
        Current graph state; must contain ``"question"``.
    cfg : PipelineConfig
        Pipeline configuration (injected via partial).

    Returns
    -------
    GraphState
        State updated with ``"query_type"``.
    """
    question = state.question

    prompt = ChatPromptTemplate.from_messages(
        [("system", _CLASSIFY_SYSTEM), ("human", _CLASSIFY_HUMAN)]
    )
    llm = _make_classifier(cfg)
    chain = prompt | llm | StrOutputParser()

    raw: str = chain.invoke({"question": question}).strip().lower()
    # Normalise — anything that is not "ml" falls back to "general"
    query_type = "ml" if "ml" in raw else "general"

    logger.info(
        f"classify_query: '{question[:60]}' → {query_type}",
    )
    return GraphState(**{**state.model_dump(), "query_type": query_type})


def retrieve(
    state: GraphState,
    *,
    pipeline: RetrievalPipeline,
) -> GraphState:
    """Retrieve relevant chunks using the full reranking retriever.

    Only reached on the ``ml`` path.

    Parameters
    ----------
    state : GraphState
        Current state; must contain ``"question"``.
    pipeline : RetrievalPipeline
        The assembled retrieval pipeline (bi-encoder + cross-encoder reranker).

    Returns
    -------
    GraphState
        State updated with ``"docs"``.
    """
    question = state.question
    docs: list[Document] = pipeline.compression_retriever.invoke(question)
    logger.info(
        f"retrieve: fetched {len(docs)} chunk(s) for '{question[:60]}'",
    )
    return GraphState(**{**state.model_dump(), "docs": docs})


def generate(state: GraphState, *, cfg: PipelineConfig) -> GraphState:
    """Generate an answer from retrieved chunks via the RAG chain.

    Parameters
    ----------
    state : GraphState
        Current state; must contain ``"question"`` and ``"docs"``.
    cfg : PipelineConfig
        Pipeline configuration.

    Returns
    -------
    GraphState
        State updated with ``"answer"``.
    """
    chain = build_rag_chain(cfg)
    context_str = _format_docs(state.docs)
    answer_text: str = chain.invoke(
        {"question": state.question, "context": context_str},
    )
    logger.info("generate: answer produced")
    return GraphState(**{**state.model_dump(), "answer": answer_text})


def answer_general(state: GraphState, *, cfg: PipelineConfig) -> GraphState:
    """Answer a general (non-ML) question directly without retrieval.

    Parameters
    ----------
    state : GraphState
        Current state; must contain ``"question"``.
    cfg : PipelineConfig
        Pipeline configuration.

    Returns
    -------
    GraphState
        State updated with ``"answer"``.
    """
    _GENERAL_SYSTEM = (
        "You are a helpful assistant. Answer the user's question clearly and "
        "concisely."
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", _GENERAL_SYSTEM), ("human", "{question}")]
    )
    llm = ChatGroq(
        model=cfg.groq_model,
        api_key=cfg.groq_api_key,
        max_tokens=cfg.llm_max_tokens,
        temperature=cfg.llm_temperature,
    )
    chain = prompt | llm | StrOutputParser()
    answer_text: str = chain.invoke({"question": state.question})
    logger.info("answer_general: direct answer produced")
    return GraphState(**{**state.model_dump(), "answer": answer_text})


# ---------------------------------------------------------------------------
# Routing edge
# ---------------------------------------------------------------------------


def _route_by_type(state: GraphState) -> str:
    """Return the next node name based on the classified query type.

    Parameters
    ----------
    state : GraphState
        Must contain ``"query_type"``.

    Returns
    -------
    str
        ``"retrieve"`` for ML queries, ``"answer_general"`` otherwise.
    """
    return "retrieve" if state.query_type == "ml" else "answer_general"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_graph(
    pipeline: RetrievalPipeline,
    config: PipelineConfig | None = None,
) -> StateGraph:
    """Compile and return the LangGraph application.

    Parameters
    ----------
    pipeline : RetrievalPipeline
        Fully built retrieval pipeline (bi-encoder + cross-encoder reranker).
    config : PipelineConfig | None
        Optional config; defaults to env-driven :class:`PipelineConfig`.

    Returns
    -------
    StateGraph
        A compiled LangGraph ``CompiledGraph`` ready for ``.invoke()``.
    """
    cfg = config or PipelineConfig()

    classify_node = partial(classify_query, cfg=cfg)
    retrieve_node = partial(retrieve, pipeline=pipeline)
    generate_node = partial(generate, cfg=cfg)
    general_node = partial(answer_general, cfg=cfg)

    graph = StateGraph(GraphState)

    graph.add_node("classify_query", classify_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("answer_general", general_node)

    graph.add_edge(START, "classify_query")
    graph.add_conditional_edges(
        "classify_query",
        _route_by_type,
        {
            "retrieve": "retrieve",
            "answer_general": "answer_general",
        },
    )
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    graph.add_edge("answer_general", END)

    logger.info(
        "LangGraph compiled: classify_query → [retrieve → generate | answer_general]",
    )
    return graph.compile()