"""LangGraph-based query routing and conditional RAG pipeline."""

from typing import Annotated

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import END, START, CompiledGraph, StateGraph
from loguru import logger
from pydantic import BaseModel, Field

from retrieval_pipeline.config import PipelineConfig
from retrieval_pipeline.llm_chain import _format_docs, build_rag_chain
from retrieval_pipeline.pipeline import RetrievalPipeline

# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class GraphState(BaseModel):
    """Shared mutable state passed between graph nodes."""

    model_config = {"arbitrary_types_allowed": True}

    question: Annotated[str, Field(min_length=1)]
    query_type: str = ""
    docs: list[Document] = Field(default_factory=list)
    answer: str = ""


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_graph(
    pipeline: RetrievalPipeline,
    config: PipelineConfig,
) -> CompiledGraph:
    """Construct and compile the LangGraph pipeline."""
    llm = ChatGroq(
        api_key=config.groq_api_key,
        model_name=config.groq_model,
    )

    classify_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Classify the question as 'ml' or 'general'. Return only one word."),
            ("human", "{question}"),
        ]
    )

    general_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("human", "{question}"),
        ]
    )

    rag_chain = build_rag_chain(config)

    # ---------------------------
    # Nodes
    # ---------------------------

    def classify_query(state: GraphState) -> GraphState:
        chain = classify_prompt | llm | StrOutputParser()
        result = chain.invoke({"question": state.question}).strip().lower()

        state.query_type = "ml" if "ml" in result else "general"

        logger.info("Query classified as: {}", state.query_type)
        return state

    def retrieve(state: GraphState) -> GraphState:
        state.docs = pipeline.compression_retriever.invoke(state.question)
        return state

    def generate(state: GraphState) -> GraphState:
        context = _format_docs(state.docs)
        state.answer = rag_chain.invoke({"question": state.question, "context": context})
        return state

    def answer_general(state: GraphState) -> GraphState:
        chain = general_prompt | llm | StrOutputParser()
        state.answer = chain.invoke({"question": state.question})
        return state

    # ---------------------------
    # Graph definition
    # ---------------------------

    graph = StateGraph(GraphState)

    graph.add_node("classify_query", classify_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("answer_general", answer_general)

    graph.add_edge(START, "classify_query")

    graph.add_conditional_edges(
        "classify_query",
        lambda state: state.query_type,
        {
            "ml": "retrieve",
            "general": "answer_general",
        },
    )

    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    graph.add_edge("answer_general", END)

    return graph.compile()
