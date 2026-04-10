"""Configuration settings for the retrieval pipeline.

Values are loaded from a ``.env`` file (or real environment variables) via
:class:`pydantic_settings.BaseSettings`.  The search order is:

1. Variables already present in the shell environment (highest priority).
2. A ``.env`` file in the current working directory.

Instantiate :class:`PipelineConfig` once and pass it through the pipeline.
"""

from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PipelineConfig(BaseSettings):
    """Centralised, environment-driven configuration for the pipeline.

    All fields fall back to a hard-coded default if neither the ``.env``
    file nor the shell environment defines the corresponding variable.

    Parameters
    ----------
    chunk_size : int
        Maximum token/character count per document chunk.
    chunk_overlap : int
        Number of overlapping characters between consecutive chunks.
    embedding_model : str
        HuggingFace model identifier used for bi-encoder embeddings.
    reranker_model : str
        HuggingFace model identifier used for cross-encoder reranking.
    chroma_dir : str
        Filesystem path where the Chroma vector store is persisted.
    collection_name : str
        Name of the Chroma collection to create or reuse.
    top_k : int
        Number of candidates retrieved by the bi-encoder.
    top_n : int
        Number of results kept after cross-encoder reranking.
    device : str
        PyTorch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
    groq_api_key : str
        Groq API key for LLM inference.
    groq_model : str
        Groq model identifier (e.g. ``"llama3-8b-8192"``).
    llm_max_tokens : int
        Maximum tokens the LLM may generate per response.
    llm_temperature : float
        Sampling temperature for the LLM (0.0 = deterministic).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Chunking ─────────────────────────────────────────────────────────
    chunk_size: int = Field(default=300, gt=0, description="Max chars per chunk.")
    chunk_overlap: int = Field(default=50, ge=0, description="Overlap between chunks.")

    # ── Models ───────────────────────────────────────────────────────────
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        min_length=1,
        description="HuggingFace bi-encoder model ID.",
    )
    reranker_model: str = Field(
        default="BAAI/bge-reranker-base",
        min_length=1,
        description="HuggingFace cross-encoder model ID.",
    )

    # ── Vector store ─────────────────────────────────────────────────────
    chroma_dir: str = Field(
        default="./chroma_store",
        min_length=1,
        description="Chroma persistence directory.",
    )
    collection_name: str = Field(
        default="generic_docs",
        min_length=1,
        description="Chroma collection name.",
    )

    # ── Retrieval ────────────────────────────────────────────────────────
    top_k: int = Field(default=6, gt=0, description="Bi-encoder candidate count.")
    top_n: int = Field(default=3, gt=0, description="Post-rerank result count.")

    # ── Hardware ─────────────────────────────────────────────────────────
    device: str = Field(default="cpu", description="PyTorch device string.")

    # ── LLM (Groq) ───────────────────────────────────────────────────────
    groq_api_key: str = Field(default="", description="Groq API key.")
    groq_model: str = Field(
        default="llama3-8b-8192",
        min_length=1,
        description="Groq model identifier.",
    )
    llm_max_tokens: int = Field(
        default=512,
        gt=0,
        description="Max tokens for LLM generation.",
    )
    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="LLM sampling temperature.",
    )

    # ── Cross-field validators ────────────────────────────────────────────

    @field_validator("top_n")
    @classmethod
    def top_n_lte_top_k(cls, v: int, info: object) -> int:
        """Ensure ``top_n`` does not exceed ``top_k``.

        Parameters
        ----------
        v : int
            The candidate value for ``top_n``.
        info : object
            Pydantic validation info carrying already-validated field data.

        Returns
        -------
        int
            The validated value.

        Raises
        ------
        ValueError
            If ``top_n > top_k``.
        """
        top_k = getattr(info, "data", {}).get("top_k")
        if top_k is not None and v > top_k:
            msg = f"top_n ({v}) must be <= top_k ({top_k})"
            raise ValueError(msg)
        return v

    @field_validator("device")
    @classmethod
    def device_is_valid(cls, v: str) -> str:
        """Validate that *device* is a recognised PyTorch device string.

        Parameters
        ----------
        v : str
            The candidate device string.

        Returns
        -------
        str
            The validated, lower-cased device string.

        Raises
        ------
        ValueError
            If *v* is not ``cpu``, ``cuda``, ``mps``, or ``cuda:<n>``.
        """
        lower = v.lower()
        valid = lower == "cpu" or lower == "mps" or lower.startswith("cuda")
        if not valid:
            msg = f"device must be 'cpu', 'mps', or 'cuda[:N]', got '{v}'"
            raise ValueError(msg)
        return lower


# ---------------------------------------------------------------------------
# Module-level constants — backwards compatibility.
# ---------------------------------------------------------------------------
_defaults = PipelineConfig()

CHUNK_SIZE: int = _defaults.chunk_size
CHUNK_OVERLAP: int = _defaults.chunk_overlap
EMBEDDING_MODEL: str = _defaults.embedding_model
RERANKER_MODEL: str = _defaults.reranker_model
CHROMA_DIR: str = _defaults.chroma_dir
COLLECTION_NAME: str = _defaults.collection_name
TOP_K: int = _defaults.top_k
TOP_N: int = _defaults.top_n
DEVICE: str = _defaults.device
GROQ_API_KEY: str = _defaults.groq_api_key
GROQ_MODEL: str = _defaults.groq_model
LLM_MAX_TOKENS: int = _defaults.llm_max_tokens
LLM_TEMPERATURE: float = _defaults.llm_temperature