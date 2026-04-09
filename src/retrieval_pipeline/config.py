"""Configuration settings for the retrieval pipeline.

Values are loaded from a ``.env`` file (or real environment variables) using
``python-dotenv``.  The search order is:

1. A ``.env`` file in the current working directory (or any parent up to the
   project root, found via :func:`dotenv.find_dotenv`).
2. Variables already present in the shell environment (these take priority
   when ``override=False``, which is the default).

Instantiate :class:`PipelineConfig` once and pass it through the pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

# ---------------------------------------------------------------------------
# Load .env once at import time.
# ``find_dotenv`` walks up from cwd until it finds a .env file or hits the
# filesystem root.  ``load_dotenv`` silently does nothing if no file is found,
# so this is always safe to call.
# ---------------------------------------------------------------------------
_env_file: str = find_dotenv(usecwd=True)
load_dotenv(dotenv_path=_env_file or None, override=False)


def _env_path() -> str:
    """Return the resolved path of the loaded .env file, or '<none>' if absent."""
    return str(Path(_env_file).resolve()) if _env_file else "<none>"


@dataclass
class PipelineConfig:
    """Centralised, environment-driven configuration for the pipeline.

    All fields fall back to a hard-coded default if neither the ``.env`` file
    nor the shell environment defines the corresponding variable.

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
    """

    chunk_size: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", "300")),
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "50")),
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
    )
    reranker_model: str = field(
        default_factory=lambda: os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base"),
    )
    chroma_dir: str = field(
        default_factory=lambda: os.getenv("CHROMA_DIR", "./chroma_store"),
    )
    collection_name: str = field(
        default_factory=lambda: os.getenv("COLLECTION_NAME", "generic_docs"),
    )
    top_k: int = field(
        default_factory=lambda: int(os.getenv("TOP_K", "6")),
    )
    top_n: int = field(
        default_factory=lambda: int(os.getenv("TOP_N", "3")),
    )
    device: str = field(
        default_factory=lambda: os.getenv("DEVICE", "cpu"),
    )


# ---------------------------------------------------------------------------
# Module-level constants — kept for backwards compatibility with any code that
# imports them directly (e.g. ``from retrieval_pipeline.config import TOP_K``).
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
