"""Document loading module.

Provides :class:`DocumentLoader` which loads documents from a variety of
sources (files, directories, URLs, DataFrames, plain lists) into a uniform
``list[Document]`` representation consumed by the rest of the pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from langchain_community.document_loaders import (
    CSVLoader,
    JSONLoader,
    SQLDatabaseLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    WebBaseLoader,
)
from langchain_community.utilities import SQLDatabase
from langchain_core.documents import Document
from loguru import logger

#: File extensions supported by :meth:`DocumentLoader.load_directory`.
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".txt",
        ".md",
        ".pdf",
        ".csv",
        ".json",
        ".docx",
        ".doc",
        ".pptx",
        ".ppt",
        ".html",
        ".htm",
        ".db",
        ".sqlite",
        ".sqlite3",
    },
)


class DocumentLoader:
    """Load documents from files, directories, URLs, DataFrames, or lists.

    Parameters
    ----------
    json_jq_schema : str
        ``jq`` schema applied when loading ``.json`` files.
    sql_query : str
        SQL statement executed when loading SQLite database files.
    """

    def __init__(
        self,
        json_jq_schema: str = ".",
        sql_query: str = "SELECT * FROM documents",
    ) -> None:
        """Initialise the loader with optional JSON and SQL settings."""
        self.json_jq_schema = json_jq_schema
        self.sql_query = sql_query

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        source: str | pd.DataFrame | list,
    ) -> list[Document]:
        """Load documents from *source*.

        Parameters
        ----------
        source : str | pd.DataFrame | list
            One of:

            * ``str`` — a file path, directory path, or HTTP(S) URL.
            * ``pd.DataFrame`` — each row becomes one :class:`~langchain_core.documents.Document`.
            * ``list`` — each element becomes one document.

        Returns
        -------
        list[Document]
            Loaded documents ready for splitting.

        Raises
        ------
        FileNotFoundError
            If *source* is a path string that does not exist on disk.
        TypeError
            If *source* is not a recognised type.
        """
        if isinstance(source, pd.DataFrame):
            return self._load_dataframe(source)
        if isinstance(source, list):
            return self._load_list(source)
        if isinstance(source, str):
            return self._load_string(source)
        msg = f"Unsupported source type: {type(source)}"
        raise TypeError(msg)

    def load_directory(
        self,
        directory: str,
        extensions: list[str] | None = None,
    ) -> list[Document]:
        """Recursively load all supported files from *directory*.

        Parameters
        ----------
        directory : str
            Root directory to scan.
        extensions : list[str] | None
            Restrict loading to these extensions (e.g. ``[".pdf", ".csv"]``).
            When ``None``, all :data:`SUPPORTED_EXTENSIONS` are loaded.

        Returns
        -------
        list[Document]
            All documents found under *directory*.
        """
        allowed = SUPPORTED_EXTENSIONS
        if extensions is not None:
            allowed = frozenset(e.lower() for e in extensions)

        all_docs: list[Document] = []
        for file in Path(directory).rglob("*"):
            if file.suffix.lower() not in allowed:
                continue
            try:
                all_docs.extend(self.load(str(file)))
            except OSError as exc:
                logger.error(f"[SKIP] {file.name}: {exc}")

        logger.info(f"Total documents loaded from directory: {len(all_docs)}")
        return all_docs

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_dataframe(self, df: pd.DataFrame) -> list[Document]:
        """Convert every row of *df* into a :class:`~langchain_core.documents.Document`."""
        docs = [
            Document(
                page_content="\n".join(f"{col}: {val}" for col, val in row.items()),
                metadata={"row": i, "source": "dataframe"},
            )
            for i, row in df.iterrows()
        ]
        logger.info(f"[DataFrame] {len(docs)} rows loaded.")
        return docs

    def _load_list(self, items: list) -> list[Document]:
        """Convert each element of *items* into a :class:`~langchain_core.documents.Document`."""
        docs = [
            Document(
                page_content=str(item),
                metadata={"source": f"list[{i}]"},
            )
            for i, item in enumerate(items)
        ]
        logger.info(f"[List] {len(docs)} items loaded.")
        return docs

    def _load_string(self, source: str) -> list[Document]:
        """Dispatch a string *source* to the correct backend loader."""
        if source.startswith(("http://", "https://")):
            return self._load_url(source)
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Source not found: {source}")
        return self._load_file(path)

    def _load_url(self, url: str) -> list[Document]:
        """Load one or more pages from an HTTP(S) *url*."""
        docs = WebBaseLoader(url).load()
        logger.info(f"[URL] {len(docs)} page(s) loaded from {url}")
        return docs

    def _load_file(self, path: Path) -> list[Document]:
        """Dispatch *path* to the appropriate LangChain document loader."""
        ext = path.suffix.lower()
        docs: list[Document]

        if ext in (".txt", ".md"):
            docs = TextLoader(str(path), encoding="utf-8").load()
        elif ext == ".pdf":
            docs = self._load_pdf(path)
        elif ext == ".csv":
            docs = CSVLoader(str(path)).load()
        elif ext == ".json":
            docs = JSONLoader(
                file_path=str(path),
                jq_schema=self.json_jq_schema,
                text_content=False,
            ).load()
        elif ext in (".docx", ".doc"):
            docs = UnstructuredWordDocumentLoader(str(path)).load()
        elif ext in (".pptx", ".ppt"):
            docs = UnstructuredPowerPointLoader(str(path)).load()
        elif ext in (".html", ".htm"):
            docs = UnstructuredHTMLLoader(str(path)).load()
        elif ext in (".db", ".sqlite", ".sqlite3"):
            db = SQLDatabase.from_uri(f"sqlite:///{path}")
            docs = SQLDatabaseLoader(db=db, query=self.sql_query).load()
        else:
            logger.warning(f"Unknown extension '{ext}' - falling back to TextLoader.")
            docs = TextLoader(str(path), encoding="utf-8").load()

        logger.info(f"[{ext.upper()}] {len(docs)} document(s) loaded from '{path.name}'")
        return docs

    def _load_pdf(self, path: Path) -> list[Document]:
        """Load a PDF file via the Docling backend."""
        from langchain_docling import DoclingLoader  # — optional heavy dep

        logger.info(f"Loading PDF via Docling: {path}")
        return DoclingLoader(file_path=str(path)).load()
