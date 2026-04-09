"""Logging configuration for the retrieval pipeline.

Call :func:`setup_logging` once at application startup. All modules that do
``from loguru import logger`` automatically inherit both sinks (console +
rotating file) without any further changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logging(
    log_dir: str = "logs",
    log_file: str = "pipeline.log",
    level: str = "DEBUG",
    rotation: str = "10 MB",
    retention: str = "7 days",
    compression: str = "zip",
    colorize_console: bool = True,
) -> None:
    """Configure loguru with a console sink and a rotating file sink.

    Parameters
    ----------
    log_dir : str
        Directory where log files are stored (created automatically).
    log_file : str
        Base filename for the log file inside *log_dir*.
    level : str
        Minimum severity written to **both** sinks.
    rotation : str
        When to rotate the log file (size string, time string, or callable).
    retention : str
        How long rotated files are kept before deletion.
    compression : str
        Archive format applied to rotated files (``"zip"``, ``"gz"`` etc.).
    colorize_console : bool
        Whether to emit ANSI colour codes on stderr.
    """
    # Remove the default loguru handler so we control format & sinks fully.
    logger.remove()

    # Console sink
    console_fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    logger.add(
        sys.stderr,
        format=console_fmt,
        level=level,
        colorize=colorize_console,
        enqueue=True,
    )

    # Rotating file sink
    log_path = Path(log_dir) / log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_fmt = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"
    logger.add(
        str(log_path),
        format=file_fmt,
        level=level,
        rotation=rotation,
        retention=retention,
        compression=compression,
        enqueue=True,
        encoding="utf-8",
    )

    logger.info(f"Logging initialised - file sink: {log_path.resolve()}")
