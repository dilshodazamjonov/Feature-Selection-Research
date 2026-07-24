"""
Logging configuration for the project.
"""
import logging
import sys
from contextlib import contextmanager
from pathlib import Path


class _SuppressDuringResearchSession(logging.Filter):
    """Avoid duplicate plain console lines while canonical JSONL logging is active."""

    def filter(self, record: logging.LogRecord) -> bool:
        del record
        try:
            from credit_risk_fs.experiments.research_logging import (
                research_logging_active,
            )

            return not research_logging_active()
        except ImportError:
            return True


def setup_logging(
    name: str = "research",
    level: int = logging.INFO,
    log_file: str = None,
    format_string: str = None
) -> logging.Logger:
    """
    Setup and return a configured logger.
    
    Parameters
    ----------
    name : str
        Logger name
    level : int
        Logging level (default: INFO)
    log_file : str, optional
        Path to log file. If provided, logs will be written to both console and file.
    format_string : str, optional
        Custom format string
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    if format_string is None:
        format_string = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(_SuppressDuringResearchSession())
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "research") -> logging.Logger:
    """
    Get an existing logger or create a default one.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logging(name)
    return logger


@contextmanager
def run_log_context(log_file: str | Path, level: int = logging.INFO):
    """
    Temporarily mirror all project logs into one run-level file.

    Existing modules create named loggers at import time, while selectors may
    import later during a run. Attaching to the root logger captures both cases
    through normal propagation without permanently leaking handlers across
    matrix entries.
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setFormatter(formatter)
    handler.setLevel(level)

    root_logger = logging.getLogger()
    previous_level = root_logger.level
    root_logger.setLevel(min(previous_level, level) if previous_level else level)
    root_logger.addHandler(handler)

    try:
        yield log_path
    finally:
        root_logger.removeHandler(handler)
        root_logger.setLevel(previous_level)
        handler.close()
