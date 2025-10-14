# log_config.py

import logging
from logging import Logger
from logging.handlers import RotatingFileHandler
import sys

def configure_logging(
    level: int = logging.INFO,
    log_file: str = None,
    max_bytes: int = 10*1024*1024,
    backup_count: int = 5
) -> Logger:
    """
    Configure the root logger:
      - Console handler at `level`
      - Optional rotating file handler at `level`
    Returns the root logger.
    
    Args:
        level:       Minimum logging level (DEBUG, INFO, etc).
        log_file:    Path to a logfile (if None, no file handler added).
        max_bytes:   Max size in bytes before rotating.
        backup_count:How many old logs to keep.
    """
    # Formatter for all handlers
    fmt = "%(asctime)s %(levelname)-8s [%(name)s:%(lineno)d] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    handlers = [console_handler]

    # Optional rotating file handler
    if log_file:
        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # Configure the root logger
    logging.basicConfig(level=level, handlers=handlers)

    logger = logging.getLogger()  # root logger
    logger.debug("Logging configured: level=%s, log_file=%s", logging.getLevelName(level), log_file)
    return logger
