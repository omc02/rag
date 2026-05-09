"""
Logging utilities for the RAG pipeline
Provides structured logging configuration
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from ..config import RAGConfig


def setup_logger(
    name: str = "rag_pipeline",
    config: Optional[RAGConfig] = None,
    level: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with the specified configuration
    
    Args:
        name: Logger name
        config: RAG configuration object
        level: Override log level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Don't add handlers if they already exist
    if logger.handlers:
        return logger
    
    # Set level
    if level:
        log_level = getattr(logging, level.upper(), logging.INFO)
    elif config:
        log_level = getattr(logging, config.logging.level.upper(), logging.INFO)
    else:
        log_level = logging.INFO
    
    logger.setLevel(log_level)
    
    # Create formatter
    if config:
        formatter = logging.Formatter(config.logging.format)
    else:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if enabled
    if config and config.logging.file_enabled:
        log_file = Path(config.logging.file_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "rag_pipeline") -> logging.Logger:
    """Get an existing logger or create a basic one"""
    return logging.getLogger(name)