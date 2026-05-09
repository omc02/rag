"""
Utility helper functions for the RAG pipeline
"""

from pathlib import Path
from typing import List, Dict, Any
import hashlib
import time
from functools import wraps


def get_project_root() -> Path:
    """Get the project root directory"""
    current = Path(__file__).parent
    
    # Go up until we find a directory with config/ or data/
    while current != current.parent:
        if (current / "config").exists() or (current / "data").exists():
            return current
        current = current.parent
    
    # Fallback to current working directory
    return Path.cwd()


def create_stable_id(text: str, prefix: str = "") -> str:
    """Create a stable ID from text content"""
    content_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}{content_hash}" if prefix else content_hash


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / (1024**2):.1f} MB"
    else:
        return f"{size_bytes / (1024**3):.1f} GB"


def retry_with_backoff(max_retries: int = 3, backoff_factor: float = 2.0):
    """Decorator for retrying functions with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:  # Last attempt
                        raise e
                    
                    wait_time = backoff_factor ** attempt
                    time.sleep(wait_time)
                    
            return None
        return wrapper
    return decorator


def validate_api_keys(required_keys: List[str]) -> Dict[str, bool]:
    """Validate that required API keys are present in environment"""
    import os
    
    results = {}
    for key in required_keys:
        value = os.getenv(key)
        results[key] = bool(value and value.strip())
    
    return results


def chunks_to_stats(chunks: List[Any]) -> Dict[str, Any]:
    """Calculate statistics for document chunks"""
    if not chunks:
        return {"total_chunks": 0}
    
    lengths = [len(chunk.page_content) for chunk in chunks]
    sources = set()
    
    for chunk in chunks:
        source = chunk.metadata.get("source_file", "unknown")
        sources.add(source)
    
    return {
        "total_chunks": len(chunks),
        "average_length": sum(lengths) / len(lengths) if lengths else 0,
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "total_length": sum(lengths),
        "unique_sources": len(sources),
        "sources": list(sources)
    }


def format_duration(seconds: float) -> str:
    """Format duration in human readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe filesystem usage"""
    import re
    
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    filename = re.sub(r'\s+', '_', filename)  # Replace spaces with underscores
    filename = filename.strip('._')  # Remove leading/trailing dots and underscores
    
    return filename[:255]  # Limit length


def deep_merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """Deep merge two dictionaries"""
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result