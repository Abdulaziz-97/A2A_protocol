"""
GQESL A2A — Structured Logging

Ensures that cryptographic keys, exact natural language intents, and
raw semantic tensors are never leaked into standard application logs.
"""

from __future__ import annotations

import logging
from typing import Any

from gqesl_a2a import config

# We use a custom logger format that has been defined in config.py
logger = logging.getLogger("gqesl_a2a")

# Sensitive keys that should trigger redaction if present in kwargs
_SENSITIVE_KEYS = {
    "tensor",
    "intent_tensor",
    "codebook",
    "w1",
    "w2",
    "hmac_key",
    "shared_secret",
    "salt",
    "basis",
    "raw_intent",
    "messages",
    "keys",
    "session_keys"
}

def _redact_value(key: str, value: Any) -> Any:
    """Redact sensitive values based on key names or types."""
    key_lower = key.lower()
    
    # Check if the key matches known sensitive patterns
    if any(s in key_lower for s in _SENSITIVE_KEYS):
        if isinstance(value, bytes):
            return f"<REDACTED BYTES: len={len(value)}>"
        elif hasattr(value, "shape"): # numpy array
            return f"<REDACTED TENSOR: shape={value.shape}>"
        elif isinstance(value, list) and len(value) > 10 and isinstance(value[0], float):
            return f"<REDACTED VECTOR: len={len(value)}>"
        else:
            return "<REDACTED>"
            
    # For safe keys, still truncate huge arrays/bytes just in case
    if isinstance(value, bytes) and len(value) > 64:
        return f"{value[:16].hex()}... (len={len(value)})"
        
    return value


def safe_log(level: int, msg: str, **kwargs: Any) -> None:
    """
    Safely log a message, redacting any sensitive data passed in kwargs.
    
    Usage:
        safe_log(logging.INFO, "Encoded packet", idx=42, tensor=raw_tensor)
    """
    # Only process if this log level is active
    if not logger.isEnabledFor(level):
        return
        
    if not kwargs:
        logger.log(level, msg)
        return
        
    safe_kwargs = {k: _redact_value(k, v) for k, v in kwargs.items()}
    
    # Format kwargs as key=value string
    kw_str = " ".join(f"{k}={v}" for k, v in safe_kwargs.items())
    logger.log(level, f"{msg} | {kw_str}")

def log_info(msg: str, **kwargs: Any) -> None:
    safe_log(logging.INFO, msg, **kwargs)

def log_debug(msg: str, **kwargs: Any) -> None:
    safe_log(logging.DEBUG, msg, **kwargs)

def log_warning(msg: str, **kwargs: Any) -> None:
    safe_log(logging.WARNING, msg, **kwargs)

def log_error(msg: str, **kwargs: Any) -> None:
    safe_log(logging.ERROR, msg, **kwargs)
