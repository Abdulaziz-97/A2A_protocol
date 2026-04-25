"""
GQESL A2A — Global Configuration

All thresholds, dimensions, and environment-configurable settings.
"""

import os
import logging

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.environ.get("GQESL_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

# ---------------------------------------------------------------------------
# Tensor Dimensions
# ---------------------------------------------------------------------------
TENSOR_DIM = 384                 # Dimension of all intent / concept vectors
CODEBOOK_SIZE = 4096             # Number of codebook entries (12-bit index)
BASIS_CONCEPTS = 256             # Max distinct concept dimensions in basis

# ---------------------------------------------------------------------------
# Cryptographic Parameters
# ---------------------------------------------------------------------------
HMAC_KEY_LEN = 32                # bytes
SALT_LEN = 32                    # bytes
SHARED_SECRET_LEN = 32           # bytes (X25519 output)

# HKDF info labels — each produces a distinct derived key from the same root
HKDF_INFO_W1 = b"gqesl-w1"
HKDF_INFO_W2 = b"gqesl-w2"
HKDF_INFO_CODEBOOK = b"gqesl-codebook"
HKDF_INFO_HMAC = b"gqesl-hmac"
HKDF_INFO_SALT = b"gqesl-salt"
HKDF_INFO_BASIS = b"gqesl-basis"

# ---------------------------------------------------------------------------
# RCC-8 Thresholds
# ---------------------------------------------------------------------------
EQ_THRESHOLD = 0.95              # Cosine similarity >= this → EQ (identical)
PO_THRESHOLD = 0.85              # >= this → PO (partial overlap)
EC_THRESHOLD = 0.60              # >= this → EC (externally connected)
# Below EC_THRESHOLD → DC (disconnected)

# ---------------------------------------------------------------------------
# Drift Monitor
# ---------------------------------------------------------------------------
DRIFT_CHECK_INTERVAL = 20        # Check drift every N messages
DRIFT_THRESHOLD = 0.15           # Cosine variance threshold for re-negotiation
DRIFT_HISTORY_K = 10             # Number of recent usage vectors to consider

# ---------------------------------------------------------------------------
# Key Rotation
# ---------------------------------------------------------------------------
KEY_ROTATION_INTERVAL = 500      # Messages per epoch (W1/W2/codebook/salt rotate)
SESSION_MAX_MESSAGES = 10_000    # Full session teardown limit (HMAC key rotates)

# ---------------------------------------------------------------------------
# DeepSeek LLM Configuration (all env-overridable)
# ---------------------------------------------------------------------------
DEEPSEEK_API_KEY = os.environ.get(
    "DEEPSEEK_API_KEY",
    "sk-e4aaba5c9bad4e67a751b35dc46efacc",
)
DEEPSEEK_BASE_URL = os.environ.get(
    "DEEPSEEK_BASE_URL",
    "https://api.deepseek.com",
)
DEEPSEEK_MODEL = os.environ.get(
    "DEEPSEEK_MODEL_NAME",
    "deepseek-v4-flash",
)
DEEPSEEK_MAX_RETRIES = int(os.environ.get("DEEPSEEK_MAX_RETRIES", "3"))
DEEPSEEK_RETRY_BACKOFF = float(os.environ.get("DEEPSEEK_RETRY_BACKOFF", "2.0"))

# ---------------------------------------------------------------------------
# Transport & Session Execution
# ---------------------------------------------------------------------------
MESSAGEBUS_QUEUE_MAXSIZE = 100
SESSION_TEARDOWN_BUFFER = 10     # Messages to allow post-termination signal
ARCTANH_CLIP = 0.9999            # Bound for stable inverse tanh

# ---------------------------------------------------------------------------
# Performance Targets (documentation constants, not enforced at runtime)
# ---------------------------------------------------------------------------
TARGET_ENCODE_LATENCY_MS = 10    # Encode-transmit-collapse cycle
TARGET_CODEBOOK_GEN_MS = 500     # Session startup codebook generation
TARGET_ECDH_LATENCY_MS = 100     # Key exchange
TARGET_LEDGER_SEARCH_MS = 5      # Nearest neighbor over 1000 entries
TARGET_RECOVERY_COSINE = 0.92    # Minimum roundtrip cosine similarity

# ---------------------------------------------------------------------------
# Wire Packet
# ---------------------------------------------------------------------------
WIRE_PACKET_SIZE = 38            # Expected packet size in bytes
