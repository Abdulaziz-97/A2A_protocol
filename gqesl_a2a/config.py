"""
GQESL A2A — Global Configuration

All thresholds, dimensions, and environment-configurable settings.
"""

import os
import logging

# logging
LOG_LEVEL = os.environ.get("GQESL_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

# tensor
TENSOR_DIM = 384
CODEBOOK_SIZE = 4096
BASIS_CONCEPTS = 256

# crypto
HMAC_KEY_LEN = 32
SALT_LEN = 32
SHARED_SECRET_LEN = 32

# HKDF labels
HKDF_INFO_W1 = b"gqesl-w1"
HKDF_INFO_W2 = b"gqesl-w2"
HKDF_INFO_CODEBOOK = b"gqesl-codebook"
HKDF_INFO_HMAC = b"gqesl-hmac"
HKDF_INFO_SALT = b"gqesl-salt"
HKDF_INFO_BASIS = b"gqesl-basis"

# RCC-8
EQ_THRESHOLD = 0.95
PO_THRESHOLD = 0.85
EC_THRESHOLD = 0.60

# auto resolve
AUTO_RESOLVE_THRESHOLD = float(os.environ.get("GQESL_AUTO_RESOLVE_THRESHOLD", "0.75"))

# drift
DRIFT_CHECK_INTERVAL = 20
DRIFT_THRESHOLD = 0.15
DRIFT_HISTORY_K = 10

# rotation
KEY_ROTATION_INTERVAL = 500
SESSION_MAX_MESSAGES = 10_000

# llm
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

# transport
MESSAGEBUS_QUEUE_MAXSIZE = 100
SESSION_TEARDOWN_BUFFER = 10
ARCTANH_CLIP = 0.9999

# targets
TARGET_ENCODE_LATENCY_MS = 10
TARGET_CODEBOOK_GEN_MS = 500
TARGET_ECDH_LATENCY_MS = 100
TARGET_LEDGER_SEARCH_MS = 5
TARGET_RECOVERY_COSINE = 0.92

# wire
WIRE_PACKET_SIZE = 38
