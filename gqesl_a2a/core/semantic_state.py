"""
GQESL A2A — Semantic State & RCC-8 Engine

Implements the encode/decode pipeline and RCC-8 topological relation
computation.

CRITICAL DESIGN DECISIONS:
  - Decode uses W.T (transpose) not pseudo-inverse — matrices are
    QR-orthogonal so W @ W.T == I.
  - arctanh is clipped to [-0.999, 0.999] to avoid numerical infinity.
  - All tensor operations use float64 for numerical stability, converting
    to float32 only at wire boundaries.
  - Codebook quantisation uses L2 distance (not cosine) — random vectors
    in 384-D space are nearly orthogonal, so cosine gives poor results.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

import numpy as np
from pydantic import BaseModel, field_validator

from gqesl_a2a.config import (
    ARCTANH_CLIP,
    CODEBOOK_SIZE,
    EC_THRESHOLD,
    EQ_THRESHOLD,
    PO_THRESHOLD,
    TENSOR_DIM,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Wire Packet Model
# ═══════════════════════════════════════════════════════════════════════════

class SemanticMessage(BaseModel):
    """The ~41-byte wire packet.

    Fields:
        v:   Protocol version (1 byte uint8)
        idx: Codebook index (0–4095), transmitted as 2-byte uint16
        r:   RCC-8 relation code, 2 ASCII characters
        c:   Monotonic counter, 4-byte uint32
        h:   HMAC-SHA256 digest, 32 bytes
    Total: 1 + 2 + 2 + 4 + 32 = 41 bytes
    """

    v: int = 1
    idx: int
    r: str
    c: int
    h: bytes

    @field_validator("r")
    @classmethod
    def validate_relation(cls, v: str) -> str:
        if v not in ("EQ", "PO", "EC", "DC"):
            raise ValueError(f"Invalid RCC-8 relation: {v}")
        return v

    @field_validator("idx")
    @classmethod
    def validate_idx(cls, v: int) -> int:
        if not 0 <= v < CODEBOOK_SIZE:
            raise ValueError(f"Codebook index out of range: {v}")
        return v


def pack_packet(msg: SemanticMessage) -> bytes:
    """Pack SemanticMessage to compact 41-byte wire format."""
    import struct
    return (
        struct.pack(">B", msg.v)
        + struct.pack(">H", msg.idx)
        + msg.r.encode("ascii")
        + struct.pack(">I", msg.c)
        + msg.h
    )


def unpack_packet(data: bytes) -> SemanticMessage:
    """Unpack SemanticMessage from compact 41-byte wire format."""
    import struct
    v = struct.unpack(">B", data[:1])[0]
    idx = struct.unpack(">H", data[1:3])[0]
    r = data[3:5].decode("ascii")
    c = struct.unpack(">I", data[5:9])[0]
    h = data[9:41]
    return SemanticMessage(v=v, idx=idx, r=r, c=c, h=h)


# ═══════════════════════════════════════════════════════════════════════════
# RCC-8 Relation & Coordination Strategy
# ═══════════════════════════════════════════════════════════════════════════

class RCC8Relation(str, Enum):
    EQ = "EQ"
    PO = "PO"
    EC = "EC"
    DC = "DC"


class CoordinationStrategy(str, Enum):
    EXACT_MATCH = "EXACT_MATCH"
    SPLIT_EXECUTION = "SPLIT_EXECUTION"
    FULL_HANDOFF = "FULL_HANDOFF"
    NEGOTIATE_FIRST = "NEGOTIATE_FIRST"


RCC8_STRATEGY_MAP = {
    RCC8Relation.EQ: CoordinationStrategy.EXACT_MATCH,
    RCC8Relation.PO: CoordinationStrategy.SPLIT_EXECUTION,
    RCC8Relation.EC: CoordinationStrategy.FULL_HANDOFF,
    RCC8Relation.DC: CoordinationStrategy.NEGOTIATE_FIRST,
}


def compute_rcc8_relation(
    similarity: float,
    eq_threshold: float = EQ_THRESHOLD,
    po_threshold: float = PO_THRESHOLD,
    ec_threshold: float = EC_THRESHOLD,
) -> RCC8Relation:
    if similarity >= eq_threshold:
        return RCC8Relation.EQ
    elif similarity >= po_threshold:
        return RCC8Relation.PO
    elif similarity >= ec_threshold:
        return RCC8Relation.EC
    else:
        return RCC8Relation.DC


# ═══════════════════════════════════════════════════════════════════════════
# Cosine Utilities
# ═══════════════════════════════════════════════════════════════════════════

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    dot = float(np.dot(a, b))
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return dot / (na * nb)


def cosine_distance_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    q = query.astype(np.float64)
    m = matrix.astype(np.float64)
    query_norm = q / (np.linalg.norm(q) + 1e-12)
    mat_norms = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)
    return mat_norms @ query_norm


# ═══════════════════════════════════════════════════════════════════════════
# Encode Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def encode_tensor(
    intent_tensor: np.ndarray,
    salt: np.ndarray,
    W1: np.ndarray,
    W2: np.ndarray,
    codebook: np.ndarray,
    ledger_vectors: Optional[np.ndarray] = None,
) -> tuple[int, RCC8Relation, np.ndarray]:
    """Full encode pipeline: normalise → salt → project → RCC-8 → quantise.

    All operations in float64 for numerical stability.
    """
    # Upcast to float64
    t = intent_tensor.astype(np.float64)
    s = salt.astype(np.float64)
    w1 = W1.astype(np.float64)
    w2 = W2.astype(np.float64)
    cb = codebook.astype(np.float64)

    # 1. Normalise to [-0.5, 0.5] range to prevent tanh saturation
    norm = np.linalg.norm(t)
    if norm > 0:
        normalised = t / norm * 0.5
    else:
        normalised = t.copy()

    # 2. Salt injection
    salted = normalised + s

    # 3. Non-linear projection: tanh(salted @ W1) @ W2
    intermediate = np.tanh(salted @ w1)
    projected = intermediate @ w2

    # 4. RCC-8 relation against semantic ledger
    if ledger_vectors is not None and ledger_vectors.shape[0] > 0:
        lv = ledger_vectors.astype(np.float64)
        similarities = cosine_distance_matrix(projected.astype(np.float32), lv.astype(np.float32))
        max_sim = float(np.max(similarities))
        relation = compute_rcc8_relation(max_sim)
    else:
        relation = RCC8Relation.DC

    # 5. Codebook quantisation — L2 distance (not cosine)
    # Random vectors in 384-D have near-zero cosine similarity,
    # so L2 gives much better nearest-neighbour results.
    diffs = cb - projected[np.newaxis, :]
    l2_dists = np.linalg.norm(diffs, axis=1)
    idx = int(np.argmin(l2_dists))

    quant_error = float(l2_dists[idx])
    if quant_error > 5.0:
        logger.warning("High quantisation L2 error (%.3f)", quant_error)

    return idx, relation, projected.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Decode Pipeline  (Fix #1: uses W.T, not pseudo-inverse)
# ═══════════════════════════════════════════════════════════════════════════

def decode_tensor(
    idx: int,
    codebook: np.ndarray,
    salt: np.ndarray,
    W1: np.ndarray,
    W2: np.ndarray,
) -> np.ndarray:
    """Full decode pipeline: codebook lookup → reverse W2 → arctanh → reverse W1 → desalt.

    Because W1 and W2 are orthogonal (QR), their inverse is their transpose.
    All operations in float64 for stability.
    """
    cb = codebook.astype(np.float64)
    s = salt.astype(np.float64)
    w1 = W1.astype(np.float64)
    w2 = W2.astype(np.float64)

    # 1. Codebook lookup
    cb_vec = cb[idx].copy()

    # 2. Reverse W2: projected = intermediate @ W2 → intermediate = projected @ W2.T
    intermediate = cb_vec @ w2.T

    # 3. Inverse tanh with safe clipping
    clipped = np.clip(intermediate, -ARCTANH_CLIP, ARCTANH_CLIP)
    pre_tanh = np.arctanh(clipped)
    
    # 3.5. L2 normalise after arctanh to prevent scale explosion (Item 8)
    pt_norm = np.linalg.norm(pre_tanh)
    if pt_norm > 0:
        pre_tanh = pre_tanh / pt_norm * 0.5

    # 4. Reverse W1: pre_tanh = salted @ W1 → salted = pre_tanh @ W1.T
    salted = pre_tanh @ w1.T

    # 5. Remove salt
    normalised = salted - s

    # 6. Rescale back to unit vector
    norm = np.linalg.norm(normalised)
    if norm > 0:
        reconstructed = normalised / norm
    else:
        reconstructed = normalised

    return reconstructed.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Collapse
# ═══════════════════════════════════════════════════════════════════════════

def collapse_tensor(
    tensor: np.ndarray,
    basis_matrix: np.ndarray,
) -> tuple[int, float]:
    """Collapse a reconstructed tensor to the nearest concept.

    Returns (concept_index, cosine_similarity).
    """
    similarities = cosine_distance_matrix(tensor, basis_matrix)
    best_idx = int(np.argmax(similarities))
    best_sim = float(similarities[best_idx])
    return best_idx, best_sim
