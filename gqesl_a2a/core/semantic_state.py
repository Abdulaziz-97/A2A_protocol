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
from collections import OrderedDict
from enum import Enum
from typing import Any, Optional

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


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    dot = float(np.dot(a, b))
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return dot / (na * nb)


def vector_stats(arr: np.ndarray) -> dict[str, Any]:
    """Compact summary for dashboards (JSON-serialisable)."""
    a = np.asarray(arr, dtype=np.float64)
    flat = a.ravel()
    if flat.size == 0:
        return {"shape": [], "norm": 0.0, "min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "shape": list(a.shape),
        "norm": float(np.linalg.norm(flat)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
    }


def cosine_distance_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    q = query.astype(np.float64)
    m = matrix.astype(np.float64)
    query_norm = q / (np.linalg.norm(q) + 1e-12)
    mat_norms = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)
    return mat_norms @ query_norm


def encode_tensor(
    intent_tensor: np.ndarray,
    salt: np.ndarray,
    W1: np.ndarray,
    W2: np.ndarray,
    codebook: np.ndarray,
    ledger_vectors: Optional[np.ndarray] = None,
    trace: Optional[list[dict[str, Any]]] = None,
) -> tuple[int, RCC8Relation, np.ndarray]:
    """Full encode pipeline: normalise → salt → project → RCC-8 → quantise.

    All operations in float64 for numerical stability.

    If ``trace`` is a list, each transformation appends a dict:
    ``{"step", "detail", "tensor_stats"?}`` for UI / debugging.
    """
    # fp64 math
    t = intent_tensor.astype(np.float64)
    s = salt.astype(np.float64)
    w1 = W1.astype(np.float64)
    w2 = W2.astype(np.float64)
    cb = codebook.astype(np.float64)

    if trace is not None:
        trace.append({
            "step": "encode.input",
            "detail": "Intent tensor (384-d) before normalise",
            "tensor_stats": vector_stats(t),
        })

    # normalize
    norm = np.linalg.norm(t)
    if norm > 0:
        normalised = t / norm * 0.5
    else:
        normalised = t.copy()

    if trace is not None:
        trace.append({
            "step": "encode.normalise",
            "detail": f"L2 norm before={norm:.6f}; scale to max half-radius 0.5",
            "tensor_stats": vector_stats(normalised),
        })

    # add salt
    salted = normalised + s

    if trace is not None:
        trace.append({
            "step": "encode.salt_inject",
            "detail": "Add per-message salt (HKDF-derived, same length as tensor)",
            "tensor_stats": vector_stats(salted),
        })

    # project
    intermediate = np.tanh(salted @ w1)
    projected = intermediate @ w2

    if trace is not None:
        trace.append({
            "step": "encode.project_tanh",
            "detail": "intermediate = tanh(salted @ W1); projected = intermediate @ W2",
            "tensor_stats": vector_stats(intermediate),
            "secondary_stats": vector_stats(projected),
        })

    # RCC-8 check
    if ledger_vectors is not None and ledger_vectors.shape[0] > 0:
        lv = ledger_vectors.astype(np.float64)
        similarities = cosine_distance_matrix(
            normalised.astype(np.float32), lv.astype(np.float32)
        )
        max_sim = float(np.max(similarities))
        relation = compute_rcc8_relation(max_sim)
        rcc_detail = f"max cosine vs vocabulary={max_sim:.4f} -> relation {relation.value}"
    else:
        relation = RCC8Relation.DC
        rcc_detail = "empty ledger -> default DC (disconnected)"

    if trace is not None:
        trace.append({
            "step": "encode.rcc8_ledger",
            "detail": rcc_detail,
            "ledger_rows": int(ledger_vectors.shape[0]) if ledger_vectors is not None else 0,
        })

    # L2 quantize
    diffs = cb - projected[np.newaxis, :]
    l2_dists = np.linalg.norm(diffs, axis=1)
    idx = int(np.argmin(l2_dists))

    quant_error = float(l2_dists[idx])
    if quant_error > 5.0:
        logger.warning("High quantisation L2 error (%.3f)", quant_error)

    if trace is not None:
        trace.append({
            "step": "encode.quantize_l2",
            "detail": f"codebook index={idx}, L2 quantisation error={quant_error:.4f}",
            "codebook_idx": idx,
            "quant_error_l2": quant_error,
        })

    return idx, relation, projected.astype(np.float32)


def decode_tensor(
    idx: int,
    codebook: np.ndarray,
    salt: np.ndarray,
    W1: np.ndarray,
    W2: np.ndarray,
    trace: Optional[list[dict[str, Any]]] = None,
) -> np.ndarray:
    """Full decode pipeline: codebook lookup → reverse W2 → arctanh → reverse W1 → desalt.

    Because W1 and W2 are orthogonal (QR), their inverse is their transpose.
    All operations in float64 for stability.

    If ``trace`` is a list, append one dict per transformation (for UI).
    """
    cb = codebook.astype(np.float64)
    s = salt.astype(np.float64)
    w1 = W1.astype(np.float64)
    w2 = W2.astype(np.float64)

    # lookup
    cb_vec = cb[idx].copy()

    if trace is not None:
        trace.append({
            "step": "decode.codebook_lookup",
            "detail": f"idx={idx} -> 384-d codebook vector",
            "tensor_stats": vector_stats(cb_vec),
        })

    # reverse W2
    intermediate = cb_vec @ w2.T

    if trace is not None:
        trace.append({
            "step": "decode.reverse_W2",
            "detail": "intermediate = codebook_vec @ W2.T",
            "tensor_stats": vector_stats(intermediate),
        })

    # arctanh
    clipped = np.clip(intermediate, -ARCTANH_CLIP, ARCTANH_CLIP)
    pre_tanh = np.arctanh(clipped)

    if trace is not None:
        trace.append({
            "step": "decode.arctanh",
            "detail": f"clip to +/-{ARCTANH_CLIP} then arctanh",
            "tensor_stats": vector_stats(pre_tanh),
        })

    # re-normalize
    pt_norm = np.linalg.norm(pre_tanh)
    if pt_norm > 0:
        pre_tanh = pre_tanh / pt_norm * 0.5

    if trace is not None:
        trace.append({
            "step": "decode.post_arctanh_norm",
            "detail": "L2 normalise to half-radius 0.5 (stability)",
            "tensor_stats": vector_stats(pre_tanh),
        })

    # reverse W1
    salted = pre_tanh @ w1.T

    if trace is not None:
        trace.append({
            "step": "decode.reverse_W1",
            "detail": "salted = pre_tanh @ W1.T",
            "tensor_stats": vector_stats(salted),
        })

    # remove salt
    normalised = salted - s

    if trace is not None:
        trace.append({
            "step": "decode.desalt",
            "detail": "normalised = salted − salt",
            "tensor_stats": vector_stats(normalised),
        })

    # unit scale
    norm = np.linalg.norm(normalised)
    if norm > 0:
        reconstructed = normalised / norm
    else:
        reconstructed = normalised

    if trace is not None:
        trace.append({
            "step": "decode.unit_rescale",
            "detail": f"L2 norm before rescale={norm:.6f} -> unit intent direction",
            "tensor_stats": vector_stats(reconstructed),
        })

    return reconstructed.astype(np.float32)


def encode_pipeline_vector_stages(
    intent_tensor: np.ndarray,
    salt: np.ndarray,
    W1: np.ndarray,
    W2: np.ndarray,
    codebook: np.ndarray,
) -> tuple[OrderedDict[str, np.ndarray], int]:
    """Ordered 384-d snapshots after each encode transform (for docs / markdown).

    Mirrors ``encode_tensor`` geometry through quantisation. Does not run RCC-8.
    """
    t = intent_tensor.astype(np.float64)
    s = salt.astype(np.float64)
    w1 = W1.astype(np.float64)
    w2 = W2.astype(np.float64)
    cb = codebook.astype(np.float64)

    out: OrderedDict[str, np.ndarray] = OrderedDict()
    out["01_raw_intent"] = t.astype(np.float32)

    norm = np.linalg.norm(t)
    normalised = (t / norm * 0.5) if norm > 0 else t.copy()
    out["02_normalised_half_radius"] = normalised.astype(np.float32)

    salted = normalised + s
    out["03_plus_salt"] = salted.astype(np.float32)

    intermediate = np.tanh(salted @ w1)
    projected = intermediate @ w2
    out["04_tanh_W1_hidden"] = intermediate.astype(np.float32)
    out["05_W2_projected"] = projected.astype(np.float32)

    diffs = cb - projected[np.newaxis, :]
    idx = int(np.argmin(np.linalg.norm(diffs, axis=1)))
    out["06_codebook_row_idx"] = cb[idx].astype(np.float32)

    return out, idx


def decode_pipeline_vector_stages(
    idx: int,
    codebook: np.ndarray,
    salt: np.ndarray,
    W1: np.ndarray,
    W2: np.ndarray,
) -> OrderedDict[str, np.ndarray]:
    """Ordered 384-d snapshots for each decode step (for docs / markdown)."""
    cb = codebook.astype(np.float64)
    s = salt.astype(np.float64)
    w1 = W1.astype(np.float64)
    w2 = W2.astype(np.float64)

    out: OrderedDict[str, np.ndarray] = OrderedDict()
    cb_vec = cb[idx].copy()
    out["01_codebook_row"] = cb_vec.astype(np.float32)

    intermediate = cb_vec @ w2.T
    out["02_reverse_W2"] = intermediate.astype(np.float32)

    clipped = np.clip(intermediate, -ARCTANH_CLIP, ARCTANH_CLIP)
    pre_tanh = np.arctanh(clipped)
    out["03_arctanh"] = pre_tanh.astype(np.float32)

    pt_norm = np.linalg.norm(pre_tanh)
    if pt_norm > 0:
        pre_tanh = pre_tanh / pt_norm * 0.5
    out["04_post_arctanh_norm"] = pre_tanh.astype(np.float32)

    salted = pre_tanh @ w1.T
    out["05_reverse_W1_salted"] = salted.astype(np.float32)

    normalised = salted - s
    out["06_desalted"] = normalised.astype(np.float32)

    n2 = np.linalg.norm(normalised)
    reconstructed = (normalised / n2) if n2 > 0 else normalised
    out["07_unit_intent"] = reconstructed.astype(np.float32)

    return out


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
