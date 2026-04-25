"""
GQESL A2A — Cryptographic Foundation

Handles ECDH key exchange (X25519), HKDF key derivation, projection matrix
generation (QR-orthogonal), codebook generation, per-message salt, HMAC
signing / verification, key rotation, and the module-level key registry.

SECURITY NOTE: SessionKeys objects live ONLY in the _key_registry dict
(process memory). They are NEVER written to LangGraph checkpointed state.
"""

from __future__ import annotations

import hashlib
import hmac as _hmac
import logging
import struct
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
)

from gqesl_a2a.config import (
    CODEBOOK_SIZE,
    HKDF_INFO_BASIS,
    HKDF_INFO_CODEBOOK,
    HKDF_INFO_HMAC,
    HKDF_INFO_SALT,
    HKDF_INFO_W1,
    HKDF_INFO_W2,
    HMAC_KEY_LEN,
    KEY_ROTATION_INTERVAL,
    SALT_LEN,
    SESSION_MAX_MESSAGES,
    TENSOR_DIM,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# SessionKeys — holds ALL derived cryptographic material for one session
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SessionKeys:
    """All session-scoped cryptographic material.

    W1 and W2 are orthogonal matrices (Q from QR decomposition).
    This guarantees that ``W.T`` is the exact inverse — no pseudo-inverse needed.
    """

    W1: np.ndarray                          # (TENSOR_DIM, TENSOR_DIM) orthogonal
    W2: np.ndarray                          # (TENSOR_DIM, TENSOR_DIM) orthogonal
    codebook: np.ndarray                    # (CODEBOOK_SIZE, TENSOR_DIM) L2-normalised
    hmac_key: bytes                         # 32 bytes — constant for entire session
    salt_seed: bytes                        # 32 bytes — rotates with epoch
    basis_seed: bytes                       # 32 bytes — for concept basis matrix
    shared_secret: bytes                    # 32 bytes — raw ECDH output
    session_nonce: bytes                    # 16 bytes — session-unique nonce
    epoch: int = 0                          # current key rotation epoch


# ═══════════════════════════════════════════════════════════════════════════
# Module-level Key Registry  (Issue #8 — secrets NEVER in state)
# ═══════════════════════════════════════════════════════════════════════════

_key_registry: dict[str, SessionKeys] = {}


def register_session_keys(session_id: str, keys: SessionKeys) -> None:
    """Store session keys in the process-local registry."""
    _key_registry[session_id] = keys
    logger.debug("Registered session keys for %s (epoch %d)", session_id, keys.epoch)


def get_session_keys(session_id: str) -> SessionKeys:
    """Retrieve session keys.  Raises KeyError if session is unknown."""
    return _key_registry[session_id]


def clear_session_keys(session_id: str) -> None:
    """Remove session keys from registry (session teardown)."""
    _key_registry.pop(session_id, None)
    logger.info("Cleared session keys for %s", session_id)


def has_session(session_id: str) -> bool:
    return session_id in _key_registry


# ═══════════════════════════════════════════════════════════════════════════
# ECDH Key Exchange (X25519)
# ═══════════════════════════════════════════════════════════════════════════

def generate_keypair() -> tuple[X25519PrivateKey, bytes]:
    """Generate an X25519 keypair.

    Returns (private_key_object, public_key_raw_bytes).
    """
    private_key = X25519PrivateKey.generate()
    public_bytes = private_key.public_key().public_bytes(
        Encoding.Raw, PublicFormat.Raw
    )
    return private_key, public_bytes


def compute_shared_secret(
    private_key: X25519PrivateKey,
    peer_public_bytes: bytes,
) -> bytes:
    """Compute the 32-byte ECDH shared secret."""
    peer_public_key = X25519PublicKey.from_public_bytes(peer_public_bytes)
    return private_key.exchange(peer_public_key)


# ═══════════════════════════════════════════════════════════════════════════
# HKDF Key Derivation
# ═══════════════════════════════════════════════════════════════════════════

def _hkdf_derive(
    shared_secret: bytes,
    info: bytes,
    length: int,
    salt: Optional[bytes] = None,
) -> bytes:
    """Derive ``length`` bytes from the shared secret using HKDF-SHA256."""
    hkdf = HKDF(
        algorithm=SHA256(),
        length=length,
        salt=salt,
        info=info,
    )
    return hkdf.derive(shared_secret)


def _epoch_info(base_info: bytes, epoch: int) -> bytes:
    """Append the epoch counter to an HKDF info label for key rotation."""
    return base_info + epoch.to_bytes(4, "big")


# ═══════════════════════════════════════════════════════════════════════════
# Projection Matrix Generation (QR-orthogonal)
# ═══════════════════════════════════════════════════════════════════════════

def generate_projection_matrix(seed: bytes, dim: int = TENSOR_DIM) -> np.ndarray:
    """Generate a deterministic orthogonal matrix via QR decomposition.

    The resulting matrix Q satisfies ``Q @ Q.T == I`` (within float64
    precision), which means the inverse is simply ``Q.T``.
    """
    rng = np.random.default_rng(
        int.from_bytes(hashlib.sha256(seed).digest()[:8], "big")
    )
    random_matrix = rng.standard_normal((dim, dim)).astype(np.float64)
    Q, R = np.linalg.qr(random_matrix)
    # Ensure deterministic sign convention (positive diagonal of R)
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    Q = Q * signs[np.newaxis, :]
    return Q.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Codebook Generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_codebook(
    seed: bytes,
    size: int = CODEBOOK_SIZE,
    dim: int = TENSOR_DIM,
) -> np.ndarray:
    """Generate a codebook of ``size`` L2-normalised random vectors."""
    rng = np.random.default_rng(
        int.from_bytes(hashlib.sha256(seed).digest()[:8], "big")
    )
    vectors = rng.standard_normal((size, dim)).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)  # avoid division by zero
    return vectors / norms


def warm_codebook(
    codebook: np.ndarray,
    basis_matrix: np.ndarray,
    W1: np.ndarray,
    W2: np.ndarray,
    n_warm: int = 0,
) -> np.ndarray:
    """Warm up the codebook by projecting concept basis vectors.

    Replaces the first ``n_warm`` codebook entries with the projected
    representations of basis vectors (tanh(basis * 0.5 @ W1) @ W2).
    This ensures the codebook covers the semantic space the agents will
    actually use, enabling high cosine recovery after decode.

    Called during session bootstrap, before any messages are sent.
    """
    if n_warm <= 0:
        n_warm = min(basis_matrix.shape[0], codebook.shape[0] // 2)

    cb = codebook.copy()
    w1 = W1.astype(np.float64)
    w2 = W2.astype(np.float64)

    for i in range(min(n_warm, basis_matrix.shape[0])):
        # Project basis vector through the same pipeline as encode
        bv = basis_matrix[i].astype(np.float64)
        norm = np.linalg.norm(bv)
        if norm > 0:
            bv = bv / norm * 0.5  # Same normalisation as encode
        projected = np.tanh(bv @ w1) @ w2
        # L2-normalise for codebook
        pnorm = np.linalg.norm(projected)
        if pnorm > 0:
            projected = projected / pnorm
        cb[i] = projected.astype(np.float32)

    return cb


# ═══════════════════════════════════════════════════════════════════════════
# Full Session Key Derivation
# ═══════════════════════════════════════════════════════════════════════════

def derive_session_keys(
    shared_secret: bytes,
    session_nonce: bytes,
    epoch: int = 0,
    *,
    _existing_hmac_key: Optional[bytes] = None,
) -> SessionKeys:
    """Derive all session keys from the ECDH shared secret.

    ``_existing_hmac_key`` is used during key rotation so the HMAC key
    stays constant for the entire session lifetime.
    """
    salt = session_nonce  # session nonce acts as HKDF salt

    # W1 and W2 — rotate with epoch
    w1_seed = _hkdf_derive(shared_secret, _epoch_info(HKDF_INFO_W1, epoch), 32, salt)
    w2_seed = _hkdf_derive(shared_secret, _epoch_info(HKDF_INFO_W2, epoch), 32, salt)
    W1 = generate_projection_matrix(w1_seed)
    W2 = generate_projection_matrix(w2_seed)

    # Codebook — rotates with epoch
    cb_seed = _hkdf_derive(shared_secret, _epoch_info(HKDF_INFO_CODEBOOK, epoch), 32, salt)
    codebook = generate_codebook(cb_seed)

    # HMAC key — constant for session (not epoch-scoped)
    if _existing_hmac_key is not None:
        hmac_key = _existing_hmac_key
    else:
        hmac_key = _hkdf_derive(shared_secret, HKDF_INFO_HMAC, HMAC_KEY_LEN, salt)

    # Salt seed — rotates with epoch
    salt_seed = _hkdf_derive(shared_secret, _epoch_info(HKDF_INFO_SALT, epoch), SALT_LEN, salt)

    # Basis seed — constant for session (concept basis is session-private)
    basis_seed = _hkdf_derive(shared_secret, HKDF_INFO_BASIS, 32, salt)

    return SessionKeys(
        W1=W1,
        W2=W2,
        codebook=codebook,
        hmac_key=hmac_key,
        salt_seed=salt_seed,
        basis_seed=basis_seed,
        shared_secret=shared_secret,
        session_nonce=session_nonce,
        epoch=epoch,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Per-Message Salt
# ═══════════════════════════════════════════════════════════════════════════

def compute_salt(salt_seed: bytes, counter: int, dim: int = TENSOR_DIM) -> np.ndarray:
    """Derive a deterministic per-message salt vector.

    Uses HKDF to derive a seed for a numpy RNG, which then generates
    well-behaved float values.  The output is normalised to a small
    magnitude (≤ 0.1) so it acts as a perturbation rather than dominating
    the intent tensor.
    """
    # Derive a seed for the RNG (not raw float reinterpretation, which can
    # produce NaN / inf IEEE 754 patterns)
    raw = _hkdf_derive(
        salt_seed,
        info=b"gqesl-msg-salt" + counter.to_bytes(8, "big"),
        length=8,  # 8 bytes → uint64 seed
    )
    rng = np.random.default_rng(int.from_bytes(raw, "big"))
    salt_vec = rng.standard_normal(dim).astype(np.float32)
    # Normalise to small magnitude
    norm = np.linalg.norm(salt_vec)
    if norm > 0:
        salt_vec = salt_vec / norm * 0.1
    return salt_vec


# ═══════════════════════════════════════════════════════════════════════════
# HMAC Signing & Verification
# ═══════════════════════════════════════════════════════════════════════════

def _pack_payload(v: int, idx: int, relation: str, counter: int) -> bytes:
    """Pack wire fields into a canonical byte string for HMAC computation."""
    return struct.pack(">B", v) + struct.pack(">H", idx) + relation.encode("ascii") + struct.pack(">I", counter)


def sign_packet(hmac_key: bytes, v: int, idx: int, relation: str, counter: int) -> bytes:
    """Compute HMAC-SHA256 over ``(v || idx || relation || counter)``."""
    payload = _pack_payload(v, idx, relation, counter)
    return _hmac.new(hmac_key, payload, hashlib.sha256).digest()


def verify_packet(
    hmac_key: bytes,
    v: int,
    idx: int,
    relation: str,
    counter: int,
    hmac_digest: bytes,
) -> bool:
    """Constant-time HMAC verification."""
    expected = sign_packet(hmac_key, v, idx, relation, counter)
    return _hmac.compare_digest(expected, hmac_digest)


# ═══════════════════════════════════════════════════════════════════════════
# Key Rotation
# ═══════════════════════════════════════════════════════════════════════════

def should_rotate(counter: int, current_epoch: int) -> bool:
    """Return True when the counter crosses the next epoch boundary."""
    return counter > 0 and counter >= (current_epoch + 1) * KEY_ROTATION_INTERVAL


def rotate_keys(session_id: str) -> SessionKeys:
    """Rotate session keys to the next epoch.

    Re-derives W1, W2, codebook, and salt_seed.  HMAC key stays constant.
    Updates the registry in-place.
    """
    old = get_session_keys(session_id)
    new_epoch = old.epoch + 1
    new_keys = derive_session_keys(
        old.shared_secret,
        old.session_nonce,
        new_epoch,
        _existing_hmac_key=old.hmac_key,
    )
    register_session_keys(session_id, new_keys)
    logger.info("Rotated keys for session %s → epoch %d", session_id, new_epoch)
    return new_keys


def should_terminate_session(counter: int) -> bool:
    """Return True when the session should be torn down entirely."""
    return counter >= SESSION_MAX_MESSAGES
