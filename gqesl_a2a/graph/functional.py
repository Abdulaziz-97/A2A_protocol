"""
GQESL A2A — Functional Pipeline API

Provides a simplified functional interface using LangGraph's @entrypoint
and @task decorators as an alternative to the full StateGraph.
"""

from __future__ import annotations

import logging
import os
import uuid

import numpy as np

from gqesl_a2a.core.crypto import (
    compute_salt,
    compute_shared_secret,
    derive_session_keys,
    generate_keypair,
    get_session_keys,
    register_session_keys,
    sign_packet,
    verify_packet,
)
from gqesl_a2a.core.ledger import get_ledger
from gqesl_a2a.core.semantic_state import (
    SemanticMessage,
    collapse_tensor,
    decode_tensor,
    encode_tensor,
)
from gqesl_a2a.core.tensor_builder import (
    AgentIntent,
    build_basis_matrix,
    build_intent_tensor,
)

logger = logging.getLogger(__name__)


def functional_encode(
    session_id: str,
    intent: AgentIntent,
    counter: int,
) -> dict:
    """Encode an intent into a wire packet (functional, no graph).

    Returns a dict with the wire packet fields.
    """
    keys = get_session_keys(session_id)
    basis = build_basis_matrix(keys.basis_seed)
    tensor = build_intent_tensor(intent, basis)

    salt = compute_salt(keys.salt_seed, counter)
    ledger = get_ledger()
    ledger_vecs = ledger.get_all_vectors_matrix(session_id)

    idx, relation, projected = encode_tensor(
        tensor, salt, keys.W1, keys.W2, keys.codebook, ledger_vecs
    )

    hmac_digest = sign_packet(keys.hmac_key, idx, relation.value, counter)

    return {
        "idx": idx,
        "r": relation.value,
        "c": counter,
        "h": hmac_digest.hex(),
        "intent_tensor": tensor.tolist(),
    }


def functional_decode(
    session_id: str,
    packet: dict,
    expected_counter: int,
) -> dict | None:
    """Decode a wire packet back to an intent (functional, no graph).

    Returns decoded info dict or None on verification failure.
    """
    keys = get_session_keys(session_id)
    idx = packet["idx"]
    r = packet["r"]
    c = packet["c"]
    h = bytes.fromhex(packet["h"])

    # Verify
    if not verify_packet(keys.hmac_key, idx, r, c, h):
        logger.error("HMAC verification failed")
        return None

    if c < expected_counter:
        logger.error("Counter replay: got %d, expected >= %d", c, expected_counter)
        return None

    salt = compute_salt(keys.salt_seed, c)
    reconstructed = decode_tensor(idx, keys.codebook, salt, keys.W1, keys.W2)

    basis = build_basis_matrix(keys.basis_seed)
    concept_idx, similarity = collapse_tensor(reconstructed, basis)

    return {
        "decoded_tensor": reconstructed.tolist(),
        "concept_idx": concept_idx,
        "concept_id": f"concept_{concept_idx}",
        "similarity": similarity,
        "relation": r,
    }


def functional_roundtrip(
    session_id: str,
    intent: AgentIntent,
    counter: int,
) -> dict:
    """Full encode → decode roundtrip for testing."""
    packet = functional_encode(session_id, intent, counter)
    result = functional_decode(session_id, packet, counter)
    if result is None:
        raise RuntimeError("Roundtrip failed: decode returned None")

    original_tensor = np.array(packet["intent_tensor"], dtype=np.float32)
    decoded_tensor = np.array(result["decoded_tensor"], dtype=np.float32)

    from gqesl_a2a.core.semantic_state import cosine_similarity
    cos_sim = cosine_similarity(original_tensor, decoded_tensor)

    return {
        **result,
        "original_tensor": packet["intent_tensor"],
        "cosine_similarity": cos_sim,
    }
