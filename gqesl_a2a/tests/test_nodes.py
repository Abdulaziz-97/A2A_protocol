"""
Tests for graph/nodes.py

Covers:
  - No NL strings in state after build_intent_node (Fix #2)
  - wire_packet is None after verify_node (Fix #9)
  - session_keys not in GQESLState (Fix #8)
  - Individual node behaviour
"""

import numpy as np
import pytest

from gqesl_a2a.core.crypto import (
    clear_session_keys,
    compute_salt,
    derive_session_keys,
    get_session_keys,
    register_session_keys,
    sign_packet,
)
from gqesl_a2a.core.ledger import SemanticLedger, get_ledger, set_ledger
from gqesl_a2a.core.tensor_builder import build_basis_matrix
from gqesl_a2a.graph.nodes import (
    build_intent_node,
    collapse_node,
    decode_node,
    encode_node,
    error_handler_node,
    rcc8_route_node,
    sign_node,
    verify_node,
)


@pytest.fixture(autouse=True)
def setup_session():
    """Create a test session with known keys."""
    from gqesl_a2a.core.crypto import warm_codebook
    set_ledger(SemanticLedger())
    secret = b"\xaa" * 32
    nonce = b"\xbb" * 16
    keys = derive_session_keys(secret, nonce)

    # Warm the codebook with projected basis vectors
    basis = build_basis_matrix(keys.basis_seed)
    keys.codebook = warm_codebook(keys.codebook, basis, keys.W1, keys.W2)

    register_session_keys("test-session", keys)

    # Register basis concepts
    ledger = get_ledger()
    for i in range(20):
        ledger.register(basis[i], f"concept_{i}", "test-session")

    yield
    clear_session_keys("test-session")


class TestEphemeralNL:
    """Fix #2: No NL in checkpointed state."""

    def test_build_intent_no_messages_in_state(self):
        state = {
            "session_id": "test-session",
            "intent": {
                "task_type": "EXTRACT",
                "entity_type": "PERSON",
                "output_format": "JSON",
                "priority": 0.5,
            },
        }
        result = build_intent_node(state)
        # State must not contain 'messages' field
        assert "messages" not in result
        # Result should only have structured data
        assert "intent" in result
        assert "intent_tensor" in result
        assert isinstance(result["intent"], dict)
        assert isinstance(result["intent_tensor"], list)


class TestWirePacketNulling:
    """Fix #9: wire_packet is None after verify."""

    def test_wire_packet_nulled_on_success(self):
        keys = get_session_keys("test-session")
        hmac = sign_packet(keys.hmac_key, 1, 42, "EQ", 0)
        state = {
            "session_id": "test-session",
            "counter": 0,
            "wire_packet": {
                "v": 1,
                "idx": 42,
                "r": "EQ",
                "c": 0,
                "h": hmac.hex(),
            },
        }
        result = verify_node(state)
        assert result["wire_packet"] is None

    def test_wire_packet_nulled_on_failure(self):
        state = {
            "session_id": "test-session",
            "counter": 0,
            "wire_packet": {
                "v": 1,
                "idx": 42,
                "r": "EQ",
                "c": 0,
                "h": ("00" * 32),  # Wrong HMAC
            },
        }
        result = verify_node(state)
        assert result["wire_packet"] is None
        assert result["error"] == "hmac_verification_failed"


class TestSessionKeysNotInState:
    """Fix #8: session_keys must never appear in state."""

    def test_no_session_keys_in_build_intent_result(self):
        state = {
            "session_id": "test-session",
            "intent": {"task_type": "EXTRACT", "entity_type": "DATA",
                       "output_format": "JSON", "priority": 0.5},
        }
        result = build_intent_node(state)
        assert "session_keys" not in result

    def test_no_session_keys_in_encode_result(self):
        state = {
            "session_id": "test-session",
            "counter": 0,
            "key_epoch": 0,
            "intent_tensor": np.random.randn(384).tolist(),
        }
        result = encode_node(state)
        assert "session_keys" not in result


class TestRCC8Routing:
    def test_route_eq(self):
        result = rcc8_route_node({"strategy": "EXACT_MATCH"})
        # rcc8_route reads strategy, which is already set
        # This node just passes through
        assert result is not None

    def test_route_dc(self):
        result = rcc8_route_node({"strategy": "NEGOTIATE_FIRST"})
        assert result is not None


class TestErrorHandler:
    def test_clears_error(self):
        state = {"error": "hmac_verification_failed", "error_source": "verify_node"}
        result = error_handler_node(state)
        assert result["error"] is None
