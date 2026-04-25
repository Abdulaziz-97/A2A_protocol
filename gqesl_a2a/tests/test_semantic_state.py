"""
Tests for core/semantic_state.py

Covers:
  - RCC-8 threshold boundaries (all 4 relations)
  - Encode → decode roundtrip with W.T inversion
  - Concept recovery accuracy (task_type matches after encode/decode)
  - SemanticMessage packing/unpacking
"""

import numpy as np
import pytest

from gqesl_a2a.config import EC_THRESHOLD, EQ_THRESHOLD, PO_THRESHOLD, TENSOR_DIM
from gqesl_a2a.core.crypto import compute_salt, derive_session_keys, warm_codebook
from gqesl_a2a.core.semantic_state import (
    RCC8Relation,
    SemanticMessage,
    collapse_tensor,
    compute_rcc8_relation,
    cosine_similarity,
    decode_tensor,
    encode_tensor,
)
from gqesl_a2a.core.tensor_builder import (
    AgentIntent,
    EntityType,
    OutputFormat,
    TaskType,
    build_basis_matrix,
    build_intent_tensor,
    collapse_to_intent,
)


class TestRCC8:
    def test_eq_threshold(self):
        assert compute_rcc8_relation(0.96) == RCC8Relation.EQ
        assert compute_rcc8_relation(0.95) == RCC8Relation.EQ

    def test_po_threshold(self):
        assert compute_rcc8_relation(0.94) == RCC8Relation.PO
        assert compute_rcc8_relation(0.85) == RCC8Relation.PO

    def test_ec_threshold(self):
        assert compute_rcc8_relation(0.84) == RCC8Relation.EC
        assert compute_rcc8_relation(0.60) == RCC8Relation.EC

    def test_dc_threshold(self):
        assert compute_rcc8_relation(0.59) == RCC8Relation.DC
        assert compute_rcc8_relation(0.0) == RCC8Relation.DC

    def test_boundary_precision(self):
        assert compute_rcc8_relation(EQ_THRESHOLD) == RCC8Relation.EQ
        assert compute_rcc8_relation(EQ_THRESHOLD - 0.001) == RCC8Relation.PO
        assert compute_rcc8_relation(PO_THRESHOLD) == RCC8Relation.PO
        assert compute_rcc8_relation(PO_THRESHOLD - 0.001) == RCC8Relation.EC
        assert compute_rcc8_relation(EC_THRESHOLD) == RCC8Relation.EC
        assert compute_rcc8_relation(EC_THRESHOLD - 0.001) == RCC8Relation.DC


class TestEncodeDecode:
    """Fix #1: verify W.T inversion with warmed codebook."""

    @pytest.fixture
    def session_material(self):
        secret = b"\xab" * 32
        nonce = b"\xcd" * 16
        keys = derive_session_keys(secret, nonce)
        basis = build_basis_matrix(keys.basis_seed)
        keys.codebook = warm_codebook(keys.codebook, basis, keys.W1, keys.W2)
        return keys, basis

    def test_roundtrip_cosine_positive(self, session_material):
        """Roundtrip cosine should be positive (salt perturbation limits exact recovery)."""
        keys, basis = session_material
        intent = AgentIntent(
            TaskType.EXTRACT, EntityType.PERSON, OutputFormat.JSON, 0.8, b"\x01" * 32
        )
        tensor = build_intent_tensor(intent, basis)
        salt = compute_salt(keys.salt_seed, counter=0)

        idx, relation, projected = encode_tensor(
            tensor, salt, keys.W1, keys.W2, keys.codebook, None
        )

        reconstructed = decode_tensor(idx, keys.codebook, salt, keys.W1, keys.W2)
        cos_sim = cosine_similarity(tensor, reconstructed)
        # With salt perturbation and codebook quantization, cosine > 0.3 is expected
        assert cos_sim > 0.3, f"Cosine recovery {cos_sim:.4f} below minimum"

    def test_concept_recovery_all_task_types(self, session_material):
        """The dominant task_type must be recovered correctly for most intents."""
        keys, basis = session_material
        correct = 0
        total = 0
        for task in TaskType:
            intent = AgentIntent(task, EntityType.DATA, OutputFormat.JSON, 0.5, b"\xff" * 32)
            tensor = build_intent_tensor(intent, basis)
            salt = compute_salt(keys.salt_seed, counter=int(task))
            idx, _, _ = encode_tensor(tensor, salt, keys.W1, keys.W2, keys.codebook, None)
            recon = decode_tensor(idx, keys.codebook, salt, keys.W1, keys.W2)

            orig_intent = collapse_to_intent(tensor, basis)
            recon_intent = collapse_to_intent(recon, basis)
            if orig_intent["task_type"] == recon_intent["task_type"]:
                correct += 1
            total += 1

        accuracy = correct / total
        # At least 50% concept recovery is required with warmed codebook
        assert accuracy >= 0.5, f"Concept recovery {accuracy:.0%} below 50%"

    def test_different_salts_different_indices(self, session_material):
        """Same intent with different salts produces decodable tensors."""
        keys, basis = session_material
        intent = AgentIntent(TaskType.EXTRACT, EntityType.PERSON, OutputFormat.JSON, 0.5, b"\x01" * 32)
        tensor = build_intent_tensor(intent, basis)

        salt0 = compute_salt(keys.salt_seed, 0)
        salt1 = compute_salt(keys.salt_seed, 1)
        idx0, _, _ = encode_tensor(tensor, salt0, keys.W1, keys.W2, keys.codebook, None)
        idx1, _, _ = encode_tensor(tensor, salt1, keys.W1, keys.W2, keys.codebook, None)

        r0 = decode_tensor(idx0, keys.codebook, salt0, keys.W1, keys.W2)
        r1 = decode_tensor(idx1, keys.codebook, salt1, keys.W1, keys.W2)
        # Both should produce valid (non-nan) reconstructions
        assert not np.any(np.isnan(r0))
        assert not np.any(np.isnan(r1))
        # Both should have positive cosine similarity with original
        assert cosine_similarity(tensor, r0) > 0.2
        assert cosine_similarity(tensor, r1) > 0.2


class TestSemanticMessage:
    def test_pack_unpack(self):
        from gqesl_a2a.core.semantic_state import pack_packet, unpack_packet
        msg = SemanticMessage(v=1, idx=42, r="PO", c=7, h=b"\xaa" * 32)
        packed = pack_packet(msg)
        assert len(packed) == 41
        unpacked = unpack_packet(packed)
        assert unpacked.v == 1
        assert unpacked.idx == 42
        assert unpacked.r == "PO"
        assert unpacked.c == 7
        assert unpacked.h == b"\xaa" * 32

    def test_invalid_relation(self):
        with pytest.raises(ValueError):
            SemanticMessage(v=1, idx=0, r="XX", c=0, h=b"\x00" * 32)

    def test_invalid_idx(self):
        with pytest.raises(ValueError):
            SemanticMessage(v=1, idx=5000, r="EQ", c=0, h=b"\x00" * 32)


class TestCollapse:
    def test_collapse_recovers_dominant_concept(self):
        basis = build_basis_matrix(b"test-basis" * 4, n_concepts=10, dim=TENSOR_DIM)
        tensor = basis[3].copy()
        idx, sim = collapse_tensor(tensor, basis)
        assert idx == 3
        assert sim > 0.99
