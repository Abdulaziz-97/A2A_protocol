"""
Tests for core/crypto.py

Covers:
  - ECDH shared secret agreement
  - HKDF determinism
  - W1/W2 orthogonality (W @ W.T ≈ I)
  - Codebook generation & normalisation
  - HMAC sign/verify
  - Key rotation
  - Session termination check
"""

import numpy as np
import pytest

from gqesl_a2a.config import CODEBOOK_SIZE, SESSION_MAX_MESSAGES, TENSOR_DIM
from gqesl_a2a.core.crypto import (
    SessionKeys,
    clear_session_keys,
    compute_salt,
    compute_shared_secret,
    derive_session_keys,
    generate_codebook,
    generate_keypair,
    generate_projection_matrix,
    get_session_keys,
    has_session,
    register_session_keys,
    rotate_keys,
    should_rotate,
    should_terminate_session,
    sign_packet,
    verify_packet,
)


class TestECDH:
    def test_shared_secret_agreement(self):
        """Both agents must derive the identical shared secret."""
        priv_a, pub_a = generate_keypair()
        priv_b, pub_b = generate_keypair()
        secret_a = compute_shared_secret(priv_a, pub_b)
        secret_b = compute_shared_secret(priv_b, pub_a)
        assert secret_a == secret_b
        assert len(secret_a) == 32

    def test_different_sessions_different_secrets(self):
        priv_a, pub_a = generate_keypair()
        priv_b, pub_b = generate_keypair()
        priv_c, pub_c = generate_keypair()
        secret_ab = compute_shared_secret(priv_a, pub_b)
        secret_ac = compute_shared_secret(priv_a, pub_c)
        assert secret_ab != secret_ac


class TestHKDF:
    def test_deterministic_derivation(self):
        """Same inputs → identical keys."""
        secret = b"\xaa" * 32
        nonce = b"\xbb" * 16
        keys1 = derive_session_keys(secret, nonce, epoch=0)
        keys2 = derive_session_keys(secret, nonce, epoch=0)
        assert np.allclose(keys1.W1, keys2.W1)
        assert np.allclose(keys1.W2, keys2.W2)
        assert np.allclose(keys1.codebook, keys2.codebook)
        assert keys1.hmac_key == keys2.hmac_key
        assert keys1.salt_seed == keys2.salt_seed

    def test_different_nonces_different_keys(self):
        secret = b"\xaa" * 32
        keys1 = derive_session_keys(secret, b"\x01" * 16)
        keys2 = derive_session_keys(secret, b"\x02" * 16)
        assert not np.allclose(keys1.W1, keys2.W1)

    def test_different_epochs_rotate_w1_w2(self):
        secret = b"\xaa" * 32
        nonce = b"\xbb" * 16
        keys0 = derive_session_keys(secret, nonce, epoch=0)
        keys1 = derive_session_keys(secret, nonce, epoch=1, _existing_hmac_key=keys0.hmac_key)
        assert not np.allclose(keys0.W1, keys1.W1)
        assert not np.allclose(keys0.W2, keys1.W2)
        assert keys0.hmac_key == keys1.hmac_key  # HMAC key stays constant


class TestOrthogonality:
    """Fix #1: W matrices must be orthogonal so W.T is the exact inverse."""

    def test_w1_orthogonal(self):
        keys = derive_session_keys(b"\xcc" * 32, b"\xdd" * 16)
        identity = keys.W1 @ keys.W1.T
        assert np.allclose(identity, np.eye(TENSOR_DIM), atol=1e-5)

    def test_w2_orthogonal(self):
        keys = derive_session_keys(b"\xcc" * 32, b"\xdd" * 16)
        identity = keys.W2 @ keys.W2.T
        assert np.allclose(identity, np.eye(TENSOR_DIM), atol=1e-5)

    def test_projection_matrix_deterministic(self):
        m1 = generate_projection_matrix(b"test-seed")
        m2 = generate_projection_matrix(b"test-seed")
        assert np.allclose(m1, m2)


class TestCodebook:
    def test_correct_shape(self):
        cb = generate_codebook(b"seed", size=4096, dim=384)
        assert cb.shape == (4096, 384)

    def test_l2_normalised(self):
        cb = generate_codebook(b"seed")
        norms = np.linalg.norm(cb, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_deterministic(self):
        cb1 = generate_codebook(b"seed")
        cb2 = generate_codebook(b"seed")
        assert np.allclose(cb1, cb2)


class TestHMAC:
    def test_sign_verify(self):
        key = b"\x42" * 32
        h = sign_packet(key, 1, idx=42, relation="PO", counter=7)
        assert verify_packet(key, 1, 42, "PO", 7, h)

    def test_reject_tampered_idx(self):
        key = b"\x42" * 32
        h = sign_packet(key, 1, 42, "PO", 7)
        assert not verify_packet(key, 1, 43, "PO", 7, h)

    def test_reject_tampered_relation(self):
        key = b"\x42" * 32
        h = sign_packet(key, 1, 42, "PO", 7)
        assert not verify_packet(key, 1, 42, "EQ", 7, h)

    def test_reject_tampered_counter(self):
        key = b"\x42" * 32
        h = sign_packet(key, 1, 42, "PO", 7)
        assert not verify_packet(key, 1, 42, "PO", 8, h)

    def test_reject_wrong_key(self):
        h = sign_packet(b"\x42" * 32, 1, 42, "PO", 7)
        assert not verify_packet(b"\x43" * 32, 1, 42, "PO", 7, h)


class TestSalt:
    def test_deterministic(self):
        s1 = compute_salt(b"seed" * 8, counter=5)
        s2 = compute_salt(b"seed" * 8, counter=5)
        assert np.allclose(s1, s2)

    def test_different_counters(self):
        s1 = compute_salt(b"seed" * 8, counter=0)
        s2 = compute_salt(b"seed" * 8, counter=1)
        assert not np.allclose(s1, s2)

    def test_small_magnitude(self):
        s = compute_salt(b"seed" * 8, counter=0)
        assert np.linalg.norm(s) <= 0.15  # Should be ~0.1


class TestKeyRotation:
    def test_should_rotate(self):
        assert not should_rotate(0, 0)
        assert not should_rotate(499, 0)
        assert should_rotate(500, 0)
        assert not should_rotate(500, 1)
        assert should_rotate(1000, 1)

    def test_rotate_keys_updates_registry(self):
        keys = derive_session_keys(b"\xee" * 32, b"\xff" * 16)
        register_session_keys("test-rotate", keys)
        old_w1 = keys.W1.copy()
        new_keys = rotate_keys("test-rotate")
        assert new_keys.epoch == 1
        assert not np.allclose(old_w1, new_keys.W1)
        assert new_keys.hmac_key == keys.hmac_key
        clear_session_keys("test-rotate")


class TestSessionTermination:
    def test_should_terminate(self):
        assert not should_terminate_session(9999)
        assert should_terminate_session(10000)
        assert should_terminate_session(10001)


class TestKeyRegistry:
    def test_register_and_get(self):
        keys = derive_session_keys(b"\x11" * 32, b"\x22" * 16)
        register_session_keys("test-reg", keys)
        retrieved = get_session_keys("test-reg")
        assert np.allclose(retrieved.W1, keys.W1)
        clear_session_keys("test-reg")

    def test_clear(self):
        keys = derive_session_keys(b"\x33" * 32, b"\x44" * 16)
        register_session_keys("test-clear", keys)
        clear_session_keys("test-clear")
        assert not has_session("test-clear")

    def test_missing_raises(self):
        with pytest.raises(KeyError):
            get_session_keys("nonexistent")
