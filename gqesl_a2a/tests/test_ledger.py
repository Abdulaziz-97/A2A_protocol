"""
Tests for core/ledger.py

Covers:
  - Register and search concepts
  - Session isolation
  - Drift scoring
  - Centroid update
"""

import numpy as np
import pytest

from gqesl_a2a.config import TENSOR_DIM
from gqesl_a2a.core.ledger import SemanticLedger


@pytest.fixture
def ledger():
    return SemanticLedger()


class TestRegisterSearch:
    def test_register_and_find(self, ledger):
        vec = np.random.randn(TENSOR_DIM).astype(np.float32)
        ledger.register(vec, "test_concept", "session_1")
        results = ledger.search(vec, "session_1", k=1)
        assert len(results) == 1
        assert results[0][0] == "test_concept"
        assert results[0][1] > 0.99  # same vector

    def test_nearest_neighbour(self, ledger):
        v1 = np.zeros(TENSOR_DIM, dtype=np.float32)
        v1[0] = 1.0
        v2 = np.zeros(TENSOR_DIM, dtype=np.float32)
        v2[1] = 1.0
        query = np.zeros(TENSOR_DIM, dtype=np.float32)
        query[0] = 0.9
        query[1] = 0.1

        ledger.register(v1, "concept_a", "s1")
        ledger.register(v2, "concept_b", "s1")
        results = ledger.search(query, "s1", k=1)
        assert results[0][0] == "concept_a"

    def test_empty_search(self, ledger):
        vec = np.random.randn(TENSOR_DIM).astype(np.float32)
        results = ledger.search(vec, "empty_session")
        assert results == []


class TestSessionIsolation:
    def test_different_sessions(self, ledger):
        vec = np.random.randn(TENSOR_DIM).astype(np.float32)
        ledger.register(vec, "shared_name", "session_a")
        results = ledger.search(vec, "session_b")
        assert results == []

    def test_clear_session(self, ledger):
        vec = np.random.randn(TENSOR_DIM).astype(np.float32)
        ledger.register(vec, "c1", "s1")
        ledger.clear_session("s1")
        assert ledger.concept_count("s1") == 0


class TestDrift:
    def test_no_drift_single_usage(self, ledger):
        vec = np.random.randn(TENSOR_DIM).astype(np.float32)
        ledger.register(vec, "c1", "s1")
        assert ledger.drift_score("c1", "s1") == 0.0

    def test_drift_with_variation(self, ledger):
        base = np.random.randn(TENSOR_DIM).astype(np.float32)
        ledger.register(base, "c1", "s1")

        # varied usage
        for i in range(10):
            noisy = base + np.random.randn(TENSOR_DIM).astype(np.float32) * 0.5
            ledger.record_usage("c1", "s1", noisy)

        score = ledger.drift_score("c1", "s1")
        assert score > 0  # drift expected

    def test_centroid_update(self, ledger):
        base = np.random.randn(TENSOR_DIM).astype(np.float32)
        ledger.register(base, "c1", "s1")

        new_vec = np.random.randn(TENSOR_DIM).astype(np.float32)
        ledger.update_centroid("c1", "s1", new_vec)

        # find updated
        results = ledger.search(new_vec, "s1")
        assert results[0][1] > 0.99


class TestGetAll:
    def test_get_all(self, ledger):
        v1 = np.random.randn(TENSOR_DIM).astype(np.float32)
        v2 = np.random.randn(TENSOR_DIM).astype(np.float32)
        ledger.register(v1, "c1", "s1")
        ledger.register(v2, "c2", "s1")
        all_concepts = ledger.get_all("s1")
        assert len(all_concepts) == 2
        assert "c1" in all_concepts
        assert "c2" in all_concepts

    def test_get_all_vectors_matrix(self, ledger):
        for i in range(5):
            ledger.register(np.random.randn(TENSOR_DIM).astype(np.float32), f"c{i}", "s1")
        mat = ledger.get_all_vectors_matrix("s1")
        assert mat is not None
        assert mat.shape == (5, TENSOR_DIM)

    def test_get_rcc8_ledger_matrix_filters_prefixes(self, ledger):
        v = np.random.randn(TENSOR_DIM).astype(np.float32)
        ledger.register(v, "concept_0", "s1")
        ledger.register(v * 0.9, "warm_EXTRACT_PERSON_JSON", "s1")
        ledger.register(v * 0.8, "learned_foo_c1", "s1")
        mat = ledger.get_rcc8_ledger_matrix("s1")
        assert mat is not None
        assert mat.shape[0] == 2
