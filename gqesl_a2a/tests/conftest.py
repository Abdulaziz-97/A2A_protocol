import asyncio
import pytest
import numpy as np

from gqesl_a2a.config import TENSOR_DIM
from gqesl_a2a.core.crypto import (
    derive_session_keys,
    register_session_keys,
    clear_session_keys,
)
from gqesl_a2a.core.ledger import SemanticLedger, set_ledger
from gqesl_a2a.core.tensor_builder import (
    AgentIntent,
    TaskType,
    EntityType,
    OutputFormat,
    build_basis_matrix,
)
from gqesl_a2a.agents.session import InProcessBus
from gqesl_a2a.graph.state import GQESLState


@pytest.fixture
def session_keys():
    """Provides registered session keys for 'test-session'."""
    set_ledger(SemanticLedger())
    keys = derive_session_keys(b"\xaa" * 32, b"\xbb" * 16)
    register_session_keys("test-session", keys)
    yield keys
    clear_session_keys("test-session")


@pytest.fixture
def populated_ledger(session_keys):
    """Provides a ledger populated with 10 concepts."""
    ledger = SemanticLedger()
    set_ledger(ledger)
    rng = np.random.default_rng(42)
    for i in range(10):
        vec = rng.standard_normal(TENSOR_DIM).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        ledger.register(vec, f"concept_{i}", "test-session")
    return ledger


@pytest.fixture
def sample_intent():
    """Provides a basic AgentIntent."""
    return AgentIntent(
        task_type=TaskType.EXTRACT,
        entity_type=EntityType.PERSON,
        output_format=OutputFormat.JSON,
        priority=0.8,
        source_ref=b"\x01" * 32,
    )


@pytest.fixture
def base_state(sample_intent, session_keys):
    """Provides a base GQESLState dict."""
    basis = build_basis_matrix(session_keys.basis_seed)
    return {
        "session_id": "test-session",
        "counter": 0,
        "key_epoch": 0,
        "intent": {
            "task_type": sample_intent.task_type.name,
            "entity_type": sample_intent.entity_type.name,
            "output_format": sample_intent.output_format.name,
            "priority": sample_intent.priority,
        },
        "task_description": "",
    }


@pytest.fixture
def inprocess_bus():
    """Provides a tuple of (Agent A bus, Agent B bus)."""
    return InProcessBus.create_pair()
