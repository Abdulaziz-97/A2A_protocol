"""
Tests for full end-to-end pipeline

Covers:
  - Session bootstrap → Agent A encode → Agent B decode → correct concept
  - Multiple message exchange
  - Counter sync
"""

import numpy as np
import pytest

from gqesl_a2a.agents.agent_a import AgentA
from gqesl_a2a.agents.agent_b import AgentB
from gqesl_a2a.agents.session import bootstrap_session, sync_counters
from gqesl_a2a.core.crypto import clear_session_keys, get_session_keys
from gqesl_a2a.core.ledger import SemanticLedger, set_ledger
from gqesl_a2a.core.semantic_state import cosine_similarity, SemanticMessage, pack_packet
from gqesl_a2a.core.tensor_builder import (
    AgentIntent,
    EntityType,
    OutputFormat,
    TaskType,
    build_basis_matrix,
    build_intent_tensor,
)


@pytest.fixture
def session():
    set_ledger(SemanticLedger())
    info_a, info_b = bootstrap_session()
    yield info_a, info_b
    clear_session_keys(info_a.session_id)


class TestFullPipeline:
    def test_single_message(self, session):
        info_a, info_b = session
        agent_a = AgentA(info_a)
        agent_b = AgentB(info_b)

        intent = AgentIntent(
            TaskType.EXTRACT, EntityType.PERSON, OutputFormat.JSON,
            0.8, b"\x01" * 32,
        )
        result_a = agent_a.encode_and_sign(intent)
        result_b = agent_b.verify_and_decode(result_a["packet"])

        assert result_b is not None
        assert result_b["concept_id"].startswith("concept_")

        # Cosine recovery check
        original = np.array(result_a["intent_tensor"], dtype=np.float32)
        decoded = np.array(result_b["decoded_tensor"], dtype=np.float32)
        cos_sim = cosine_similarity(original, decoded)
        assert cos_sim > 0.3, f"Cosine {cos_sim:.4f} too low"

    def test_multiple_messages(self, session):
        info_a, info_b = session
        agent_a = AgentA(info_a)
        agent_b = AgentB(info_b)

        intents = [
            AgentIntent(TaskType.EXTRACT, EntityType.PERSON, OutputFormat.JSON, 0.8, b"\x01" * 32),
            AgentIntent(TaskType.SUMMARIZE, EntityType.DOCUMENT, OutputFormat.TEXT, 0.6, b"\x02" * 32),
            AgentIntent(TaskType.CLASSIFY, EntityType.ORGANIZATION, OutputFormat.TABLE, 0.9, b"\x03" * 32),
        ]

        for intent in intents:
            result_a = agent_a.encode_and_sign(intent)
            result_b = agent_b.verify_and_decode(result_a["packet"])
            assert result_b is not None

    def test_counter_monotonicity(self, session):
        """Replayed packets must be rejected."""
        info_a, info_b = session
        agent_a = AgentA(info_a)
        agent_b = AgentB(info_b)

        intent = AgentIntent(
            TaskType.EXTRACT, EntityType.DATA, OutputFormat.JSON,
            0.5, b"\x00" * 32,
        )

        # Send first message
        result_a = agent_a.encode_and_sign(intent)
        result_b = agent_b.verify_and_decode(result_a["packet"])
        assert result_b is not None

        # Replay same packet → must fail (counter not fresh)
        result_replay = agent_b.verify_and_decode(result_a["packet"])
        assert result_replay is None

    def test_tampered_hmac_rejected(self, session):
        info_a, info_b = session
        agent_a = AgentA(info_a)
        agent_b = AgentB(info_b)

        intent = AgentIntent(
            TaskType.SEARCH, EntityType.LOCATION, OutputFormat.JSON,
            0.5, b"\x00" * 32,
        )
        result_a = agent_a.encode_and_sign(intent)

        # Tamper with HMAC
        packet = SemanticMessage(**result_a["packet"])
        tampered_bytes = pack_packet(packet)
        # Manually alter the HMAC portion (last 32 bytes)
        tampered_bytes = tampered_bytes[:-32] + b"\x00" * 32
        
        # Unpack back to dict for agent_b.verify_and_decode which expects a dict
        from gqesl_a2a.core.semantic_state import unpack_packet
        tampered_msg = unpack_packet(tampered_bytes)
        tampered_dict = {
            "v": tampered_msg.v,
            "idx": tampered_msg.idx,
            "r": tampered_msg.r,
            "c": tampered_msg.c,
            "h": tampered_msg.h.hex(),
        }
        
        result_b = agent_b.verify_and_decode(tampered_dict)
        assert result_b is None

    def test_wire_packet_size(self, session):
        info_a, _ = session
        agent_a = AgentA(info_a)

        intent = AgentIntent(TaskType.EXTRACT, EntityType.PERSON, OutputFormat.JSON, 0.5, b"\x00" * 32)
        result_a = agent_a.encode_and_sign(intent)
        
        assert len(result_a["wire_bytes"]) == 41, "Wire packet must be exactly 41 bytes"


class TestCounterSync:
    def test_sync_advances_to_max(self):
        synced = sync_counters("test", local_counter=5, peer_counter=10)
        assert synced == 11

    def test_sync_symmetric(self):
        s1 = sync_counters("test", 5, 10)
        s2 = sync_counters("test", 10, 5)
        assert s1 == s2 == 11
