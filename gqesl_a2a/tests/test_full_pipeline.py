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
from gqesl_a2a.core.concepts import KNOWN_CONCEPTS
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

        # warmed concept
        intent = KNOWN_CONCEPTS[0]
        result_a = agent_a.encode_and_sign(intent)
        result_b = agent_b.verify_and_decode(result_a["packet"])

        assert result_b is not None
        assert result_b["concept_id"].startswith("concept_")

        # cosine check
        original = np.array(result_a["intent_tensor"], dtype=np.float32)
        decoded = np.array(result_b["decoded_tensor"], dtype=np.float32)
        cos_sim = cosine_similarity(original, decoded)
        assert cos_sim > 0.3, f"Cosine {cos_sim:.4f} too low"

    def test_known_intent_not_dc_on_wire(self, session):
        """Warmed triples should relate as EQ/PO/EC — not disconnected."""
        info_a, info_b = session
        agent_a = AgentA(info_a)
        intent = KNOWN_CONCEPTS[0]
        result_a = agent_a.encode_and_sign(intent)
        assert result_a["relation"] != "DC", (
            f"expected non-DC RCC-8 for warmed intent, got {result_a['relation']}"
        )

    def test_unknown_intent_can_be_dc(self, session):
        """Triple not in warmed vocabulary should often yield DC (no close template)."""
        info_a, _info_b = session
        agent_a = AgentA(info_a)
        intent = AgentIntent(
            TaskType.COMPARE,
            EntityType.DOCUMENT,
            OutputFormat.GRAPH,
            0.5,
            b"\x00" * 32,
        )
        result_a = agent_a.encode_and_sign(intent)
        assert result_a["relation"] == "DC"

    def test_multiple_messages(self, session):
        info_a, info_b = session
        agent_a = AgentA(info_a)
        agent_b = AgentB(info_b)

        intents = [
            next(i for i in KNOWN_CONCEPTS if i.task_type == TaskType.EXTRACT),
            next(i for i in KNOWN_CONCEPTS if i.task_type == TaskType.SUMMARIZE),
            next(
                i
                for i in KNOWN_CONCEPTS
                if i.task_type == TaskType.CLASSIFY
                and i.entity_type == EntityType.ORGANIZATION
            ),
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

        intent = next(i for i in KNOWN_CONCEPTS if i.task_type == TaskType.EXTRACT and i.entity_type == EntityType.DATA)

        # first send
        result_a = agent_a.encode_and_sign(intent)
        result_b = agent_b.verify_and_decode(result_a["packet"])
        assert result_b is not None

        # replay fails
        result_replay = agent_b.verify_and_decode(result_a["packet"])
        assert result_replay is None

    def test_tampered_hmac_rejected(self, session):
        info_a, info_b = session
        agent_a = AgentA(info_a)
        agent_b = AgentB(info_b)

        intent = next(i for i in KNOWN_CONCEPTS if i.task_type == TaskType.SEARCH and i.entity_type == EntityType.LOCATION)
        result_a = agent_a.encode_and_sign(intent)

        # tamper HMAC
        packet = SemanticMessage(**result_a["packet"])
        tampered_bytes = pack_packet(packet)
        # break digest
        tampered_bytes = tampered_bytes[:-32] + b"\x00" * 32
        
        # unpack dict
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

        intent = KNOWN_CONCEPTS[0]
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
