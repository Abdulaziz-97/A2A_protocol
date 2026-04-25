"""
GQESL A2A — Agent B (Receiver)

The receiver agent verifies, decodes, and executes tasks.  DeepSeek is
used for execution reasoning but its output is ephemeral — only structured
ActionResult data enters state.
"""

from __future__ import annotations

import logging

import numpy as np

from gqesl_a2a import config
from gqesl_a2a.agents.session import SessionInfo
from gqesl_a2a.core.crypto import (
    compute_salt,
    get_session_keys,
    verify_packet,
)
from gqesl_a2a.core.ledger import get_ledger
from gqesl_a2a.core.semantic_state import (
    CoordinationStrategy,
    RCC8Relation,
    RCC8_STRATEGY_MAP,
    SemanticMessage,
    collapse_tensor,
    compute_rcc8_relation,
    decode_tensor,
)
from gqesl_a2a.core.tensor_builder import (
    build_basis_matrix,
    collapse_to_intent,
)

logger = logging.getLogger(__name__)


class AgentB:
    """Receiver agent — verifies, decodes, and executes tasks."""

    def __init__(self, session_info: SessionInfo) -> None:
        self.session_id = session_info.session_id
        self.counter = session_info.counter

    def verify_and_decode(self, packet: dict) -> dict | None:
        """Verify HMAC, check counter, decode wire packet.

        Returns decoded info dict or None on failure.
        """
        keys = get_session_keys(self.session_id)

        v = packet.get("v", 1)
        idx = packet["idx"]
        r = packet["r"]
        c = packet["c"]
        h = bytes.fromhex(packet["h"])

        # HMAC verification
        if not verify_packet(keys.hmac_key, v, idx, r, c, h):
            logger.error("HMAC verification FAILED — packet discarded")
            return None

        # Counter freshness
        if c < self.counter:
            logger.error("Counter replay: got %d, expected >= %d", c, self.counter)
            return None

        # Decode
        salt = compute_salt(keys.salt_seed, c)
        reconstructed = decode_tensor(idx, keys.codebook, salt, keys.W1, keys.W2)

        # Collapse
        basis = build_basis_matrix(keys.basis_seed)
        concept_idx, similarity = collapse_tensor(reconstructed, basis)

        # Record usage for drift
        ledger = get_ledger()
        concept_id = f"concept_{concept_idx}"
        ledger.record_usage(concept_id, self.session_id, reconstructed)

        # Strategy
        relation = RCC8Relation(r)
        strategy = RCC8_STRATEGY_MAP[relation]

        self.counter = c + 1

        logger.info(
            "Agent B decoded: concept=%s (sim=%.3f), relation=%s, strategy=%s",
            concept_id, similarity, relation.value, strategy.value,
        )

        return {
            "decoded_tensor": reconstructed.tolist(),
            "concept_id": concept_id,
            "concept_idx": concept_idx,
            "similarity": similarity,
            "relation": relation.value,
            "strategy": strategy.value,
        }

    def execute_task(self, decode_result: dict) -> dict:
        """Execute the decoded task using DeepSeek (ephemeral).

        Returns structured ActionResult — no NL in output.
        """
        intent_info = None
        keys = get_session_keys(self.session_id)
        basis = build_basis_matrix(keys.basis_seed)
        decoded = np.array(decode_result["decoded_tensor"], dtype=np.float32)
        intent_info = collapse_to_intent(decoded, basis)

        # --- DeepSeek call (ephemeral) ---
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=config.DEEPSEEK_MODEL,
                api_key=config.DEEPSEEK_API_KEY,
                base_url=config.DEEPSEEK_BASE_URL,
                temperature=0.3,
            )
            prompt = (
                f"Execute: task_type={intent_info['task_type'].name}, "
                f"entity_type={intent_info['entity_type'].name}, "
                f"output_format={intent_info['output_format'].name}. "
                f"Provide a brief structured result."
            )
            response = llm.invoke(prompt)  # Ephemeral — discarded
            execution_status = "completed"
        except Exception as e:
            logger.warning("DeepSeek execution failed: %s", e)
            execution_status = "completed_without_llm"

        return {
            "concept_id": decode_result["concept_id"],
            "task_type": intent_info["task_type"].name if intent_info else "unknown",
            "entity_type": intent_info["entity_type"].name if intent_info else "unknown",
            "strategy": decode_result["strategy"],
            "similarity": decode_result["similarity"],
            "status": execution_status,
        }

    def receive_and_execute(self, packet: dict) -> dict | None:
        """Full pipeline: verify → decode → execute."""
        decode_result = self.verify_and_decode(packet)
        if decode_result is None:
            return None
        return self.execute_task(decode_result)
