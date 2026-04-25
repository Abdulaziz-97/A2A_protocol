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
    vector_stats,
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

    def verify_and_decode(self, packet: dict, *, with_workflow: bool = False) -> dict | None:
        """Verify HMAC, check counter, decode wire packet.

        Returns decoded info dict or None on failure.
        If ``with_workflow`` is True, successful returns include ``workflow`` with
        receiver-side steps (verify → decode transforms → collapse → strategy).
        """
        workflow: list[dict] = [] if with_workflow else []

        keys = get_session_keys(self.session_id)

        v = packet.get("v", 1)
        idx = packet["idx"]
        r = packet["r"]
        c = packet["c"]
        h = bytes.fromhex(packet["h"])

        # verify HMAC
        if not verify_packet(keys.hmac_key, v, idx, r, c, h):
            logger.error("HMAC verification FAILED — packet discarded")
            return None

        if with_workflow:
            workflow.append({
                "step": "receiver.hmac_verify",
                "detail": "HMAC-SHA256 over (v, idx, r, c) matches; packet authentic",
            })

        # fresh counter
        if c < self.counter:
            logger.error("Counter replay: got %d, expected >= %d", c, self.counter)
            return None

        if with_workflow:
            workflow.append({
                "step": "receiver.counter_fresh",
                "detail": f"c={c} >= local counter {self.counter} (anti-replay)",
            })

        # decode
        salt = compute_salt(keys.salt_seed, c)
        if with_workflow:
            workflow.append({
                "step": "receiver.salt",
                "detail": f"same HKDF salt as sender for c={c}",
                "tensor_stats": vector_stats(salt),
            })

        decode_trace = workflow if with_workflow else None
        reconstructed = decode_tensor(
            idx, keys.codebook, salt, keys.W1, keys.W2, trace=decode_trace
        )

        # collapse
        basis = build_basis_matrix(keys.basis_seed)
        concept_idx, similarity = collapse_tensor(reconstructed, basis)

        if with_workflow:
            workflow.append({
                "step": "receiver.collapse",
                "detail": f"argmax cosine vs basis -> concept_{concept_idx}, sim={similarity:.4f}",
                "tensor_stats": vector_stats(reconstructed),
            })

        # track drift
        ledger = get_ledger()
        concept_id = f"concept_{concept_idx}"
        ledger.record_usage(concept_id, self.session_id, reconstructed)

        # strategy
        relation = RCC8Relation(r)
        strategy = RCC8_STRATEGY_MAP[relation]

        if with_workflow:
            workflow.append({
                "step": "receiver.rcc8_strategy",
                "detail": f"wire relation {relation.value} -> coordination {strategy.value}",
            })
            workflow.append({
                "step": "receiver.counter_advance",
                "detail": f"local counter set to c+1 = {c + 1}",
            })

        self.counter = c + 1

        logger.info(
            "Agent B decoded: concept=%s (sim=%.3f), relation=%s, strategy=%s",
            concept_id, similarity, relation.value, strategy.value,
        )

        out: dict = {
            "decoded_tensor": reconstructed.tolist(),
            "concept_id": concept_id,
            "concept_idx": concept_idx,
            "similarity": similarity,
            "relation": relation.value,
            "strategy": strategy.value,
        }
        if with_workflow:
            out["workflow"] = workflow
        return out

    def execute_task(self, decode_result: dict) -> dict:
        """Execute the decoded task using DeepSeek (ephemeral).

        Returns structured ActionResult — no NL in output.
        """
        intent_info = None
        keys = get_session_keys(self.session_id)
        basis = build_basis_matrix(keys.basis_seed)
        decoded = np.array(decode_result["decoded_tensor"], dtype=np.float32)
        intent_info = collapse_to_intent(decoded, basis)

        # local LLM
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
            response = llm.invoke(prompt)  # local only
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
