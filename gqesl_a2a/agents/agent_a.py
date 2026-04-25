"""
GQESL A2A — Agent A (Sender)

The sender agent uses DeepSeek for internal reasoning to construct task
intents.  All LLM output is ephemeral — only structured AgentIntent and
tensors enter the pipeline.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from gqesl_a2a import config
from gqesl_a2a.agents.session import SessionInfo, bootstrap_session
from gqesl_a2a.core.crypto import (
    compute_salt,
    get_session_keys,
    rotate_keys,
    should_rotate,
    should_terminate_session,
    sign_packet,
)
from gqesl_a2a.core.ledger import get_ledger
from gqesl_a2a.core.semantic_state import (
    SemanticMessage,
    encode_tensor,
    pack_packet,
)
from gqesl_a2a.core.tensor_builder import (
    AgentIntent,
    EntityType,
    OutputFormat,
    TaskType,
    build_basis_matrix,
    build_intent_tensor,
)

logger = logging.getLogger(__name__)


class AgentA:
    """Sender agent — constructs intents, encodes, and transmits."""

    def __init__(self, session_info: SessionInfo) -> None:
        self.session_id = session_info.session_id
        self.counter = session_info.counter

    def build_intent_from_task(self, task_description: str) -> AgentIntent:
        """Use DeepSeek to parse a task description into structured AgentIntent.

        LLM output is LOCAL ONLY — never stored or checkpointed.
        """
        try:
            from langchain_openai import ChatOpenAI
            import json

            llm = ChatOpenAI(
                model=config.DEEPSEEK_MODEL,
                api_key=config.DEEPSEEK_API_KEY,
                base_url=config.DEEPSEEK_BASE_URL,
                temperature=0.3,
            )

            prompt = (
                "Given this task, output ONLY a JSON object with keys: "
                "task_type (EXTRACT|SUMMARIZE|CLASSIFY|TRANSLATE|GENERATE|VERIFY|SEARCH|COMPARE), "
                "entity_type (PERSON|ORGANIZATION|LOCATION|EVENT|PRODUCT|DOCUMENT|CODE|DATA), "
                "output_format (JSON|TABLE|TEXT|TENSOR|BINARY|GRAPH), "
                "priority (float 0.0-1.0). "
                f"Task: {task_description}"
            )

            response = llm.invoke(prompt)  # Ephemeral — local var only
            content = response.content.strip()
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            parsed = json.loads(content)

            return AgentIntent(
                task_type=TaskType[parsed.get("task_type", "EXTRACT")],
                entity_type=EntityType[parsed.get("entity_type", "DATA")],
                output_format=OutputFormat[parsed.get("output_format", "JSON")],
                priority=float(parsed.get("priority", 0.5)),
                source_ref=os.urandom(32),
                debug_label=task_description[:50],
            )
        except Exception as e:
            logger.warning("DeepSeek intent parsing failed (%s), using defaults", e)
            return AgentIntent(
                task_type=TaskType.EXTRACT,
                entity_type=EntityType.DATA,
                output_format=OutputFormat.JSON,
                priority=0.5,
                source_ref=os.urandom(32),
            )

    def encode_and_sign(self, intent: AgentIntent) -> dict:
        """Encode an intent into a signed wire packet.

        Returns the packet dict ready for transmission.
        """
        keys = get_session_keys(self.session_id)

        # Check rotation
        if should_rotate(self.counter, keys.epoch):
            keys = rotate_keys(self.session_id)

        # Check termination
        if should_terminate_session(self.counter):
            raise RuntimeError("Session max messages reached — must re-bootstrap")

        basis = build_basis_matrix(keys.basis_seed)
        tensor = build_intent_tensor(intent, basis)
        salt = compute_salt(keys.salt_seed, self.counter)

        ledger = get_ledger()
        ledger_vecs = ledger.get_all_vectors_matrix(self.session_id)

        idx, relation, projected = encode_tensor(
            tensor, salt, keys.W1, keys.W2, keys.codebook, ledger_vecs
        )

        v = 1
        hmac_digest = sign_packet(keys.hmac_key, v, idx, relation.value, self.counter)

        packet = {
            "v": v,
            "idx": idx,
            "r": relation.value,
            "c": self.counter,
            "h": hmac_digest.hex(),
        }

        wire_msg = SemanticMessage(
            v=v, idx=idx, r=relation.value, c=self.counter, h=hmac_digest
        )
        wire_bytes = pack_packet(wire_msg)

        self.counter += 1

        logger.info(
            "Agent A encoded: idx=%d, relation=%s, counter=%d, wire_size=%d bytes",
            idx, relation.value, self.counter - 1, len(wire_bytes),
        )

        return {
            "packet": packet,
            "wire_bytes": wire_bytes,
            "intent_tensor": tensor.tolist(),
            "relation": relation.value,
        }

    def send_task(self, task_description: str) -> dict:
        """Full pipeline: parse task → build intent → encode → sign."""
        intent = self.build_intent_from_task(task_description)
        return self.encode_and_sign(intent)
