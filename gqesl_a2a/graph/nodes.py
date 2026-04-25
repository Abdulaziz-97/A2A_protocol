"""
GQESL A2A — LangGraph Node Functions

Every node follows two security invariants:
  1. Crypto keys are fetched from ``crypto.get_session_keys()`` — never from state.
  2. DeepSeek LLM calls use local variables — NL never enters state.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Literal, Optional

import numpy as np
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from gqesl_a2a import config
from gqesl_a2a.core.crypto import (
    clear_session_keys,
    compute_salt,
    compute_shared_secret,
    derive_session_keys,
    generate_keypair,
    get_session_keys,
    register_session_keys,
    rotate_keys,
    should_rotate,
    should_terminate_session,
    sign_packet,
    verify_packet,
)
from gqesl_a2a.core.ledger import get_ledger
from gqesl_a2a.core.semantic_state import (
    CoordinationStrategy,
    RCC8Relation,
    RCC8_STRATEGY_MAP,
    SemanticMessage,
    collapse_tensor,
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
from gqesl_a2a.graph.state import GQESLState

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# DeepSeek LLM (ephemeral — never stored in state)
# ═══════════════════════════════════════════════════════════════════════════

def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=config.DEEPSEEK_MODEL,
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
        temperature=0.3,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Negotiation Resume Model (Fix #6)
# ═══════════════════════════════════════════════════════════════════════════

class NegotiationResume(BaseModel):
    action: Literal["accept", "counter", "reject"]
    counter_proposal_tensor: Optional[list[float]] = None
    accept: bool = False


# ═══════════════════════════════════════════════════════════════════════════
# NODE: key_exchange_node
# ═══════════════════════════════════════════════════════════════════════════

def key_exchange_node(state: GQESLState) -> dict:
    """Run ECDH handshake and derive all session keys."""
    private_key, own_pub = generate_keypair()
    peer_pub = state.get("peer_public_key")

    if peer_pub is None:
        # First agent — generate keypair, wait for peer
        session_id = str(uuid.uuid4())
        return {
            "session_id": session_id,
            "own_public_key": own_pub,
            "counter": 0,
            "key_epoch": 0,
        }

    # Second agent or both keys available — compute shared secret
    session_id = state.get("session_id", str(uuid.uuid4()))
    shared_secret = compute_shared_secret(private_key, peer_pub)
    session_nonce = os.urandom(16)

    keys = derive_session_keys(shared_secret, session_nonce, epoch=0)
    register_session_keys(session_id, keys)

    # Register basis concepts in ledger (session bootstrap)
    ledger = get_ledger()
    basis = build_basis_matrix(keys.basis_seed)
    for i in range(min(basis.shape[0], 55)):  # Register first 55 concepts
        ledger.register(basis[i], f"concept_{i}", session_id)

    logger.info("Session %s: key exchange complete, %d concepts registered",
                session_id, ledger.concept_count(session_id))

    return {
        "session_id": session_id,
        "own_public_key": own_pub,
        "counter": 0,
        "key_epoch": 0,
    }


# ═══════════════════════════════════════════════════════════════════════════
# NODE: build_intent_node  (Fix #2: DeepSeek output is ephemeral)
# ═══════════════════════════════════════════════════════════════════════════

def build_intent_node(state: GQESLState) -> dict:
    """Use DeepSeek (locally) to reason about the task, then build tensor."""
    session_id = state["session_id"]
    keys = get_session_keys(session_id)
    task_desc = state.get("task_description", "")

    # --- DeepSeek call in LOCAL SCOPE — NL never enters state ---
    if task_desc:
        import json
        import time
        llm = _get_llm()
        prompt = (
            "Given this task, output ONLY a JSON object with keys: "
            "task_type (one of: EXTRACT,SUMMARIZE,CLASSIFY,TRANSLATE,GENERATE,VERIFY,SEARCH,COMPARE), "
            "entity_type (one of: PERSON,ORGANIZATION,LOCATION,EVENT,PRODUCT,DOCUMENT,CODE,DATA), "
            "output_format (one of: JSON,TABLE,TEXT,TENSOR,BINARY,GRAPH), "
            "priority (float 0.0-1.0). "
            f"Task: {task_desc}"
        )
        
        parsed = None
        for attempt in range(config.DEEPSEEK_MAX_RETRIES):
            try:
                response = llm.invoke(prompt)  # Local var — discarded after parsing
                content = response.content.strip()
                if "```" in content:
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                parsed = json.loads(content)
                break
            except Exception as e:
                logger.warning("DeepSeek call failed (attempt %d/%d): %s", attempt + 1, config.DEEPSEEK_MAX_RETRIES, e)
                if attempt < config.DEEPSEEK_MAX_RETRIES - 1:
                    time.sleep(config.DEEPSEEK_RETRY_BACKOFF ** attempt)
        
        if parsed:
            task_type = TaskType[parsed.get("task_type", "EXTRACT")]
            entity_type = EntityType[parsed.get("entity_type", "DATA")]
            output_format = OutputFormat[parsed.get("output_format", "JSON")]
            priority = float(parsed.get("priority", 0.5))
        else:
            logger.error("DeepSeek parsing failed after retries, using defaults")
            task_type = TaskType.EXTRACT
            entity_type = EntityType.DATA
            output_format = OutputFormat.JSON
            priority = 0.5
    else:
        # Use intent from state if provided
        intent_dict = state.get("intent", {})
        task_type = TaskType[intent_dict.get("task_type", "EXTRACT")]
        entity_type = EntityType[intent_dict.get("entity_type", "DATA")]
        output_format = OutputFormat[intent_dict.get("output_format", "JSON")]
        priority = float(intent_dict.get("priority", 0.5))

    intent = AgentIntent(
        task_type=task_type,
        entity_type=entity_type,
        output_format=output_format,
        priority=priority,
        source_ref=os.urandom(32),
    )

    basis = build_basis_matrix(keys.basis_seed)
    tensor = build_intent_tensor(intent, basis)

    return {
        "intent": {
            "task_type": intent.task_type.name,
            "entity_type": intent.entity_type.name,
            "output_format": intent.output_format.name,
            "priority": intent.priority,
        },
        "intent_tensor": tensor.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# NODE: encode_node  (checks rotation + termination)
# ═══════════════════════════════════════════════════════════════════════════

def encode_node(state: GQESLState) -> dict:
    """Salt → project → quantise → RCC-8."""
    session_id = state["session_id"]
    counter = state.get("counter", 0)

    # Check session termination
    if should_terminate_session(counter):
        # Generate the termination packet (idx=4095)
        logger.info("Session max messages reached. Sending termination packet.")
        return {
            "projected_tensor": np.zeros(384, dtype=np.float32).tolist(),
            "rcc8_relation": "DC",
            "key_epoch": keys.epoch,
            "_is_termination_packet": True,
        }

    # Check key rotation
    keys = get_session_keys(session_id)
    if should_rotate(counter, keys.epoch):
        keys = rotate_keys(session_id)
        logger.info("Key rotation triggered at counter %d → epoch %d", counter, keys.epoch)

    intent_tensor = np.array(state["intent_tensor"], dtype=np.float32)
    salt = compute_salt(keys.salt_seed, counter)

    ledger = get_ledger()
    ledger_vecs = ledger.get_all_vectors_matrix(session_id)

    idx, relation, projected = encode_tensor(
        intent_tensor, salt, keys.W1, keys.W2, keys.codebook, ledger_vecs
    )

    return {
        "projected_tensor": projected.tolist(),
        "rcc8_relation": relation.value,
        "key_epoch": keys.epoch,
    }


# ═══════════════════════════════════════════════════════════════════════════
# NODE: sign_node
# ═══════════════════════════════════════════════════════════════════════════

def sign_node(state: GQESLState) -> dict:
    """HMAC sign the wire packet."""
    session_id = state["session_id"]
    keys = get_session_keys(session_id)
    counter = state.get("counter", 0)

    intent_tensor = np.array(state["intent_tensor"], dtype=np.float32)
    salt = compute_salt(keys.salt_seed, counter)
    ledger = get_ledger()
    ledger_vecs = ledger.get_all_vectors_matrix(session_id)

    is_term = state.get("_is_termination_packet", False)
    if is_term:
        idx = 4095
        relation = RCC8Relation.DC
    else:
        idx, relation, _ = encode_tensor(
            intent_tensor, salt, keys.W1, keys.W2, keys.codebook, ledger_vecs
        )

    v = 1
    hmac_digest = sign_packet(keys.hmac_key, v, idx, relation.value, counter)

    packet = SemanticMessage(v=v, idx=idx, r=relation.value, c=counter, h=hmac_digest)

    return {
        "wire_packet": {
            "idx": packet.idx,
            "r": packet.r,
            "c": packet.c,
            "h": packet.h.hex(),
        },
        "counter": counter + 1,
    }


# ═══════════════════════════════════════════════════════════════════════════
# NODE: verify_node  (Fix #9: nulls wire_packet after success)
# ═══════════════════════════════════════════════════════════════════════════

def verify_node(state: GQESLState) -> dict:
    """Verify HMAC and counter freshness.  Nulls wire_packet on success."""
    session_id = state["session_id"]
    keys = get_session_keys(session_id)
    packet_dict = state.get("wire_packet")

    if packet_dict is None:
        return {"error": "no_wire_packet", "error_source": "verify_node"}

    v = packet_dict.get("v", 1)
    idx = packet_dict["idx"]
    r = packet_dict["r"]
    c = packet_dict["c"]
    h = bytes.fromhex(packet_dict["h"])

    # HMAC verification
    if not verify_packet(keys.hmac_key, v, idx, r, c, h):
        logger.error("HMAC verification FAILED for counter %d", c)
        return {
            "error": "hmac_verification_failed",
            "error_source": "verify_node",
            "wire_packet": None,
        }

    # Counter freshness (must be >= current counter)
    current_counter = state.get("counter", 0)
    # Check teardown buffer (Item 2)
    if c > config.SESSION_MAX_MESSAGES + config.SESSION_TEARDOWN_BUFFER:
        logger.error("Counter exceeded hard teardown buffer")
        return {"error": "session_max_messages_reached", "error_source": "verify_node", "wire_packet": None}
        
    if c < current_counter:
        logger.error("Counter replay detected: received %d, expected >= %d", c, current_counter)
        return {
            "error": "counter_replay",
            "error_source": "verify_node",
            "wire_packet": None,
        }

    # Termination packet detection (idx=4095)
    if idx == 4095:
        logger.info("Termination packet received (idx=4095) at counter %d", c)
        return {
            "error": "session_terminated_by_peer",
            "error_source": "verify_node",
            "wire_packet": None,
        }

    # Success — null wire_packet to prevent known-plaintext exposure in checkpoint
    return {
        "rcc8_relation": r,
        "counter": c + 1,
        "wire_packet": None,  # Fix #9
        "error": None,
        "error_source": None,
    }


# ═══════════════════════════════════════════════════════════════════════════
# NODE: decode_node  (Fix #1: W.T inversion)
# ═══════════════════════════════════════════════════════════════════════════

def decode_node(state: GQESLState) -> dict:
    """Codebook lookup → reverse project using W.T → desalt."""
    session_id = state["session_id"]
    keys = get_session_keys(session_id)
    packet_dict = state.get("wire_packet") or {}

    # Get idx from the packet that was verified (we saved relation already)
    # We need to retrieve idx before it was nulled — it's passed separately
    idx = state.get("_decoded_idx")
    if idx is None:
        # Fallback: idx might still be available
        idx = packet_dict.get("idx", 0)

    counter = state.get("counter", 1) - 1  # Counter was incremented in verify
    salt = compute_salt(keys.salt_seed, counter)

    reconstructed = decode_tensor(idx, keys.codebook, salt, keys.W1, keys.W2)

    return {
        "decoded_tensor": reconstructed.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# NODE: rcc8_route_node
# ═══════════════════════════════════════════════════════════════════════════

def rcc8_route_node(state: GQESLState) -> dict:
    """Read RCC-8 relation and set coordination strategy."""
    relation_str = state.get("rcc8_relation", "DC")
    relation = RCC8Relation(relation_str)
    strategy = RCC8_STRATEGY_MAP[relation]
    return {"strategy": strategy.value}


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY NODES
# ═══════════════════════════════════════════════════════════════════════════

def exact_dispatch_node(state: GQESLState) -> dict:
    """EQ strategy: execute the decoded task as-is."""
    logger.info("EXACT_MATCH: executing decoded intent directly")
    return {}


def parallel_split_node(state: GQESLState) -> dict:
    """PO strategy: split into parallel sub-tasks."""
    logger.info("SPLIT_EXECUTION: splitting into parallel sub-tasks")
    return {"action_result": [{"strategy": "split", "status": "delegated"}]}


def handoff_node(state: GQESLState) -> dict:
    """EC strategy: clean handoff to receiving agent."""
    logger.info("FULL_HANDOFF: handing off to receiver")
    return {}


def negotiate_node(state: GQESLState) -> dict:
    """DC strategy: concepts disconnected, negotiation required.

    Uses LangGraph interrupt() to pause and wait for a NegotiationResume.
    """
    from langgraph.types import interrupt

    logger.info("NEGOTIATE_FIRST: pausing for negotiation")

    resume_data = interrupt({
        "reason": "concepts_disconnected",
        "relation": state.get("rcc8_relation", "DC"),
        "expected_format": {
            "action": "accept | counter | reject",
            "counter_proposal_tensor": "list[float] (384-dim) if action=='counter'",
            "accept": "bool",
        },
    })

    resume = NegotiationResume(**resume_data)

    if resume.action == "accept":
        return {}
    elif resume.action == "counter":
        if resume.counter_proposal_tensor:
            return {"decoded_tensor": resume.counter_proposal_tensor}
        return {}
    else:  # reject
        return {
            "error": "negotiation_rejected",
            "error_source": "negotiate_node",
        }


# ═══════════════════════════════════════════════════════════════════════════
# NODE: collapse_node
# ═══════════════════════════════════════════════════════════════════════════

def collapse_node(state: GQESLState) -> dict:
    """Collapse decoded tensor to nearest concept via argmax."""
    session_id = state["session_id"]
    keys = get_session_keys(session_id)

    decoded = np.array(state["decoded_tensor"], dtype=np.float32)
    basis = build_basis_matrix(keys.basis_seed)

    concept_idx, similarity = collapse_tensor(decoded, basis)

    # Record usage for drift tracking
    ledger = get_ledger()
    concept_id = f"concept_{concept_idx}"
    ledger.record_usage(concept_id, session_id, decoded)

    return {
        "collapsed_concept": concept_id,
        "collapsed_similarity": similarity,
    }


# ═══════════════════════════════════════════════════════════════════════════
# NODE: execute_node  (Fix #2: DeepSeek ephemeral)
# ═══════════════════════════════════════════════════════════════════════════

def execute_node(state: GQESLState) -> dict:
    """Agent B executes the decoded task.  DeepSeek output is local-scope only."""
    session_id = state["session_id"]
    keys = get_session_keys(session_id)
    basis = build_basis_matrix(keys.basis_seed)
    decoded = np.array(state["decoded_tensor"], dtype=np.float32)

    intent_info = collapse_to_intent(decoded, basis)

    # --- DeepSeek call (ephemeral — local scope only) ---
    try:
        llm = _get_llm()
        prompt = (
            f"Execute this task:\n"
            f"  Task type: {intent_info['task_type'].name}\n"
            f"  Entity type: {intent_info['entity_type'].name}\n"
            f"  Output format: {intent_info['output_format'].name}\n"
            f"  Priority: {intent_info['priority']:.2f}\n"
            f"Provide a brief structured result."
        )
        response = llm.invoke(prompt)  # Local var — never in state
        result_text = response.content[:200]  # Truncate for safety
    except Exception as e:
        logger.warning("DeepSeek execution failed: %s", e)
        result_text = "execution_completed"

    # Only structured data enters state
    return {
        "action_result": [{
            "concept": state.get("collapsed_concept", "unknown"),
            "task_type": intent_info["task_type"].name,
            "entity_type": intent_info["entity_type"].name,
            "status": "completed",
            "similarity": state.get("collapsed_similarity", 0.0),
        }],
    }


# ═══════════════════════════════════════════════════════════════════════════
# NODE: respond_node
# ═══════════════════════════════════════════════════════════════════════════

def respond_node(state: GQESLState) -> dict:
    """Build a response tensor and prepare it for sending back."""
    session_id = state["session_id"]
    keys = get_session_keys(session_id)

    # Build a simple acknowledgment tensor
    basis = build_basis_matrix(keys.basis_seed)
    ack_intent = AgentIntent(
        task_type=TaskType.VERIFY,
        entity_type=EntityType.DATA,
        output_format=OutputFormat.JSON,
        priority=0.9,
        source_ref=b"\x00" * 32,
    )
    response_tensor = build_intent_tensor(ack_intent, basis)

    return {"response_tensor": response_tensor.tolist()}


# ═══════════════════════════════════════════════════════════════════════════
# NODE: drift_monitor_node
# ═══════════════════════════════════════════════════════════════════════════

def drift_monitor_node(state: GQESLState) -> dict:
    """Check all concepts for drift, trigger re-negotiation if needed."""
    session_id = state["session_id"]
    ledger = get_ledger()

    drifting = ledger.get_drifting_concepts(session_id)
    for concept_id, score in drifting:
        logger.warning("Drift detected: concept '%s' score=%.4f — re-negotiating",
                       concept_id, score)
        ledger.update_centroid(concept_id, session_id)

    return {}


# ═══════════════════════════════════════════════════════════════════════════
# NODE: error_handler_node
# ═══════════════════════════════════════════════════════════════════════════

def error_handler_node(state: GQESLState) -> dict:
    """Handle errors from verify, negotiate, or encode nodes."""
    error = state.get("error", "unknown")
    source = state.get("error_source", "unknown")

    logger.error("Error in %s: %s", source, error)

    if error == "hmac_verification_failed":
        logger.critical("HMAC FAILURE — packet discarded, alerting")
    elif error == "counter_replay":
        logger.critical("COUNTER REPLAY — session may need re-initialization")
    elif error == "session_max_messages_reached":
        session_id = state.get("session_id", "")
        clear_session_keys(session_id)
        logger.info("Session %s terminated (max messages reached)", session_id)
    elif error == "negotiation_rejected":
        logger.warning("Negotiation rejected — task abandoned")

    return {"error": None, "error_source": None}


# ═══════════════════════════════════════════════════════════════════════════
# NODE: counter_sync_node
# ═══════════════════════════════════════════════════════════════════════════

def counter_sync_node(state: GQESLState) -> dict:
    """Synchronise counters on reconnect.

    Both agents exchange signed counters and advance to max(a, b) + 1.
    """
    session_id = state["session_id"]
    keys = get_session_keys(session_id)
    local_counter = state.get("counter", 0)

    # Sign our counter
    sync_hmac = sign_packet(keys.hmac_key, 1, 0, "EQ", local_counter)

    # In a real deployment, this would exchange via MessageBus.
    # For now, we just set the counter (peer counter comes from state).
    peer_counter = state.get("_peer_counter", local_counter)
    synced = max(local_counter, peer_counter) + 1

    logger.info("Counter synced: local=%d, peer=%d → %d",
                local_counter, peer_counter, synced)

    return {"counter": synced}


# ═══════════════════════════════════════════════════════════════════════════
# NODE: session_terminate_node  (Issue #12)
# ═══════════════════════════════════════════════════════════════════════════

def session_terminate_node(state: GQESLState) -> dict:
    """Tear down session when SESSION_MAX_MESSAGES reached."""
    session_id = state.get("session_id", "")
    clear_session_keys(session_id)
    ledger = get_ledger()
    ledger.clear_session(session_id)
    logger.info("Session %s fully terminated", session_id)
    return {"error": "session_terminated", "error_source": "session_terminate_node"}
