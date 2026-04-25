"""
GQESL A2A — LangGraph State Definition

Defines the ``GQESLState`` TypedDict that flows through all graph nodes.

SECURITY INVARIANTS:
  - NO ``session_keys`` field — keys live in ``crypto._key_registry`` only
  - NO ``messages`` field — DeepSeek LLM output is ephemeral
  - ``wire_packet`` is set to None after ``verify_node`` succeeds
  - ``action_result`` uses a custom reducer for parallel Send branches
"""

from __future__ import annotations

from typing import Annotated, Any, Optional

from typing_extensions import TypedDict


# ═══════════════════════════════════════════════════════════════════════════
# Custom Reducer for Parallel Branches
# ═══════════════════════════════════════════════════════════════════════════

def merge_action_results(
    existing: Optional[list[dict]],
    new: Optional[list[dict] | dict],
) -> list[dict]:
    """Reducer that merges action results from parallel Send branches.

    Each Send branch returns its result as a list (or single dict).
    This reducer accumulates all results into a single flat list.
    """
    if existing is None:
        existing = []
    if new is None:
        return existing
    if isinstance(new, dict):
        new = [new]
    return existing + new


# ═══════════════════════════════════════════════════════════════════════════
# GQESLState TypedDict
# ═══════════════════════════════════════════════════════════════════════════

class GQESLState(TypedDict, total=False):
    """LangGraph state for the GQESL pipeline.

    Fields are ``total=False`` so nodes can return partial updates.
    """

    # ── Session Identity ──
    session_id: str
    key_epoch: int               # Current key rotation epoch
    counter: int                 # Monotonic message counter

    # ── Sender Pipeline ──
    intent: dict                 # AgentIntent serialised as dict (no NL strings)
    intent_tensor: list[float]   # 384-dim float list
    projected_tensor: list[float]
    wire_packet: Optional[dict]  # SemanticMessage as dict — NULLED after verify

    # ── Receiver Pipeline ──
    decoded_tensor: list[float]  # 384-dim float list
    rcc8_relation: str           # 2-char relation code: "EQ", "PO", "EC", "DC"
    strategy: str                # CoordinationStrategy value
    collapsed_concept: str       # Concept ID from collapse
    collapsed_similarity: float  # Cosine similarity of collapse

    # ── Action Results (with reducer for parallel branches) ──
    action_result: Annotated[list[dict], merge_action_results]

    # ── Error Handling ──
    error: Optional[str]
    error_source: Optional[str]  # Which node produced the error

    # ── Task Description (internal, never on wire) ──
    task_description: str        # Human-readable task for DeepSeek (ephemeral use)

    # ── Peer Public Key (for key exchange) ──
    peer_public_key: bytes
    own_public_key: bytes

    # ── Response ──
    response_tensor: list[float]
    response_packet: Optional[dict]
