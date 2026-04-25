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


class GQESLState(TypedDict, total=False):
    """LangGraph state for the GQESL pipeline.

    Fields are ``total=False`` so nodes can return partial updates.
    """

    # session
    session_id: str
    key_epoch: int               # key epoch
    counter: int                 # message counter

    # sender
    intent: dict                 # structured intent
    intent_tensor: list[float]   # 384-d vector
    projected_tensor: list[float]
    wire_packet: Optional[dict]  # cleared after verify

    # receiver
    decoded_tensor: list[float]  # 384-d vector
    rcc8_relation: str           # EQ/PO/EC/DC
    strategy: str                # coordination mode
    collapsed_concept: str       # concept id
    collapsed_similarity: float  # cosine sim

    # action
    action_result: Annotated[list[dict], merge_action_results]

    # error
    error: Optional[str]
    error_source: Optional[str]  # source node

    # task
    task_description: str        # local task text

    # key exchange
    peer_public_key: bytes
    own_public_key: bytes

    # response
    response_tensor: list[float]
    response_packet: Optional[dict]
