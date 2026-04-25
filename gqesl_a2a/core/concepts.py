"""
Shared vocabulary for ledger warming.

Both agents derive identical tensors for the same AgentIntent under the same
session keys, so warming these entries at bootstrap yields synchronized RCC-8
context without extra wire traffic.

Vectors registered in the ledger for RCC-8 live in the same space as
encode_tensor's normalised (pre-salt) intent step — not the salted projection.
"""

from __future__ import annotations

from gqesl_a2a.core.tensor_builder import AgentIntent, EntityType, OutputFormat, TaskType


def warm_concept_id(intent: AgentIntent) -> str:
    """Stable id for a warmed concept (projected / intent-template row)."""
    return (
        f"warm_{intent.task_type.name}_"
        f"{intent.entity_type.name}_{intent.output_format.name}"
    )


# deduped templates

KNOWN_CONCEPTS: list[AgentIntent] = [
    AgentIntent(
        TaskType.EXTRACT, EntityType.PERSON, OutputFormat.JSON,
        1.0, b"\x00" * 32,
    ),
    AgentIntent(
        TaskType.SUMMARIZE, EntityType.DOCUMENT, OutputFormat.TEXT,
        0.8, b"\x00" * 32,
    ),
    AgentIntent(
        TaskType.CLASSIFY, EntityType.DATA, OutputFormat.TABLE,
        0.6, b"\x00" * 32,
    ),
    AgentIntent(
        TaskType.CLASSIFY, EntityType.ORGANIZATION, OutputFormat.TABLE,
        0.8, b"\x00" * 32,
    ),
    AgentIntent(
        TaskType.SEARCH, EntityType.ORGANIZATION, OutputFormat.JSON,
        0.9, b"\x00" * 32,
    ),
    AgentIntent(
        TaskType.SEARCH, EntityType.LOCATION, OutputFormat.JSON,
        0.9, b"\x00" * 32,
    ),
    AgentIntent(
        TaskType.GENERATE, EntityType.CODE, OutputFormat.TEXT,
        0.95, b"\x00" * 32,
    ),
    AgentIntent(
        TaskType.VERIFY, EntityType.DATA, OutputFormat.JSON,
        0.6, b"\x00" * 32,
    ),
    AgentIntent(
        TaskType.TRANSLATE, EntityType.DOCUMENT, OutputFormat.TEXT,
        0.4, b"\x00" * 32,
    ),
    AgentIntent(
        TaskType.COMPARE, EntityType.PRODUCT, OutputFormat.TABLE,
        0.75, b"\x00" * 32,
    ),
    AgentIntent(
        TaskType.EXTRACT, EntityType.DATA, OutputFormat.JSON,
        0.5, b"\x00" * 32,
    ),
]
