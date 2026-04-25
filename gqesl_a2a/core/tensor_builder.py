"""
GQESL A2A — Intent Tensor Builder

Constructs 384-dimensional intent tensors from structured AgentIntent
dataclasses.  No natural language input is accepted — the tensor is built
as a weighted sum of concept basis vectors derived from HKDF seed.

The basis matrix is session-private (derived from ``b"gqesl-basis"``
HKDF seed) so even the concept-to-dimension mapping is secret.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

from gqesl_a2a.config import BASIS_CONCEPTS, TENSOR_DIM

logger = logging.getLogger(__name__)


class TaskType(IntEnum):
    EXTRACT = 0
    SUMMARIZE = 1
    CLASSIFY = 2
    TRANSLATE = 3
    GENERATE = 4
    VERIFY = 5
    SEARCH = 6
    COMPARE = 7


class EntityType(IntEnum):
    PERSON = 0
    ORGANIZATION = 1
    LOCATION = 2
    EVENT = 3
    PRODUCT = 4
    DOCUMENT = 5
    CODE = 6
    DATA = 7


class OutputFormat(IntEnum):
    JSON = 0
    TABLE = 1
    TEXT = 2
    TENSOR = 3
    BINARY = 4
    GRAPH = 5


@dataclass(frozen=True)
class AgentIntent:
    """Structured representation of an agent's task intent.

    No string fields (except optional debug_label which is NEVER encoded
    or transmitted).
    """

    task_type: TaskType
    entity_type: EntityType
    output_format: OutputFormat
    priority: float               # 0..1
    source_ref: bytes             # 32-byte hash
    debug_label: Optional[str] = None  # local only


class AgentIntentSchema(BaseModel):
    """Pydantic model for safe structured JSON extraction from LLMs."""
    
    task_type: TaskType
    entity_type: EntityType
    output_format: OutputFormat
    priority: float = Field(ge=0.0, le=1.0, description="Priority from 0.0 to 1.0")
    source_ref_hex: str = Field(description="64-character hex string representing a SHA-256 hash")
    debug_label: Optional[str] = None

    def to_intent(self) -> AgentIntent:
        """Convert the validated schema into the core dataclass."""
        try:
            ref_bytes = bytes.fromhex(self.source_ref_hex)
        except ValueError:
            ref_bytes = b"\x00" * 32
            
        if len(ref_bytes) != 32:
            ref_bytes = ref_bytes[:32].ljust(32, b"\x00")
            
        return AgentIntent(
            task_type=self.task_type,
            entity_type=self.entity_type,
            output_format=self.output_format,
            priority=self.priority,
            source_ref=ref_bytes,
            debug_label=self.debug_label,
        )


def build_basis_matrix(
    seed: bytes,
    n_concepts: int = BASIS_CONCEPTS,
    dim: int = TENSOR_DIM,
) -> np.ndarray:
    """Generate a deterministic, orthogonal concept basis matrix.

    Uses QR decomposition to produce orthogonal basis vectors from the
    ``b"gqesl-basis"`` HKDF-derived seed.  The resulting matrix has shape
    ``(n_concepts, dim)`` where each row is a unit-length basis vector.

    Since ``n_concepts`` (256) < ``dim`` (384), we generate a full
    ``dim × dim`` orthogonal matrix and take the first ``n_concepts`` rows.
    """
    rng = np.random.default_rng(
        int.from_bytes(hashlib.sha256(seed).digest()[:8], "big")
    )
    random_matrix = rng.standard_normal((dim, dim)).astype(np.float64)
    Q, R = np.linalg.qr(random_matrix)
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    Q = Q * signs[np.newaxis, :]
    basis = Q[:n_concepts].astype(np.float32)

    # unit check
    norms = np.linalg.norm(basis, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), "Basis vectors must be unit length"

    logger.debug("Built basis matrix: (%d, %d) from seed", n_concepts, dim)
    return basis


# index map

_TASK_OFFSET = 0
_ENTITY_OFFSET = len(TaskType)
_OUTPUT_OFFSET = _ENTITY_OFFSET + len(EntityType)
_PRIORITY_IDX = _OUTPUT_OFFSET + len(OutputFormat)
_SOURCE_OFFSET = _PRIORITY_IDX + 1
_SOURCE_DIMS = 32  # byte dims


def build_intent_tensor(
    intent: AgentIntent,
    basis_matrix: np.ndarray,
) -> np.ndarray:
    """Construct a 384-dim intent tensor as a weighted sum of basis vectors.

    The tensor is a superposition:
        tensor = w_task * B[task_idx]
               + w_entity * B[entity_idx]
               + w_output * B[output_idx]
               + priority * B[priority_idx]
               + sum_over_bytes( byte_weight * B[source_idx + j] )

    All weights are positive and the result is L2-normalised.
    """
    tensor = np.zeros(TENSOR_DIM, dtype=np.float32)

    # task
    task_idx = _TASK_OFFSET + int(intent.task_type)
    tensor += 1.0 * basis_matrix[task_idx]

    # entity
    entity_idx = _ENTITY_OFFSET + int(intent.entity_type)
    tensor += 1.0 * basis_matrix[entity_idx]

    # format
    output_idx = _OUTPUT_OFFSET + int(intent.output_format)
    tensor += 0.7 * basis_matrix[output_idx]

    # priority
    tensor += intent.priority * basis_matrix[_PRIORITY_IDX]

    # source ref
    for j in range(min(_SOURCE_DIMS, len(intent.source_ref))):
        byte_weight = intent.source_ref[j] / 255.0 * 0.3
        src_idx = _SOURCE_OFFSET + j
        if src_idx < basis_matrix.shape[0]:
            tensor += byte_weight * basis_matrix[src_idx]

    # normalize
    norm = np.linalg.norm(tensor)
    if norm > 0:
        tensor = tensor / norm

    return tensor


def collapse_to_intent(
    tensor: np.ndarray,
    basis_matrix: np.ndarray,
) -> dict:
    """Collapse a reconstructed tensor back to structured intent components.

    Returns a dict with the most likely task_type, entity_type, output_format
    by computing cosine similarity against each concept's basis vector.
    """
    # normalize
    norm = np.linalg.norm(tensor)
    if norm > 0:
        tensor = tensor / norm

    # task argmax
    task_sims = np.array([
        _cosine_sim(tensor, basis_matrix[_TASK_OFFSET + i])
        for i in range(len(TaskType))
    ])
    task_type = TaskType(int(np.argmax(task_sims)))

    # entity argmax
    entity_sims = np.array([
        _cosine_sim(tensor, basis_matrix[_ENTITY_OFFSET + i])
        for i in range(len(EntityType))
    ])
    entity_type = EntityType(int(np.argmax(entity_sims)))

    # format argmax
    output_sims = np.array([
        _cosine_sim(tensor, basis_matrix[_OUTPUT_OFFSET + i])
        for i in range(len(OutputFormat))
    ])
    output_format = OutputFormat(int(np.argmax(output_sims)))

    # priority proj
    priority = float(np.clip(
        _cosine_sim(tensor, basis_matrix[_PRIORITY_IDX]), 0.0, 1.0
    ))

    return {
        "task_type": task_type,
        "entity_type": entity_type,
        "output_format": output_format,
        "priority": priority,
    }


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    dot = float(np.dot(a, b))
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
