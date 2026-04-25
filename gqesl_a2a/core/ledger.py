"""
GQESL A2A — Semantic Ledger

Session-scoped vector store for registered concept vectors.  Supports
concept registration, nearest-neighbour search, drift scoring, and
centroid updates for re-negotiation.

Two backends:
  - SemanticLedger:  In-memory dict-based (default, fast, no dependencies)
  - LanceDBLedger:   LanceDB-backed (optional, persistent, async-wrapped)

LanceDB async decision: sync API internally, ``run_in_executor`` wrapper
for async LangGraph nodes in production.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from gqesl_a2a.config import DRIFT_HISTORY_K, DRIFT_THRESHOLD, TENSOR_DIM

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ConceptEntry:
    """A single concept registered in the ledger."""
    concept_id: str
    session_id: str
    vector: np.ndarray                     # (TENSOR_DIM,)
    usage_history: list[np.ndarray] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# In-Memory Semantic Ledger (default backend)
# ═══════════════════════════════════════════════════════════════════════════

class SemanticLedger:
    """In-memory concept vector store, keyed by session_id.

    Thread-safe for single-threaded LangGraph execution (no locks needed
    because LangGraph nodes run sequentially within a graph invocation).
    """

    def __init__(self) -> None:
        # session_id → concept_id → ConceptEntry
        self._store: dict[str, dict[str, ConceptEntry]] = defaultdict(dict)

    def register(
        self,
        concept_vector: np.ndarray,
        concept_id: str,
        session_id: str,
    ) -> None:
        """Register or update a concept vector in the ledger."""
        concept_vector = concept_vector.astype(np.float32).copy()
        norm = np.linalg.norm(concept_vector)
        if norm > 0:
            concept_vector = concept_vector / norm

        if concept_id in self._store[session_id]:
            # Update existing
            entry = self._store[session_id][concept_id]
            entry.vector = concept_vector
            entry.usage_history.append(concept_vector.copy())
        else:
            entry = ConceptEntry(
                concept_id=concept_id,
                session_id=session_id,
                vector=concept_vector,
                usage_history=[concept_vector.copy()],
            )
            self._store[session_id][concept_id] = entry

        logger.debug("Registered concept '%s' in session %s", concept_id, session_id)

    def search(
        self,
        query_vector: np.ndarray,
        session_id: str,
        k: int = 1,
    ) -> list[tuple[str, float]]:
        """Find the k nearest concepts by cosine similarity.

        Returns list of (concept_id, similarity) tuples, sorted descending.
        """
        concepts = self._store.get(session_id, {})
        if not concepts:
            return []

        query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-12)

        results = []
        for cid, entry in concepts.items():
            sim = float(np.dot(query_norm, entry.vector))
            results.append((cid, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def get_all(self, session_id: str) -> dict[str, np.ndarray]:
        """Return all concept vectors for a session."""
        concepts = self._store.get(session_id, {})
        return {cid: entry.vector for cid, entry in concepts.items()}

    def get_all_vectors_matrix(self, session_id: str) -> Optional[np.ndarray]:
        """Return all concept vectors as a matrix (N, TENSOR_DIM).

        Returns None if no concepts are registered.
        """
        vectors = self.get_all(session_id)
        if not vectors:
            return None
        return np.stack(list(vectors.values()))

    def record_usage(
        self,
        concept_id: str,
        session_id: str,
        usage_vector: np.ndarray,
    ) -> None:
        """Record a usage vector for drift tracking."""
        concepts = self._store.get(session_id, {})
        if concept_id not in concepts:
            logger.warning("Cannot record usage for unknown concept '%s'", concept_id)
            return

        entry = concepts[concept_id]
        usage_vector = usage_vector.astype(np.float32).copy()
        norm = np.linalg.norm(usage_vector)
        if norm > 0:
            usage_vector = usage_vector / norm
        entry.usage_history.append(usage_vector)

        # Trim history
        if len(entry.usage_history) > DRIFT_HISTORY_K * 2:
            entry.usage_history = entry.usage_history[-DRIFT_HISTORY_K:]

    def drift_score(self, concept_id: str, session_id: str) -> float:
        """Compute drift score as cosine variance of last K usage vectors.

        Returns 0.0 if insufficient history.  Higher values indicate more drift.
        """
        concepts = self._store.get(session_id, {})
        if concept_id not in concepts:
            return 0.0

        history = concepts[concept_id].usage_history
        if len(history) < 2:
            return 0.0

        recent = history[-DRIFT_HISTORY_K:]
        if len(recent) < 2:
            return 0.0

        # Compute pairwise cosine similarities with the centroid
        centroid = np.mean(recent, axis=0)
        c_norm = np.linalg.norm(centroid)
        if c_norm < 1e-12:
            return 0.0
        centroid = centroid / c_norm

        sims = np.array([
            float(np.dot(v / (np.linalg.norm(v) + 1e-12), centroid))
            for v in recent
        ])

        # Variance of cosine similarities — higher means more drift
        return float(np.var(sims))

    def update_centroid(
        self,
        concept_id: str,
        session_id: str,
        new_vector: Optional[np.ndarray] = None,
    ) -> None:
        """Re-negotiate concept vector (set to new centroid or explicit vector)."""
        concepts = self._store.get(session_id, {})
        if concept_id not in concepts:
            return

        entry = concepts[concept_id]
        if new_vector is not None:
            new_vec = new_vector.astype(np.float32)
        else:
            # Auto-compute from recent history
            recent = entry.usage_history[-DRIFT_HISTORY_K:]
            new_vec = np.mean(recent, axis=0).astype(np.float32)

        norm = np.linalg.norm(new_vec)
        if norm > 0:
            new_vec = new_vec / norm

        entry.vector = new_vec
        entry.usage_history = [new_vec.copy()]
        logger.info("Updated centroid for concept '%s' in session %s", concept_id, session_id)

    def get_drifting_concepts(
        self,
        session_id: str,
        threshold: float = DRIFT_THRESHOLD,
    ) -> list[tuple[str, float]]:
        """Return all concepts whose drift score exceeds the threshold."""
        concepts = self._store.get(session_id, {})
        drifting = []
        for cid in concepts:
            score = self.drift_score(cid, session_id)
            if score > threshold:
                drifting.append((cid, score))
        return drifting

    def concept_count(self, session_id: str) -> int:
        """Number of concepts registered for a session."""
        return len(self._store.get(session_id, {}))

    def clear_session(self, session_id: str) -> None:
        """Remove all concepts for a session."""
        self._store.pop(session_id, None)


# ═══════════════════════════════════════════════════════════════════════════
# LanceDB-backed Ledger (optional persistent backend)
# ═══════════════════════════════════════════════════════════════════════════

class LanceDBLedger:
    """LanceDB-backed semantic ledger for persistent storage.

    Uses sync LanceDB API internally.  For async LangGraph nodes, use the
    ``async_*`` wrapper methods which run sync operations in an executor.
    """

    def __init__(self, db_path: str = "./.lancedb_ledger") -> None:
        try:
            import lancedb
            self._db = lancedb.connect(db_path)
            self._table_name = "concepts"
            self._usage_table = "usage_history"
            self._tables_created: set[str] = set()
            self._available = True
        except ImportError:
            logger.warning("lancedb not installed — falling back to in-memory ledger")
            self._available = False
            self._fallback = SemanticLedger()

    @property
    def available(self) -> bool:
        return self._available

    def _get_session_table_name(self, session_id: str) -> str:
        """Session-scoped table name to isolate sessions."""
        safe_id = session_id.replace("-", "_")[:32]
        return f"concepts_{safe_id}"

    def register(
        self,
        concept_vector: np.ndarray,
        concept_id: str,
        session_id: str,
    ) -> None:
        if not self._available:
            self._fallback.register(concept_vector, concept_id, session_id)
            return

        import pyarrow as pa

        table_name = self._get_session_table_name(session_id)
        vec = concept_vector.astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        data = [{
            "concept_id": concept_id,
            "session_id": session_id,
            "vector": vec.tolist(),
        }]

        try:
            if table_name not in self._tables_created:
                self._db.create_table(table_name, data=data, mode="overwrite")
                self._tables_created.add(table_name)
            else:
                table = self._db.open_table(table_name)
                table.add(data)
        except Exception as e:
            logger.error("LanceDB register failed: %s", e)

    def search(
        self,
        query_vector: np.ndarray,
        session_id: str,
        k: int = 1,
    ) -> list[tuple[str, float]]:
        if not self._available:
            return self._fallback.search(query_vector, session_id, k)

        table_name = self._get_session_table_name(session_id)
        if table_name not in self._tables_created:
            return []

        try:
            table = self._db.open_table(table_name)
            results = (
                table.search(query_vector.tolist())
                .limit(k)
                .to_list()
            )
            return [
                (r["concept_id"], 1.0 - r.get("_distance", 0.0))
                for r in results
            ]
        except Exception as e:
            logger.error("LanceDB search failed: %s", e)
            return []

    def get_all(self, session_id: str) -> dict[str, np.ndarray]:
        if not self._available:
            return self._fallback.get_all(session_id)

        table_name = self._get_session_table_name(session_id)
        if table_name not in self._tables_created:
            return {}

        try:
            table = self._db.open_table(table_name)
            df = table.to_pandas()
            return {
                row["concept_id"]: np.array(row["vector"], dtype=np.float32)
                for _, row in df.iterrows()
            }
        except Exception as e:
            logger.error("LanceDB get_all failed: %s", e)
            return {}

    # --- Async wrappers for production async nodes ---

    async def async_register(self, concept_vector, concept_id, session_id):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, self.register, concept_vector, concept_id, session_id
        )

    async def async_search(self, query_vector, session_id, k=1):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.search, query_vector, session_id, k
        )

    # Delegate remaining methods to fallback for now
    def drift_score(self, concept_id, session_id):
        if not self._available:
            return self._fallback.drift_score(concept_id, session_id)
        return 0.0  # LanceDB drift requires usage tracking

    def update_centroid(self, concept_id, session_id, new_vector=None):
        if not self._available:
            self._fallback.update_centroid(concept_id, session_id, new_vector)

    def clear_session(self, session_id):
        if not self._available:
            self._fallback.clear_session(session_id)
            return
        table_name = self._get_session_table_name(session_id)
        try:
            self._db.drop_table(table_name)
            self._tables_created.discard(table_name)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════
# Global Ledger Instance
# ═══════════════════════════════════════════════════════════════════════════

# Shared ledger instance — used by all nodes
_ledger: Optional[SemanticLedger] = None


def get_ledger() -> SemanticLedger:
    """Get the global ledger instance (lazy init)."""
    global _ledger
    if _ledger is None:
        _ledger = SemanticLedger()
    return _ledger


def set_ledger(ledger: SemanticLedger) -> None:
    """Replace the global ledger (e.g., with LanceDBLedger for persistence)."""
    global _ledger
    _ledger = ledger
