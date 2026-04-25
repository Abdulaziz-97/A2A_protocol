"""
GQESL A2A — Session Management & MessageBus

Handles:
  - Session bootstrap protocol (key exchange → basis registration → handshake verify)
  - Counter synchronisation on reconnect
  - MessageBus abstraction for transport decoupling

THREAT MODEL NOTE (Item 13):
  The initial public key exchange during session bootstrap is the ONLY point in the protocol
  vulnerable to a Man-in-the-Middle (MITM) attack. Once `session_nonce` and X25519 keys are
  exchanged and the HKDF derivations occur, all subsequent messages are HMAC-signed and
  protected. In production, the bootstrap phase MUST occur over an authenticated channel
  (e.g., mutual TLS) or use pre-shared out-of-band secrets.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from gqesl_a2a.config import MESSAGEBUS_QUEUE_MAXSIZE
from gqesl_a2a.core.crypto import (
    compute_shared_secret,
    derive_session_keys,
    generate_keypair,
    get_session_keys,
    register_session_keys,
    sign_packet,
    verify_packet,
)
from gqesl_a2a.core.concepts import KNOWN_CONCEPTS
from gqesl_a2a.core.ledger import SemanticLedger, get_ledger, warm_ledger
from gqesl_a2a.core.semantic_state import SemanticMessage
from gqesl_a2a.core.tensor_builder import build_basis_matrix

logger = logging.getLogger(__name__)


class MessageBus(ABC):
    """Abstract transport layer between agents."""

    @abstractmethod
    async def send(self, packet: bytes) -> None:
        """Send a wire packet."""
        ...

    @abstractmethod
    async def receive(self) -> bytes:
        """Receive a wire packet (blocks until available)."""
        ...


class InProcessBus(MessageBus):
    """In-memory asyncio.Queue-based transport for dev/test."""

    @classmethod
    def create_pair(cls) -> tuple["InProcessBusView", "InProcessBusView"]:
        """Create a pair of connected MessageBus views for Agent A and Agent B."""
        q_a_to_b = asyncio.Queue(maxsize=MESSAGEBUS_QUEUE_MAXSIZE)
        q_b_to_a = asyncio.Queue(maxsize=MESSAGEBUS_QUEUE_MAXSIZE)
        
        bus_a = InProcessBusView(q_a_to_b, q_b_to_a)
        bus_b = InProcessBusView(q_b_to_a, q_a_to_b)
        
        return bus_a, bus_b


class InProcessBusView(MessageBus):
    """One side's view of the InProcessBus."""

    def __init__(
        self,
        send_queue: asyncio.Queue[bytes],
        recv_queue: asyncio.Queue[bytes],
    ) -> None:
        self._send_q = send_queue
        self._recv_q = recv_queue

    async def send(self, packet: bytes) -> None:
        await self._send_q.put(packet)

    async def receive(self) -> bytes:
        return await self._recv_q.get()


class SyncInProcessBus:
    """Synchronous in-memory bus for testing without async."""

    def __init__(self) -> None:
        self._a_to_b: list[bytes] = []
        self._b_to_a: list[bytes] = []

    def send_a_to_b(self, packet: bytes) -> None:
        self._a_to_b.append(packet)

    def send_b_to_a(self, packet: bytes) -> None:
        self._b_to_a.append(packet)

    def receive_at_b(self) -> bytes | None:
        return self._a_to_b.pop(0) if self._a_to_b else None

    def receive_at_a(self) -> bytes | None:
        return self._b_to_a.pop(0) if self._b_to_a else None


@dataclass
class SessionInfo:
    """Holds session state for each agent."""
    session_id: str
    counter: int = 0
    epoch: int = 0
    is_ready: bool = False


def bootstrap_session() -> tuple[SessionInfo, SessionInfo]:
    """Run the full session bootstrap protocol for two agents.

    Steps:
      1. Both agents generate X25519 keypairs
      2. Exchange public keys
      3. Compute shared secret independently
      4. Derive all session keys
      5. Warm up codebook with projected basis vectors
      6. Register basis concepts in the ledger
      7. Handshake verify: Agent A sends a test packet, Agent B verifies

    Returns (agent_a_info, agent_b_info).
    """
    from gqesl_a2a.core.crypto import warm_codebook

    session_id = str(uuid.uuid4())
    session_nonce = os.urandom(16)

    # step 1
    priv_a, pub_a = generate_keypair()
    priv_b, pub_b = generate_keypair()

    # step 2-3
    secret_a = compute_shared_secret(priv_a, pub_b)
    secret_b = compute_shared_secret(priv_b, pub_a)
    assert secret_a == secret_b, "ECDH shared secrets do not match!"

    # step 4
    keys_a = derive_session_keys(secret_a, session_nonce, epoch=0)
    keys_b = derive_session_keys(secret_b, session_nonce, epoch=0)

    # check keys
    assert np.allclose(keys_a.W1, keys_b.W1), "W1 mismatch!"
    assert np.allclose(keys_a.W2, keys_b.W2), "W2 mismatch!"
    assert np.allclose(keys_a.codebook, keys_b.codebook), "Codebook mismatch!"
    assert keys_a.hmac_key == keys_b.hmac_key, "HMAC key mismatch!"

    # step 5
    basis = build_basis_matrix(keys_a.basis_seed)
    warmed_codebook = warm_codebook(keys_a.codebook, basis, keys_a.W1, keys_a.W2)
    keys_a.codebook = warmed_codebook
    keys_b.codebook = warmed_codebook  # same codebook

    # register keys
    register_session_keys(session_id, keys_a)

    # step 6
    ledger = get_ledger()
    for i in range(min(basis.shape[0], 55)):
        ledger.register(basis[i], f"concept_{i}", session_id)

    warmed = warm_ledger(ledger, session_id, keys_a.basis_seed, KNOWN_CONCEPTS)
    logger.info("Ledger warmed with %d shared concepts (RCC-8 vocabulary)", warmed)

    # handshake
    test_hmac = sign_packet(keys_a.hmac_key, 1, 0, "EQ", 0)
    assert verify_packet(keys_b.hmac_key, 1, 0, "EQ", 0, test_hmac), \
        "Handshake verification failed!"

    logger.info(
        "Session %s bootstrapped successfully (%d ledger rows, codebook warmed)",
        session_id,
        ledger.concept_count(session_id),
    )

    info_a = SessionInfo(session_id=session_id, is_ready=True)
    info_b = SessionInfo(session_id=session_id, is_ready=True)

    return info_a, info_b


def sync_counters(
    session_id: str,
    local_counter: int,
    peer_counter: int,
) -> int:
    """Synchronise counters on reconnect.

    Both agents advance to max(a, b) + 1.
    Peer counter must be verified via signed SyncPacket before calling.
    """
    synced = max(local_counter, peer_counter) + 1
    logger.info("Counter sync: local=%d, peer=%d → %d",
                local_counter, peer_counter, synced)
    return synced
