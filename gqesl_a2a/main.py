"""
GQESL A2A -- Main Entry Point

Wires everything together:
  1. Bootstraps a session between Agent A and Agent B
  2. Runs demo scenarios
  3. Reports metrics (compression ratio, cosine recovery, etc.)
"""

from __future__ import annotations

import logging
import sys
import time

import numpy as np

from gqesl_a2a import config
from gqesl_a2a.agents.agent_a import AgentA
from gqesl_a2a.agents.agent_b import AgentB
from gqesl_a2a.agents.session import bootstrap_session
from gqesl_a2a.core.crypto import get_session_keys
from gqesl_a2a.core.ledger import get_ledger
from gqesl_a2a.core.semantic_state import cosine_similarity
from gqesl_a2a.core.tensor_builder import (
    AgentIntent,
    EntityType,
    OutputFormat,
    TaskType,
    build_basis_matrix,
    build_intent_tensor,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("gqesl_a2a.main")


def run_demo():
    """Run a complete GQESL demonstration."""

    print("=" * 70)
    print("  GQESL A2A Protocol -- Demo")
    print("  GeoQuantum Emergent Semantic Language")
    print("=" * 70)
    print()

    # -- Step 1: Bootstrap Session --
    print("[1] Bootstrapping session...")
    t0 = time.perf_counter()
    info_a, info_b = bootstrap_session()
    bootstrap_time = (time.perf_counter() - t0) * 1000
    print(f"    [OK] Session {info_a.session_id[:12]}... bootstrapped in {bootstrap_time:.1f}ms")
    print(f"    [OK] {get_ledger().concept_count(info_a.session_id)} concepts registered")
    print()

    # -- Step 2: Create Agents --
    agent_a = AgentA(info_a)
    agent_b = AgentB(info_b)

    # -- Step 3: Demo Scenarios --
    scenarios = [
        AgentIntent(TaskType.EXTRACT, EntityType.PERSON, OutputFormat.JSON, 0.9, b"\x01" * 32),
        AgentIntent(TaskType.SUMMARIZE, EntityType.DOCUMENT, OutputFormat.TEXT, 0.7, b"\x02" * 32),
        AgentIntent(TaskType.CLASSIFY, EntityType.ORGANIZATION, OutputFormat.TABLE, 0.8, b"\x03" * 32),
        AgentIntent(TaskType.SEARCH, EntityType.LOCATION, OutputFormat.JSON, 0.5, b"\x04" * 32),
        AgentIntent(TaskType.GENERATE, EntityType.CODE, OutputFormat.TEXT, 0.95, b"\x05" * 32),
    ]

    keys = get_session_keys(info_a.session_id)
    basis = build_basis_matrix(keys.basis_seed)

    print("[2] Running encode -> wire -> decode scenarios:")
    print("-" * 70)

    latencies = []
    cosine_sims = []
    wire_sizes = []

    for i, intent in enumerate(scenarios):
        t0 = time.perf_counter()

        # Agent A encodes
        result_a = agent_a.encode_and_sign(intent)

        # Agent B receives and decodes
        result_b = agent_b.verify_and_decode(result_a["packet"])

        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)

        if result_b is None:
            print(f"    [{i+1}] FAILED -- decode returned None")
            continue

        # Compute cosine similarity
        original = np.array(result_a["intent_tensor"], dtype=np.float32)
        decoded = np.array(result_b["decoded_tensor"], dtype=np.float32)
        cos_sim = cosine_similarity(original, decoded)
        cosine_sims.append(cos_sim)

        wire_size = len(result_a["wire_bytes"])
        wire_sizes.append(wire_size)

        # Compression ratio (vs hypothetical NL description ~200 chars)
        nl_size = 200  # Conservative estimate for NL task description
        compression = nl_size / wire_size if wire_size > 0 else 0

        print(f"    [{i+1}] {intent.task_type.name:>10} + {intent.entity_type.name:<12} "
              f"-> relation={result_b['relation']} "
              f"cosine={cos_sim:.4f} "
              f"latency={latency_ms:.2f}ms "
              f"wire={wire_size}B "
              f"compression={compression:.0f}x")

    print("-" * 70)

    # -- Step 4: Summary Metrics --
    if cosine_sims:
        print()
        print("[3] Summary Metrics:")
        print(f"    Mean cosine recovery:  {np.mean(cosine_sims):.4f}")
        print(f"    Min cosine recovery:   {np.min(cosine_sims):.4f}")
        print(f"    Mean latency:          {np.mean(latencies):.2f}ms (target < 10ms)")
        print(f"    Max latency:           {np.max(latencies):.2f}ms")
        print(f"    Wire packet size:      {np.mean(wire_sizes):.0f} bytes")
        print(f"    Compression ratio:     {200/np.mean(wire_sizes):.0f}x (target > 20x)")
        print()

        if np.mean(latencies) < 10:
            print("    [PASS] Latency target MET")
        else:
            print("    [WARN] Latency above target")

    # -- Step 5: Drift Check --
    ledger = get_ledger()
    drifting = ledger.get_drifting_concepts(info_a.session_id)
    print(f"\n[4] Drift: {len(drifting)} concepts drifting out of "
          f"{ledger.concept_count(info_a.session_id)} total")

    print("\n" + "=" * 70)
    print("  Demo complete.")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
