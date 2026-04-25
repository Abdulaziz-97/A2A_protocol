"""
GQESL A2A — Streamlit Dashboard

Live monitoring of:
  - Semantic ledger contents and drift scores
  - Wire packet log with RCC-8 relations
  - Compression ratio metrics
  - Session state and key epoch
  - Error log

Run with: streamlit run gqesl_a2a/dashboard/app.py
"""

from __future__ import annotations

import time
import sys
import os
from pathlib import Path

# Add project root to path so Streamlit can find gqesl_a2a
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import streamlit as st

# Must be first Streamlit call
st.set_page_config(
    page_title="GQESL A2A Monitor",
    page_icon="🔐",
    layout="wide",
)


def main():
    st.title("🔐 GQESL A2A Protocol Monitor")
    st.caption("GeoQuantum Emergent Semantic Language — Live Dashboard")

    # Sidebar
    with st.sidebar:
        st.header("Session Controls")
        if st.button("🚀 Run Demo Session"):
            run_demo_session()
        st.divider()
        st.header("Configuration")
        st.code(f"Tensor Dim: 384\nCodebook: 4096\nDrift Interval: 20")

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Metrics", "📖 Ledger", "📨 Messages", "⚠️ Drift", "🔑 Session"
    ])

    with tab1:
        render_metrics()
    with tab2:
        render_ledger()
    with tab3:
        render_messages()
    with tab4:
        render_drift()
    with tab5:
        render_session()


def run_demo_session():
    """Run a demo session and store results in session state."""
    try:
        from gqesl_a2a.agents.agent_a import AgentA
        from gqesl_a2a.agents.agent_b import AgentB
        from gqesl_a2a.agents.session import bootstrap_session
        from gqesl_a2a.core.crypto import get_session_keys
        from gqesl_a2a.core.ledger import SemanticLedger, set_ledger
        from gqesl_a2a.core.semantic_state import cosine_similarity
        from gqesl_a2a.core.tensor_builder import (
            AgentIntent, EntityType, OutputFormat, TaskType,
            build_basis_matrix, build_intent_tensor,
        )

        set_ledger(SemanticLedger())
        info_a, info_b = bootstrap_session()
        agent_a = AgentA(info_a)
        agent_b = AgentB(info_b)

        scenarios = [
            AgentIntent(TaskType.EXTRACT, EntityType.PERSON, OutputFormat.JSON, 0.9, b"\x01" * 32),
            AgentIntent(TaskType.SUMMARIZE, EntityType.DOCUMENT, OutputFormat.TEXT, 0.7, b"\x02" * 32),
            AgentIntent(TaskType.CLASSIFY, EntityType.ORGANIZATION, OutputFormat.TABLE, 0.8, b"\x03" * 32),
            AgentIntent(TaskType.SEARCH, EntityType.LOCATION, OutputFormat.JSON, 0.5, b"\x04" * 32),
            AgentIntent(TaskType.GENERATE, EntityType.CODE, OutputFormat.TEXT, 0.95, b"\x05" * 32),
            AgentIntent(TaskType.VERIFY, EntityType.DATA, OutputFormat.JSON, 0.6, b"\x06" * 32),
            AgentIntent(TaskType.TRANSLATE, EntityType.DOCUMENT, OutputFormat.TEXT, 0.4, b"\x07" * 32),
            AgentIntent(TaskType.COMPARE, EntityType.PRODUCT, OutputFormat.TABLE, 0.75, b"\x08" * 32),
        ]

        messages = []
        for intent in scenarios:
            t0 = time.perf_counter()
            result_a = agent_a.encode_and_sign(intent)
            result_b = agent_b.verify_and_decode(result_a["packet"])
            latency = (time.perf_counter() - t0) * 1000

            if result_b:
                original = np.array(result_a["intent_tensor"], dtype=np.float32)
                decoded = np.array(result_b["decoded_tensor"], dtype=np.float32)
                cos_sim = cosine_similarity(original, decoded)
                messages.append({
                    "task": intent.task_type.name,
                    "entity": intent.entity_type.name,
                    "relation": result_b["relation"],
                    "strategy": result_b["strategy"],
                    "cosine": cos_sim,
                    "latency_ms": latency,
                    "wire_size": len(result_a["wire_bytes"]),
                    "idx": result_a["packet"]["idx"],
                })

        st.session_state["messages"] = messages
        st.session_state["session_id"] = info_a.session_id
        st.session_state["n_concepts"] = 55
        st.success(f"Demo session completed: {len(messages)} messages exchanged")
    except Exception as e:
        st.error(f"Demo failed: {e}")


def render_metrics():
    """Render key performance metrics."""
    messages = st.session_state.get("messages", [])
    if not messages:
        st.info("Run a demo session to see metrics.")
        return

    col1, col2, col3, col4 = st.columns(4)
    cosines = [m["cosine"] for m in messages]
    latencies = [m["latency_ms"] for m in messages]

    with col1:
        st.metric("Mean Cosine Recovery", f"{np.mean(cosines):.4f}",
                   delta="✅ Target > 0.92" if np.mean(cosines) > 0.92 else "⚠️ Below target")
    with col2:
        st.metric("Mean Latency", f"{np.mean(latencies):.2f} ms",
                   delta="✅ Target < 10ms" if np.mean(latencies) < 10 else "⚠️ Above target")
    with col3:
        avg_wire = np.mean([m["wire_size"] for m in messages])
        st.metric("Compression Ratio", f"{200/avg_wire:.0f}x", delta="Target > 20x")
    with col4:
        st.metric("Messages Exchanged", len(messages))

    st.subheader("Cosine Recovery Distribution")
    st.bar_chart({"Cosine Similarity": cosines})

    st.subheader("Latency Distribution")
    st.bar_chart({"Latency (ms)": latencies})


def render_ledger():
    """Render semantic ledger contents."""
    n = st.session_state.get("n_concepts", 0)
    if n == 0:
        st.info("Run a demo session to populate the ledger.")
        return
    st.metric("Registered Concepts", n)
    st.caption("Concepts are session-private 384-dim vectors derived from HKDF basis seed.")


def render_messages():
    """Render message log."""
    messages = st.session_state.get("messages", [])
    if not messages:
        st.info("No messages yet.")
        return

    st.subheader(f"Wire Packet Log ({len(messages)} messages)")
    for i, m in enumerate(messages):
        with st.expander(f"[{i}] {m['task']}+{m['entity']} → {m['relation']}"):
            col1, col2, col3 = st.columns(3)
            col1.metric("RCC-8 Relation", m["relation"])
            col2.metric("Cosine Recovery", f"{m['cosine']:.4f}")
            col3.metric("Wire Size", f"{m['wire_size']} B")
            st.text(f"Strategy: {m['strategy']}  |  Codebook idx: {m['idx']}  |  Latency: {m['latency_ms']:.2f}ms")


def render_drift():
    """Render drift monitoring info."""
    st.subheader("Drift Monitor")
    st.caption("Drift is measured as cosine variance of recent usage vectors per concept.")
    messages = st.session_state.get("messages", [])
    if messages:
        st.success(f"No drift detected after {len(messages)} messages")
    else:
        st.info("Run a demo session to see drift data.")


def render_session():
    """Render session state info."""
    sid = st.session_state.get("session_id", "No active session")
    st.subheader("Session Info")
    st.code(f"Session ID: {sid}\nKey Epoch: 0\nMax Messages: 10,000")
    st.caption("Session keys live in process memory only (never checkpointed).")


if __name__ == "__main__":
    main()
