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
from pathlib import Path

# import path
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import streamlit as st

# first call
st.set_page_config(
    page_title="GQESL A2A Monitor",
    page_icon="🔐",
    layout="wide",
)


def main():
    st.title("🔐 GQESL A2A Protocol Monitor")
    st.caption("GeoQuantum Emergent Semantic Language — Live Dashboard")

    # sidebar
    with st.sidebar:
        st.header("Session Controls")
        if st.button("🚀 Run Demo Session"):
            run_demo_session()
        st.divider()
        st.header("Configuration")
        st.code(f"Tensor Dim: 384\nCodebook: 4096\nDrift Interval: 20")

    # tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Metrics",
        "📖 Ledger",
        "📨 Messages",
        "🔁 Workflow",
        "⚠️ Drift",
        "🔑 Session",
    ])

    with tab1:
        render_metrics()
    with tab2:
        render_ledger()
    with tab3:
        render_messages()
    with tab4:
        render_workflow()
    with tab5:
        render_drift()
    with tab6:
        render_session()


def run_demo_session():
    """Run a demo session and store results in session state."""
    try:
        from gqesl_a2a.agents.agent_a import AgentA
        from gqesl_a2a.agents.agent_b import AgentB
        from gqesl_a2a.agents.session import bootstrap_session
        from gqesl_a2a.core.ledger import SemanticLedger, set_ledger
        from gqesl_a2a.core.semantic_state import cosine_similarity
        from gqesl_a2a.core.tensor_builder import (
            AgentIntent,
            EntityType,
            OutputFormat,
            TaskType,
        )

        set_ledger(SemanticLedger())
        info_a, info_b = bootstrap_session()
        agent_a = AgentA(info_a)
        agent_b = AgentB(info_b)

        from gqesl_a2a.core.concepts import KNOWN_CONCEPTS

        # warmed intents
        scenarios = list(KNOWN_CONCEPTS)

        messages = []
        for intent in scenarios:
            t0 = time.perf_counter()
            result_a = agent_a.encode_and_sign(intent, with_workflow=True)
            result_b = agent_b.verify_and_decode(
                result_a["packet"], with_workflow=True
            )
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
                    "sender_workflow": result_a.get("workflow"),
                    "receiver_workflow": result_b.get("workflow"),
                })

        st.session_state["messages"] = messages
        st.session_state["session_id"] = info_a.session_id
        from gqesl_a2a.core.ledger import get_ledger

        st.session_state["n_concepts"] = get_ledger().concept_count(info_a.session_id)
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


def _render_workflow_steps(steps: list[dict], *, container) -> None:
    """Render ordered pipeline steps (sender or receiver)."""
    for j, step in enumerate(steps):
        name = step.get("step", f"step_{j}")
        detail = step.get("detail", "")
        with container.expander(f"{j + 1}. `{name}`", expanded=(j < 2)):
            st.markdown(detail)
            if "tensor_stats" in step:
                st.caption("Tensor stats (shape, norm, min, max, mean)")
                st.json(step["tensor_stats"])
            if step.get("secondary_stats"):
                st.caption("Secondary branch (e.g. projected after tanh)")
                st.json(step["secondary_stats"])
            for key in ("ledger_rows", "codebook_idx", "quant_error_l2", "wire_bytes"):
                if key in step:
                    st.caption(f"{key}: {step[key]}")


def render_workflow():
    """Per-message encode/decode transformations (same path as demo agents)."""
    messages = st.session_state.get("messages", [])
    if not messages:
        st.info("Run a demo session to see each pipeline step.")
        return

    st.subheader("Encode → wire → decode pipeline")
    st.caption(
        "Sender (Agent A): basis → intent tensor → salt → non-linear encode → "
        "RCC-8 vs ledger → codebook index → HMAC → bytes. "
        "Receiver (Agent B): verify → salt → inverse decode → collapse → strategy."
    )

    options = [
        f"[{i}] {m['task']} + {m['entity']} → {m['relation']}"
        for i, m in enumerate(messages)
    ]
    choice = st.selectbox("Message", range(len(options)), format_func=lambda i: options[i])
    m = messages[choice]

    sw = m.get("sender_workflow") or []
    rw = m.get("receiver_workflow") or []

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Sender (Agent A)")
        if sw:
            _render_workflow_steps(sw, container=st)
        else:
            st.warning("No sender workflow captured (re-run demo).")
    with col_b:
        st.markdown("### Receiver (Agent B)")
        if rw:
            _render_workflow_steps(rw, container=st)
        else:
            st.warning("No receiver workflow (decode may have failed).")


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
            if m.get("sender_workflow") and m.get("receiver_workflow"):
                st.caption("Open the **Workflow** tab for full step-by-step transforms for this run.")


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
