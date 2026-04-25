"""
GQESL A2A — LangGraph Graph Assembly

Assembles the full StateGraph with:
  - Conditional edges for RCC-8 routing (4 strategy nodes)
  - Conditional edge for drift monitoring (fires every N messages)
  - Error routing edges
  - Session termination edge
"""

from __future__ import annotations

import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from gqesl_a2a.config import DRIFT_CHECK_INTERVAL
from gqesl_a2a.graph.nodes import (
    build_intent_node,
    collapse_node,
    counter_sync_node,
    decode_node,
    drift_monitor_node,
    encode_node,
    error_handler_node,
    exact_dispatch_node,
    execute_node,
    handoff_node,
    key_exchange_node,
    negotiate_node,
    parallel_split_node,
    respond_node,
    rcc8_route_node,
    session_terminate_node,
    sign_node,
    verify_node,
)
from gqesl_a2a.graph.state import GQESLState

logger = logging.getLogger(__name__)


def _route_rcc8(state: GQESLState) -> str:
    """Route based on RCC-8 relation to the appropriate strategy node."""
    strategy = state.get("strategy", "NEGOTIATE_FIRST")
    mapping = {
        "EXACT_MATCH": "exact_dispatch",
        "SPLIT_EXECUTION": "parallel_split",
        "FULL_HANDOFF": "handoff",
        "NEGOTIATE_FIRST": "negotiate",
    }
    return mapping.get(strategy, "negotiate")


def _should_check_drift(state: GQESLState) -> str:
    """Conditional edge: check drift every DRIFT_CHECK_INTERVAL messages."""
    counter = state.get("counter", 0)
    if counter > 0 and counter % DRIFT_CHECK_INTERVAL == 0:
        return "drift_monitor"
    return "execute"


def _check_encode_errors(state: GQESLState) -> str:
    """After encode, check for session termination or errors."""
    error = state.get("error")
    if error == "session_max_messages_reached":
        return "session_terminate"
    return "sign"


def _check_verify_result(state: GQESLState) -> str:
    """After verify, check for errors."""
    error = state.get("error")
    if error:
        return "error_handler"
    return "decode"


def _check_negotiate_result(state: GQESLState) -> str:
    """After negotiate, check for rejection."""
    error = state.get("error")
    if error:
        return "error_handler"
    return "collapse"


def build_sender_graph() -> StateGraph:
    """Build the sender-side graph: intent → encode → sign → transmit."""
    graph = StateGraph(GQESLState)

    # nodes
    graph.add_node("key_exchange", key_exchange_node)
    graph.add_node("build_intent", build_intent_node)
    graph.add_node("encode", encode_node)
    graph.add_node("sign", sign_node)
    graph.add_node("session_terminate", session_terminate_node)
    graph.add_node("error_handler", error_handler_node)

    # edges
    graph.set_entry_point("key_exchange")
    graph.add_edge("key_exchange", "build_intent")
    graph.add_edge("build_intent", "encode")
    graph.add_conditional_edges("encode", _check_encode_errors, {
        "sign": "sign",
        "session_terminate": "session_terminate",
    })
    graph.add_edge("sign", END)
    graph.add_edge("session_terminate", END)
    graph.add_edge("error_handler", END)

    return graph


def build_receiver_graph() -> StateGraph:
    """Build the receiver-side graph: verify → decode → route → execute → respond."""
    graph = StateGraph(GQESLState)

    # nodes
    graph.add_node("verify", verify_node)
    graph.add_node("decode", decode_node)
    graph.add_node("rcc8_route", rcc8_route_node)
    graph.add_node("exact_dispatch", exact_dispatch_node)
    graph.add_node("parallel_split", parallel_split_node)
    graph.add_node("handoff", handoff_node)
    graph.add_node("negotiate", negotiate_node)
    graph.add_node("collapse", collapse_node)
    graph.add_node("drift_monitor", drift_monitor_node)
    graph.add_node("execute", execute_node)
    graph.add_node("respond", respond_node)
    graph.add_node("error_handler", error_handler_node)
    graph.add_node("counter_sync", counter_sync_node)
    graph.add_node("session_terminate", session_terminate_node)

    # entry
    graph.set_entry_point("verify")

    # verify route
    graph.add_conditional_edges("verify", _check_verify_result, {
        "decode": "decode",
        "error_handler": "error_handler",
    })

    # decode route
    graph.add_edge("decode", "rcc8_route")

    # rcc8 route
    graph.add_conditional_edges("rcc8_route", _route_rcc8, {
        "exact_dispatch": "exact_dispatch",
        "parallel_split": "parallel_split",
        "handoff": "handoff",
        "negotiate": "negotiate",
    })

    # strategy route
    graph.add_edge("exact_dispatch", "collapse")
    graph.add_edge("parallel_split", "collapse")
    graph.add_edge("handoff", "collapse")
    graph.add_conditional_edges("negotiate", _check_negotiate_result, {
        "collapse": "collapse",
        "error_handler": "error_handler",
    })

    # collapse route
    graph.add_conditional_edges("collapse", _should_check_drift, {
        "drift_monitor": "drift_monitor",
        "execute": "execute",
    })
    graph.add_edge("drift_monitor", "execute")

    # execute route
    graph.add_edge("execute", "respond")
    graph.add_edge("respond", END)

    # error route
    graph.add_edge("error_handler", END)
    graph.add_edge("session_terminate", END)
    graph.add_edge("counter_sync", "verify")

    return graph


def compile_sender_graph(checkpointer=None):
    """Compile the sender graph with optional checkpointer."""
    graph = build_sender_graph()
    if checkpointer is None:
        checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


def compile_receiver_graph(checkpointer=None):
    """Compile the receiver graph with optional checkpointer."""
    graph = build_receiver_graph()
    if checkpointer is None:
        checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
