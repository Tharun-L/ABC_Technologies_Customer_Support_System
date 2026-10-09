"""
graph.py
--------
Task 1: LangGraph workflow design for the Customer Support Automation System.
Task 4: Conditional routing to direct queries to the appropriate support agent.

Workflow overview
==================

    load_memory_node (pulls SQLite history)
        -> classify_intent_node (Task 3)
        -> route_after_classification (Task 4), branches to one of:
             Sales agent | Technical agent | Billing agent | Account agent | Memory_Recall node
        -> route_after_agent: high-risk? -> human_approval node (Task 8)
                               not high-risk? -> supervisor_review node (Task 9)
        -> (Memory_Recall path goes straight to finalize_unflagged)
        -> save_memory_node (Task 7, persists every turn)
        -> END

Notes:
- `classify_intent_node` and the department agents are imported from
  agents.py (Tasks 3 & 5).
- `human_approval_node` (Task 8) is only reached for high-risk requests
  (refunds, cancellations, account closure, compensation, escalation),
  enforced via the conditional edge `route_after_agent`.
- Non-high-risk drafts go straight to the Supervisor agent (Task 9) for
  automated QA before being finalized.
- Memory_Recall queries skip department routing entirely per the brief
  ("Memory recall - no department routing").
- `save_memory_node` (Task 7) persists every turn to SQLite regardless
  of which path was taken, so future "what was my previous issue"
  queries can be answered.
"""

from __future__ import annotations
from typing import Dict, Any

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import os

from state import SupportState
from agents import (
    classify_intent_node,
    sales_agent_node,
    technical_agent_node,
    billing_agent_node,
    account_agent_node,
    memory_recall_node,
    supervisor_review_node,
)
from human_in_the_loop import human_approval_node
import memory as memory_db

CHECKPOINT_DB_PATH = os.path.join(os.path.dirname(__file__), "memory_db", "graph_checkpoints.db")


# ----------------------------------------------------------------------
# Extra nodes: load memory at the start, save memory at the end
# ----------------------------------------------------------------------
def load_memory_node(state: SupportState) -> Dict[str, Any]:
    customer_id = state["customer_id"]
    history = memory_db.get_history(customer_id)
    known_name = memory_db.get_customer_name(customer_id)
    update: Dict[str, Any] = {"conversation_history": history}
    if known_name and not state.get("customer_name"):
        update["customer_name"] = known_name
    return update


def save_memory_node(state: SupportState) -> Dict[str, Any]:
    turn_id = memory_db.save_turn(
        customer_id=state["customer_id"],
        query=state["query"],
        intent=state.get("intent"),
        department=state.get("agent_name"),
        final_response=state.get("final_response") or state.get("draft_response"),
        approval_status=state.get("approval_status", "not_required"),
        customer_name=state.get("customer_name"),
    )
    return {"turn_id": turn_id}


def finalize_unflagged_node(state: SupportState) -> Dict[str, Any]:
    """For Memory_Recall path: no agent draft/supervisor cycle needed."""
    return {
        "approval_status": "not_required",
        "final_response": state.get("memory_answer") or state.get("draft_response"),
    }


# ----------------------------------------------------------------------
# Task 4: Conditional routing functions
# ----------------------------------------------------------------------
def route_after_classification(state: SupportState) -> str:
    """Routes to the correct department agent (or memory recall) based on intent."""
    intent = state.get("intent", "Unknown")
    mapping = {
        "Sales": "sales_agent",
        "Technical": "technical_agent",
        "Billing": "billing_agent",
        "Account": "account_agent",
        "Memory_Recall": "memory_recall",
    }
    return mapping.get(intent, "billing_agent")  # safe default: route unclear queries to Billing/general support


def route_after_agent(state: SupportState) -> str:
    """Decides whether a high-risk request needs human approval, or can go
    straight to the automated Supervisor QA agent (Task 8 vs Task 9 fork)."""
    if state.get("is_high_risk"):
        return "human_approval"
    return "supervisor_review"


# ----------------------------------------------------------------------
# Build the graph
# ----------------------------------------------------------------------
def build_graph(checkpointer=None):
    graph = StateGraph(SupportState)

    # Nodes
    graph.add_node("load_memory", load_memory_node)
    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("sales_agent", sales_agent_node)
    graph.add_node("technical_agent", technical_agent_node)
    graph.add_node("billing_agent", billing_agent_node)
    graph.add_node("account_agent", account_agent_node)
    graph.add_node("memory_recall", memory_recall_node)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("supervisor_review", supervisor_review_node)
    graph.add_node("finalize_unflagged", finalize_unflagged_node)
    graph.add_node("save_memory", save_memory_node)

    # Edges
    graph.add_edge(START, "load_memory")
    graph.add_edge("load_memory", "classify_intent")

    # Task 4: conditional routing to department agents
    graph.add_conditional_edges(
        "classify_intent",
        route_after_classification,
        {
            "sales_agent": "sales_agent",
            "technical_agent": "technical_agent",
            "billing_agent": "billing_agent",
            "account_agent": "account_agent",
            "memory_recall": "memory_recall",
        },
    )

    # After each department agent drafts a response, decide: human approval
    # (Task 8, high-risk) or straight to automated supervisor QA (Task 9).
    for agent_node in ["sales_agent", "technical_agent", "billing_agent", "account_agent"]:
        graph.add_conditional_edges(
            agent_node,
            route_after_agent,
            {
                "human_approval": "human_approval",
                "supervisor_review": "supervisor_review",
            },
        )

    # Memory recall bypasses department/approval entirely
    graph.add_edge("memory_recall", "finalize_unflagged")

    graph.add_edge("human_approval", "save_memory")
    graph.add_edge("supervisor_review", "save_memory")
    graph.add_edge("finalize_unflagged", "save_memory")
    graph.add_edge("save_memory", END)

    if checkpointer is not None:
        return graph.compile(checkpointer=checkpointer)
    return graph.compile()


def get_sqlite_checkpointer():
    os.makedirs(os.path.dirname(CHECKPOINT_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
    return SqliteSaver(conn)
