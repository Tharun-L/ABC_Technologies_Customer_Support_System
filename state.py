"""
state.py
--------
Task 2: State structure for the Customer Support Automation System.

This TypedDict defines the shared "memory" that flows through every node
of the LangGraph workflow. Every node receives the current state, reads
what it needs, and returns a partial dict that LangGraph merges back in.
"""

from __future__ import annotations
from typing import TypedDict, Optional, List, Dict, Any, Literal


IntentType = Literal["Sales", "Technical", "Billing", "Account", "Memory_Recall", "Unknown"]

ApprovalStatus = Literal["not_required", "pending", "approved", "rejected"]


class SupportState(TypedDict, total=False):
    # ---- Customer information ----
    customer_id: str                     # unique customer identifier (used as SQLite key)
    customer_name: Optional[str]         # extracted/known customer name

    # ---- Query details ----
    query: str                           # the raw incoming customer message
    intent: IntentType                   # classified intent/department
    intent_confidence: float             # classifier confidence score (0-1)
    is_high_risk: bool                   # True if query matches a human-in-the-loop trigger
    risk_reason: Optional[str]           # why it was flagged high-risk (e.g. "refund_request")

    # ---- Retrieved context (RAG) ----
    retrieved_context: List[str]         # chunks retrieved from the knowledge base
    retrieved_sources: List[str]         # which documents the chunks came from

    # ---- Memory ----
    conversation_history: List[Dict[str, Any]]  # past turns pulled from SQLite for this customer
    memory_answer: Optional[str]         # direct answer when intent == Memory_Recall

    # ---- Agent output ----
    agent_name: Optional[str]            # which department agent handled this turn
    draft_response: Optional[str]        # response produced by the department agent

    # ---- Human-in-the-loop ----
    approval_status: ApprovalStatus      # not_required | pending | approved | rejected
    supervisor_notes: Optional[str]      # human supervisor's comments, if any
    human_decision_required: bool        # flag LangGraph uses to know to interrupt

    # ---- Supervisor / QA agent ----
    supervisor_validated: bool           # True once Supervisor agent has reviewed draft_response
    supervisor_feedback: Optional[str]   # AI supervisor's notes on tone/accuracy/policy

    # ---- Final output ----
    final_response: Optional[str]        # the response actually sent to the customer

    # ---- Bookkeeping ----
    turn_id: Optional[int]               # row id once persisted to SQLite
    error: Optional[str]                 # any error message captured along the way
