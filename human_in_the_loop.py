"""
human_in_the_loop.py
---------------------
Task 8: Human-in-the-loop approval process for high-risk customer requests.

Per Section 5 of the brief, these request types must NOT be auto-approved:
    - Refund requests
    - Subscription cancellation
    - Account closure requests
    - Compensation requests
    - Escalation to management

This module implements the approval *gate* as a LangGraph node that calls
`interrupt()`. When the graph hits this node for a high-risk request, it
pauses execution and surfaces the draft response + risk reason to a human
supervisor. The supervisor's decision (approve / reject / edit) is supplied
via `Command(resume=...)` when the graph is resumed, and is recorded back
into the SQLite audit trail.

In an interactive deployment (e.g. a supervisor dashboard), the `interrupt()`
payload below is exactly what would be rendered as an approval card.
"""

from __future__ import annotations
from typing import Dict, Any

from langgraph.types import interrupt

from state import SupportState


def human_approval_node(state: SupportState) -> Dict[str, Any]:
    """
    Pauses the graph and asks a human supervisor to approve, reject, or
    edit the draft response for a high-risk request.

    The `interrupt()` call suspends execution here. Whoever resumes the
    graph (via `graph.invoke(Command(resume=decision), config)`) provides
    a dict like:
        {"decision": "approve"}
        {"decision": "reject", "notes": "Customer not eligible, >14 days"}
        {"decision": "edit", "edited_response": "...", "notes": "..."}
    """
    payload = {
        "type": "approval_request",
        "risk_reason": state.get("risk_reason"),
        "customer_id": state.get("customer_id"),
        "customer_name": state.get("customer_name"),
        "query": state.get("query"),
        "agent_name": state.get("agent_name"),
        "draft_response": state.get("draft_response"),
        "instructions": (
            "This request was flagged as high-risk and requires supervisor "
            "approval before it is sent to the customer. Respond with "
            "decision: 'approve', 'reject', or 'edit'."
        ),
    }

    decision: Dict[str, Any] = interrupt(payload)

    decision_type = (decision or {}).get("decision", "reject")
    notes = (decision or {}).get("notes")

    if decision_type == "approve":
        return {
            "approval_status": "approved",
            "supervisor_notes": notes or "Approved without changes.",
            "final_response": state.get("draft_response"),
        }

    if decision_type == "edit":
        edited = (decision or {}).get("edited_response", state.get("draft_response"))
        return {
            "approval_status": "approved",
            "supervisor_notes": notes or "Approved with edits.",
            "final_response": edited,
        }

    # default / "reject"
    fallback_message = (
        "We have received your request and a member of our team will follow "
        "up with you personally to discuss next steps."
    )
    return {
        "approval_status": "rejected",
        "supervisor_notes": notes or "Rejected by supervisor.",
        "final_response": fallback_message,
    }
