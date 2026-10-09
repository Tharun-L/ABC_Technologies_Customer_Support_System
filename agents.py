"""
agents.py
---------
Task 3: Intent Classification node.
Task 5: Specialized department agents (Sales, Technical, Billing, Account).
Task 9: Supervisor agent that validates/improves responses before sending.

Each function is a LangGraph "node": it takes the SupportState dict and
returns a partial dict of updates that LangGraph merges into state.
"""

from __future__ import annotations
import re
from typing import Dict, Any

from state import SupportState
from llm_provider import get_llm
from knowledge_base import get_knowledge_base

llm = get_llm()
kb = get_knowledge_base()

VALID_INTENTS = {"Sales", "Technical", "Billing", "Account", "Memory_Recall"}

# High-risk patterns that must always trigger human-in-the-loop approval,
# regardless of department, per Section 5 of the assignment brief.
HIGH_RISK_RULES = [
    ("refund_request", r"\brefund\b"),
    ("subscription_cancellation", r"cancel.*(subscription|plan|account)"),
    ("account_closure", r"(close|delete|terminate).*(account|profile)"),
    ("compensation_request", r"(compensat|credit me|discount.*because|reimburse)"),
    ("escalation_to_management", r"(speak (with|to) (a )?manager|escalate.*management|talk to your boss)"),
]


# ----------------------------------------------------------------------
# Task 3: Intent Classification Node
# ----------------------------------------------------------------------
def classify_intent_node(state: SupportState) -> Dict[str, Any]:
    query = state["query"]

    system_prompt = (
        "You are an intent classification assistant for a SaaS customer support system. "
        "Classify the customer's query into exactly one of: Sales, Technical, Billing, "
        "Account, or Memory_Recall (use Memory_Recall only if the customer is asking about "
        "a PREVIOUS conversation/issue rather than raising a new one). "
        "Respond with only the single label."
    )
    user_prompt = f"Customer query: \"{query}\"\n\nWhich single label applies?"

    raw = llm.generate(system_prompt, user_prompt, max_tokens=10).strip()
    intent = raw if raw in VALID_INTENTS else "Unknown"

    # If using a real hosted LLM, normalize loosely-formatted answers too.
    if intent == "Unknown":
        for label in VALID_INTENTS:
            if label.lower() in raw.lower():
                intent = label
                break

    # Detect high-risk request types (human-in-the-loop trigger), independent
    # of which department ends up handling the query.
    is_high_risk = False
    risk_reason = None
    text = query.lower()
    for reason, pattern in HIGH_RISK_RULES:
        if re.search(pattern, text):
            is_high_risk = True
            risk_reason = reason
            break

    # Extract customer name if they introduce themselves (helps memory).
    name_match = re.search(r"my name is ([A-Za-z]+)", query, re.IGNORECASE)
    customer_name_update = {}
    if name_match:
        customer_name_update["customer_name"] = name_match.group(1).capitalize()

    return {
        "intent": intent,
        "intent_confidence": 0.95 if intent != "Unknown" else 0.3,
        "is_high_risk": is_high_risk,
        "risk_reason": risk_reason,
        **customer_name_update,
    }


# ----------------------------------------------------------------------
# Task 6 support: shared RAG retrieval used by every department agent
# ----------------------------------------------------------------------
def _retrieve_context(query: str, department: str) -> Dict[str, Any]:
    results = kb.retrieve_for_department(query, department, top_k=3)
    return {
        "retrieved_context": [chunk.text for chunk, _score in results],
        "retrieved_sources": list({chunk.source for chunk, _score in results}),
    }


def _make_department_agent(department: str, agent_label: str):
    """Factory that builds a department agent node function (Task 5)."""

    def node(state: SupportState) -> Dict[str, Any]:
        query = state["query"]
        rag = _retrieve_context(query, department)
        context_block = "\n".join(f"- {c}" for c in rag["retrieved_context"]) or "(no directly relevant context found)"

        system_prompt = (
            f"You are the {agent_label} agent for ABC Technologies, a SaaS company. "
            f"Use the retrieved context to answer accurately, politely, and concisely. "
            f"If the request requires human supervisor approval (refunds, cancellations, "
            f"account closure, compensation, escalation), acknowledge the request and let "
            f"the customer know it is being reviewed by a supervisor."
        )
        user_prompt = (
            f"RETRIEVED CONTEXT:\n{context_block}\n\n"
            f"CUSTOMER QUERY: {query}"
        )

        draft = llm.generate(system_prompt, user_prompt, max_tokens=400)

        return {
            "agent_name": agent_label,
            "draft_response": draft,
            **rag,
        }

    return node


# Task 5: the four specialized department agents
sales_agent_node = _make_department_agent("Sales", "Sales")
technical_agent_node = _make_department_agent("Technical", "Technical Support")
billing_agent_node = _make_department_agent("Billing", "Billing")
account_agent_node = _make_department_agent("Account", "Account")


# ----------------------------------------------------------------------
# Memory recall path (Query 5 style: "What was my previous issue?")
# ----------------------------------------------------------------------
def memory_recall_node(state: SupportState) -> Dict[str, Any]:
    import memory as memory_db

    customer_id = state["customer_id"]
    last = memory_db.get_last_issue(customer_id)

    if last is None:
        answer = "I don't see any previous interactions on file for your account yet."
    else:
        dept = last.get("department") or "general"
        answer = (
            f"Your previous issue was related to {dept} support. "
            f"You asked: \"{last['query']}\". "
            f"Our response at the time was: \"{last.get('final_response') or 'pending response'}\"."
        )

    return {
        "agent_name": "Memory",
        "draft_response": answer,
        "memory_answer": answer,
        "retrieved_context": [],
        "retrieved_sources": [],
    }


# ----------------------------------------------------------------------
# Task 9: Supervisor agent -- validates & improves responses before send
# ----------------------------------------------------------------------
def supervisor_review_node(state: SupportState) -> Dict[str, Any]:
    draft = state.get("draft_response", "")
    query = state.get("query", "")

    system_prompt = (
        "You are a senior support supervisor. Review the draft response for accuracy, "
        "tone, and policy compliance before it is sent to the customer. "
        "If acceptable, respond starting with 'VALIDATED:'. If it needs a fix, "
        "respond starting with 'REVISED:' followed by the corrected response."
    )
    user_prompt = f"Customer query: {query}\n\nDraft response: {draft}"

    feedback = llm.generate(system_prompt, user_prompt, max_tokens=400)

    final_response = draft
    if feedback.strip().upper().startswith("REVISED:"):
        final_response = feedback.split(":", 1)[1].strip()

    return {
        "supervisor_validated": True,
        "supervisor_feedback": feedback,
        "final_response": final_response,
        "approval_status": "not_required",
    }
