"""
main.py
-------
Task 10: Demonstrate the system using the five sample customer queries.

Run with:
    python3 main.py

This script:
  1. Builds the compiled LangGraph workflow (with SQLite checkpointing,
     which is what enables interrupt()/resume for human-in-the-loop).
  2. Feeds in the five demonstration queries from Section 8 of the brief,
     in order, as a single ongoing conversation from one customer.
  3. For high-risk queries (e.g. the refund request), shows the graph
     pausing for human approval, then resumes with a simulated
     supervisor decision.
  4. Prints a clear trace of: intent classification -> routing ->
     RAG retrieval -> draft response -> (approval) -> final response,
     and shows the SQLite memory recall working for Query 5.
"""

from __future__ import annotations
import json
import uuid

from langgraph.types import Command

from graph import build_graph, get_sqlite_checkpointer
import memory as memory_db


SEPARATOR = "=" * 78


def pretty_print_state(label: str, state: dict):
    print(f"\n--- {label} ---")
    keys_of_interest = [
        "intent", "is_high_risk", "risk_reason", "agent_name",
        "retrieved_sources", "approval_status", "supervisor_notes",
        "final_response",
    ]
    for k in keys_of_interest:
        if k in state and state[k] not in (None, [], ""):
            print(f"  {k}: {state[k]}")


def run_query(app, config, customer_id: str, query: str, query_label: str, expected_path: str):
    print(SEPARATOR)
    print(f"{query_label}: \"{query}\"")
    print(f"Expected path: {expected_path}")
    print(SEPARATOR)

    initial_state = {
        "customer_id": customer_id,
        "query": query,
    }

    result = app.invoke(initial_state, config=config)

    # Check whether the graph paused for human-in-the-loop approval
    if "__interrupt__" in result:
        interrupt_obj = result["__interrupt__"][0]
        payload = interrupt_obj.value
        print("\n  >>> HUMAN-IN-THE-LOOP TRIGGERED <<<")
        print(f"  Reason: {payload.get('risk_reason')}")
        print(f"  Draft awaiting approval: \"{payload.get('draft_response')[:150]}...\"")
        print("  Simulating human supervisor decision: APPROVE")
        print("  (In a real deployment this would pause for a human dashboard input.)")

        decision = Command(resume={"decision": "approve", "notes": "Approved per policy: within 14-day window."})
        result = app.invoke(decision, config=config)

    pretty_print_state(query_label, result)
    print(f"\n  FINAL RESPONSE TO CUSTOMER:\n  \"{result.get('final_response')}\"")
    return result


def main():
    checkpointer = get_sqlite_checkpointer()
    app = build_graph(checkpointer=checkpointer)

    customer_id = "cust_demo_david"
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print("\n" + SEPARATOR)
    print("AI-POWERED CUSTOMER SUPPORT AUTOMATION SYSTEM — DEMONSTRATION")
    print("Customer: David (cust_demo_david)")
    print(SEPARATOR)

    # We pre-seed Query 0 ("My name is David. I have a billing issue.")
    # exactly as it appears in Section 6 of the brief, so that Query 5's
    # memory recall ("What was my previous issue?") has something real
    # to retrieve from SQLite.
    run_query(
        app, config, customer_id,
        "My name is David. I have a billing issue with my last invoice.",
        "Seed Query (memory setup)", "Billing"
    )

    run_query(
        app, config, customer_id,
        "What are the pricing plans available for your software?",
        "Query 1", "Sales"
    )

    run_query(
        app, config, customer_id,
        "I forgot my account password.",
        "Query 2", "Account"
    )

    run_query(
        app, config, customer_id,
        "My application crashes whenever I upload a file.",
        "Query 3", "Technical Support"
    )

    run_query(
        app, config, customer_id,
        "I need a refund for my annual subscription.",
        "Query 4", "Billing — requires human approval"
    )

    run_query(
        app, config, customer_id,
        "What was my previous support issue?",
        "Query 5", "Memory recall — no department routing"
    )

    print("\n" + SEPARATOR)
    print("FULL SQLITE CONVERSATION HISTORY FOR cust_demo_david")
    print(SEPARATOR)
    history = memory_db.get_history(customer_id, limit=20)
    for turn in history:
        print(json.dumps(turn, indent=2))


if __name__ == "__main__":
    main()
