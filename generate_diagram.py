"""
generate_diagram.py
--------------------
Generates the LangGraph workflow architecture diagram (PNG) required
as a deliverable. Built with Graphviz for a clean, presentation-quality
result rather than the raw ASCII/mermaid debug output.
"""

import graphviz

dot = graphviz.Digraph("CustomerSupportWorkflow", format="png")
dot.attr(rankdir="TB", splines="spline", bgcolor="white", fontname="Helvetica")
dot.attr("node", fontname="Helvetica", fontsize="11", style="filled")
dot.attr("edge", fontname="Helvetica", fontsize="9", color="#555555")

# --- Start/End ---
dot.node("start", "START", shape="circle", fillcolor="#2ecc71", fontcolor="white", width="0.6")
dot.node("end", "END", shape="circle", fillcolor="#e74c3c", fontcolor="white", width="0.6")

# --- Core pipeline nodes ---
dot.node("load_memory", "load_memory_node\n(Task 7: pull SQLite history)", shape="box", fillcolor="#dff0d8")
dot.node("classify_intent", "classify_intent_node\n(Task 3: Sales/Tech/Billing/Account/\nMemory_Recall)", shape="box", fillcolor="#fcf3cf")

# --- Department agents ---
with dot.subgraph(name="cluster_agents") as c:
    c.attr(label="Task 5: Specialized Department Agents", style="dashed", color="#888888", fontname="Helvetica")
    c.node("sales_agent", "Sales Agent\n(Task 6: RAG -> Pricing Guide)", shape="box", fillcolor="#d6eaf8")
    c.node("technical_agent", "Technical Agent\n(Task 6: RAG -> Technical Manual)", shape="box", fillcolor="#d6eaf8")
    c.node("billing_agent", "Billing Agent\n(Task 6: RAG -> Company Policy)", shape="box", fillcolor="#d6eaf8")
    c.node("account_agent", "Account Agent\n(Task 6: RAG -> FAQ Document)", shape="box", fillcolor="#d6eaf8")

dot.node("memory_recall", "memory_recall_node\n(Task 7: SQLite lookup,\nno department routing)", shape="box", fillcolor="#e8daef")

# --- Human-in-the-loop / Supervisor fork ---
dot.node("human_approval", "human_approval_node\n(Task 8: interrupt() ->\nhuman supervisor decision)", shape="box", fillcolor="#f5b7b1")
dot.node("supervisor_review", "supervisor_review_node\n(Task 9: AI Supervisor QA)", shape="box", fillcolor="#fae5d3")

dot.node("finalize_unflagged", "finalize_unflagged_node\n(memory recall passthrough)", shape="box", fillcolor="#e8daef")
dot.node("save_memory", "save_memory_node\n(Task 7: persist turn to SQLite)", shape="box", fillcolor="#dff0d8")

# --- Edges ---
dot.edge("start", "load_memory")
dot.edge("load_memory", "classify_intent")

dot.edge("classify_intent", "sales_agent", label="Sales", style="dashed")
dot.edge("classify_intent", "technical_agent", label="Technical", style="dashed")
dot.edge("classify_intent", "billing_agent", label="Billing", style="dashed")
dot.edge("classify_intent", "account_agent", label="Account", style="dashed")
dot.edge("classify_intent", "memory_recall", label="Memory_Recall", style="dashed")

for agent in ["sales_agent", "technical_agent", "billing_agent", "account_agent"]:
    dot.edge(agent, "human_approval", label="high-risk", style="dashed", color="#c0392b", fontcolor="#c0392b")
    dot.edge(agent, "supervisor_review", label="normal", style="dashed", color="#2980b9", fontcolor="#2980b9")

dot.edge("memory_recall", "finalize_unflagged")

dot.edge("human_approval", "save_memory")
dot.edge("supervisor_review", "save_memory")
dot.edge("finalize_unflagged", "save_memory")
dot.edge("save_memory", "end")

dot.render("/home/claude/customer_support_system/workflow_diagram", cleanup=True)
print("Diagram saved.")
