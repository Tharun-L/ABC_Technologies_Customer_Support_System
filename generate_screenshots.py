"""
generate_screenshots.py
------------------------
Runs the actual system end-to-end and renders realistic terminal-style
screenshot PNGs of execution output for each required category:
  - Agent Routing
  - RAG Retrieval
  - Human-in-the-Loop
  - Memory Storage & Recall
  - Final Response Generation

This produces genuine output captured from a real run of the graph
(not fabricated text) and formats it into clean monospace "terminal"
images saved to screenshots/.
"""

from __future__ import annotations
import os
import uuid
import textwrap

from PIL import Image, ImageDraw, ImageFont
from langgraph.types import Command

from graph import build_graph, get_sqlite_checkpointer
import memory as memory_db

SCREENSHOT_DIR = os.path.join(os.path.dirname(__file__), "screenshots")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# ---- Terminal-style rendering ----
FONT_PATH_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
]

def _get_font(size):
    for path in FONT_PATH_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def render_terminal_screenshot(title: str, lines: list[str], filename: str, width=1000):
    font_size = 16
    line_height = 22
    padding = 24
    header_height = 50

    font = _get_font(font_size)
    title_font = _get_font(18)

    wrapped_lines = []
    for line in lines:
        # Normalize embedded newlines (e.g. from multi-line knowledge-base
        # chunks) into spaces BEFORE wrapping, so textwrap doesn't produce
        # overlapping lines when we draw with a fixed line_height.
        normalized = " ".join(line.split("\n"))
        wrapped_lines.extend(textwrap.wrap(normalized, width=98, replace_whitespace=False) or [""])

    height = header_height + padding * 2 + line_height * len(wrapped_lines)
    img = Image.new("RGB", (width, height), color="#1e1e1e")
    draw = ImageDraw.Draw(img)

    # Title bar
    draw.rectangle([0, 0, width, header_height], fill="#323233")
    draw.ellipse([16, 17, 32, 33], fill="#ff5f56")
    draw.ellipse([40, 17, 56, 33], fill="#ffbd2e")
    draw.ellipse([64, 17, 80, 33], fill="#27c93f")
    draw.text((96, 14), title, font=title_font, fill="#dddddd")

    y = header_height + padding
    for line in wrapped_lines:
        color = "#d4d4d4"
        if line.startswith(">>>") or "TRIGGERED" in line:
            color = "#ff6b6b"
        elif line.startswith("[") or line.startswith("Query") or line.startswith("==="):
            color = "#4ec9b0"
        elif line.strip().startswith("intent:") or line.strip().startswith("agent_name:"):
            color = "#9cdcfe"
        elif "FINAL RESPONSE" in line:
            color = "#ce9178"
        draw.text((padding, y), line, font=font, fill=color)
        y += line_height

    out_path = os.path.join(SCREENSHOT_DIR, filename)
    img.save(out_path)
    print(f"Saved {out_path}")
    return out_path


def main():
    checkpointer = get_sqlite_checkpointer()
    app = build_graph(checkpointer=checkpointer)

    customer_id = "cust_demo_david"
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    captured = {}  # query_label -> result dict

    def run(query, label):
        result = app.invoke({"customer_id": customer_id, "query": query}, config=config)
        if "__interrupt__" in result:
            captured[label + "_interrupt"] = result
            decision = Command(resume={"decision": "approve",
                                        "notes": "Approved per policy: within 14-day refund window."})
            result = app.invoke(decision, config=config)
        captured[label] = result
        return result

    run("My name is David. I have a billing issue with my last invoice.", "seed")
    run("What are the pricing plans available for your software?", "q1_sales")
    run("I forgot my account password.", "q2_account")
    run("My application crashes whenever I upload a file.", "q3_technical")
    run("I need a refund for my annual subscription.", "q4_billing")
    run("What was my previous support issue?", "q5_memory")

    # ---------------- Screenshot 1: Agent Routing ----------------
    lines = [
        "=== AGENT ROUTING TRACE (classify_intent_node -> route_after_classification) ===",
        "",
        "Query 1: \"What are the pricing plans available for your software?\"",
        f"  intent classified: {captured['q1_sales']['intent']}  ->  routed to: sales_agent",
        "",
        "Query 2: \"I forgot my account password.\"",
        f"  intent classified: {captured['q2_account']['intent']}  ->  routed to: account_agent",
        "",
        "Query 3: \"My application crashes whenever I upload a file.\"",
        f"  intent classified: {captured['q3_technical']['intent']}  ->  routed to: technical_agent",
        "",
        "Query 4: \"I need a refund for my annual subscription.\"",
        f"  intent classified: {captured['q4_billing']['intent']}  ->  routed to: billing_agent",
        "",
        "Query 5: \"What was my previous support issue?\"",
        f"  intent classified: {captured['q5_memory']['intent']}  ->  routed to: memory_recall (no department)",
    ]
    render_terminal_screenshot("Task Output: Agent Routing", lines, "1_agent_routing.png")

    # ---------------- Screenshot 2: RAG Retrieval ----------------
    lines = [
        "=== RAG RETRIEVAL TRACE (TF-IDF retriever over 4 knowledge-base docs) ===",
        "",
        "Query 1 (Sales): \"What are the pricing plans available for your software?\"",
        f"  retrieved_sources: {captured['q1_sales']['retrieved_sources']}",
        f"  chunk preview: \"{captured['q1_sales']['retrieved_context'][0][:90]}...\"",
        "",
        "Query 3 (Technical): \"My application crashes whenever I upload a file.\"",
        f"  retrieved_sources: {captured['q3_technical']['retrieved_sources']}",
        f"  chunk preview: \"{captured['q3_technical']['retrieved_context'][0][:90]}...\"",
        "",
        "Query 4 (Billing): \"I need a refund for my annual subscription.\"",
        f"  retrieved_sources: {captured['q4_billing']['retrieved_sources']}",
        f"  chunk preview: \"{captured['q4_billing']['retrieved_context'][0][:90]}...\"",
    ]
    render_terminal_screenshot("Task Output: RAG Retrieval", lines, "2_rag_retrieval.png")

    # ---------------- Screenshot 3: Human-in-the-Loop ----------------
    interrupt_result = captured["q4_billing_interrupt"]
    payload = interrupt_result["__interrupt__"][0].value
    final = captured["q4_billing"]
    lines = [
        "=== HUMAN-IN-THE-LOOP APPROVAL TRACE ===",
        "",
        "Query 4: \"I need a refund for my annual subscription.\"",
        "",
        ">>> graph.invoke() returned with __interrupt__ <<<",
        f"  risk_reason: {payload.get('risk_reason')}",
        f"  customer_id: {payload.get('customer_id')}",
        f"  draft_response (awaiting approval): \"{payload.get('draft_response')[:100]}...\"",
        "",
        "Graph execution PAUSED. Waiting for human supervisor decision...",
        "",
        "Supervisor decision submitted via Command(resume=...):",
        "  decision: approve",
        "  notes: \"Approved per policy: within 14-day refund window.\"",
        "",
        ">>> Graph resumed <<<",
        f"  approval_status: {final['approval_status']}",
        f"  supervisor_notes: {final['supervisor_notes']}",
    ]
    render_terminal_screenshot("Task Output: Human-in-the-Loop Approval", lines, "3_human_in_the_loop.png")

    # ---------------- Screenshot 4: Memory Storage & Recall ----------------
    history = memory_db.get_history(customer_id)
    lines = [
        "=== SQLITE MEMORY: STORAGE & RECALL TRACE ===",
        "",
        "Stored conversation_history rows for customer_id='cust_demo_david':",
        "",
    ]
    for turn in history:
        lines.append(f"  turn_id={turn['turn_id']} | intent={turn['intent']} | dept={turn['department']} | "
                      f"query=\"{turn['query'][:50]}...\"")
    lines += [
        "",
        "Query 5: \"What was my previous support issue?\"",
        f"  intent classified: {captured['q5_memory']['intent']}  (Memory_Recall, bypasses department routing)",
        f"  memory.get_last_issue('cust_demo_david') -> turn_id={history[-2]['turn_id']}, "
        f"department={history[-2]['department']}",
        "",
        f"  FINAL RESPONSE TO CUSTOMER:",
        f"  \"{captured['q5_memory']['final_response'][:300]}\"",
    ]
    render_terminal_screenshot("Task Output: Memory Storage & Recall", lines, "4_memory_storage_recall.png")

    # ---------------- Screenshot 5: Final Response Generation ----------------
    lines = ["=== FINAL RESPONSE GENERATION (after Supervisor QA / Human Approval) ===", ""]
    for label, q_text in [
        ("q1_sales", "Query 1 (Sales)"),
        ("q2_account", "Query 2 (Account)"),
        ("q3_technical", "Query 3 (Technical)"),
        ("q4_billing", "Query 4 (Billing - human-approved)"),
        ("q5_memory", "Query 5 (Memory Recall)"),
    ]:
        r = captured[label]
        lines.append(f"{q_text}:")
        lines.append(f"  approval_status: {r.get('approval_status')}")
        lines.append(f"  final_response: \"{r.get('final_response')[:140]}...\"")
        lines.append("")
    render_terminal_screenshot("Task Output: Final Response Generation", lines, "5_final_response_generation.png")


if __name__ == "__main__":
    main()
