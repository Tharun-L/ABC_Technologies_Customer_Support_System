# AI-Powered Customer Support Automation System

A LangGraph-based multi-agent customer support system built for ABC Technologies.
The system classifies incoming customer queries, routes them to the right
specialized department agent, grounds responses in a company knowledge base
(RAG), remembers customer conversation history (SQLite), enforces human
supervisor approval for high-risk requests, and runs every draft response
through an automated Supervisor QA agent before it is sent.

---

## 1. Project Structure

```
customer_support_system/
├── state.py                  # Task 2 — LangGraph State schema
├── knowledge_base.py          # Task 6 — RAG retriever (TF-IDF over 4 docs)
├── memory.py                  # Task 7 — SQLite conversation memory
├── llm_provider.py            # Pluggable LLM backend (real API or offline fallback)
├── agents.py                  # Tasks 3, 5, 9 — Intent classifier, dept agents, Supervisor
├── human_in_the_loop.py       # Task 8 — Human approval gate (LangGraph interrupt)
├── graph.py                   # Tasks 1, 4 — Workflow graph + conditional routing
├── main.py                    # Task 10 — Demo runner for the 5 sample queries
├── generate_diagram.py        # Generates workflow_diagram.png
├── generate_screenshots.py    # Generates the task-output screenshots
├── docs/                       # Knowledge base source documents
│   ├── company_policy.txt
│   ├── pricing_guide.txt
│   ├── technical_manual.txt
│   └── faq_document.txt
├── memory_db/                  # SQLite database files (created at runtime)
│   ├── memory.db                  — conversation history (Task 7 schema)
│   └── graph_checkpoints.db       — LangGraph checkpointer (enables interrupt/resume)
├── screenshots/                # Task output screenshots (PNG)
├── workflow_diagram.png        # LangGraph architecture diagram
├── requirements.txt
└── README.md
```

## 2. Setup

```bash
python3 -m pip install -r requirements.txt
```

That's it. **No API key is required to run the full demo.** See Section 5
for why, and how to plug in a real LLM if you want one.

## 3. Running the Demo

```bash
python3 main.py
```

This runs the five sample queries from the assignment brief (plus one seed
query so memory-recall has something to recall), printing a full trace of:
intent classification → routing → RAG retrieval → draft generation →
(human approval, if triggered) → final response. At the end it prints the
full SQLite conversation history for the demo customer.

To regenerate the workflow diagram or the screenshots:

```bash
python3 generate_diagram.py
python3 generate_screenshots.py
```

## 4. Architecture (Task 1)

```
START
  -> load_memory_node            (Task 7: pulls SQLite history for this customer)
  -> classify_intent_node        (Task 3: Sales / Technical / Billing / Account / Memory_Recall)
  -> [conditional routing]       (Task 4)
       -> sales_agent
       -> technical_agent
       -> billing_agent
       -> account_agent
       -> memory_recall_node     (bypasses department routing entirely)
  -> [conditional routing on risk]
       high-risk?  -> human_approval_node     (Task 8: interrupt() pauses for a human)
       not risky?  -> supervisor_review_node  (Task 9: AI Supervisor QA pass)
  -> save_memory_node            (Task 7: persist the turn to SQLite)
  -> END
```

See `workflow_diagram.png` for the visual version.

### State (Task 2)

`state.py` defines a single `SupportState` TypedDict shared across every
node: customer info, query + classified intent, RAG context, conversation
history, approval status, and the final response. Every node reads what it
needs and returns a partial update that LangGraph merges back in.

### Intent Classification (Task 3)

`agents.classify_intent_node` classifies each query into one of `Sales`,
`Technical`, `Billing`, `Account`, or `Memory_Recall`, and independently
flags high-risk requests (refunds, cancellations, account closure,
compensation, escalation to management) regardless of department.

### Conditional Routing (Task 4)

Two conditional edges in `graph.py`:
- `route_after_classification` sends the query to the matching department
  agent (or straight to memory recall).
- `route_after_agent` sends the drafted response to either the human
  approval gate or the automated Supervisor, based on the high-risk flag.

### Department Agents (Task 5)

`agents.py` builds four agents (Sales, Technical, Billing, Account) from a
shared factory function. Each agent retrieves relevant context from the
knowledge base biased toward its own domain document, then drafts a
response.

### RAG Pipeline (Task 6)

`knowledge_base.py` loads the four knowledge-base documents, splits them
into paragraph-level chunks, and indexes them with TF-IDF (scikit-learn).
Retrieval uses cosine similarity, with a small ranking boost toward the
document most associated with the active department. TF-IDF was chosen
deliberately so the whole RAG pipeline runs **offline and deterministically**
— no embeddings API key needed — while still being a real
retrieve-then-generate pattern.

### SQLite Memory (Task 7)

`memory.py` defines a `customers` table and a `conversation_history` table.
Every turn (query, intent, department, final response, approval status,
timestamp) is saved via `save_turn()`. `get_history()` and `get_last_issue()`
power the memory-recall path. This is separate from LangGraph's own
`SqliteSaver` checkpointer (`memory_db/graph_checkpoints.db`), which exists
purely to make `interrupt()`/resume possible across separate `invoke()` calls
for the human-in-the-loop step — the two SQLite files serve different jobs.

### Human-in-the-Loop (Task 8)

`human_in_the_loop.human_approval_node` calls LangGraph's `interrupt()`,
pausing graph execution and surfacing the draft response + risk reason. The
caller resumes with `Command(resume={"decision": "approve" | "reject" | "edit", ...})`.
This node is only reached for the five high-risk categories called out in
the brief: refunds, subscription cancellation, account closure,
compensation requests, and escalation to management.

### Supervisor Agent (Task 9)

`agents.supervisor_review_node` is an automated QA pass that every
non-high-risk draft goes through before being finalized. It can either
validate the draft as-is or revise it, and is independent from — and
complementary to — the human supervisor gate in Task 8 (which exists for
the higher-stakes categories that should never be auto-approved).

### Demonstration (Task 10)

`main.py` runs the five sample queries end-to-end. Sample output:

| Query | Classified Intent | Routed To | Approval |
|---|---|---|---|
| "What are the pricing plans available for your software?" | Sales | sales_agent | not_required |
| "I forgot my account password." | Account | account_agent | not_required |
| "My application crashes whenever I upload a file." | Technical | technical_agent | not_required |
| "I need a refund for my annual subscription." | Billing | billing_agent | **human approval required** |
| "What was my previous support issue?" | Memory_Recall | memory_recall_node | not_required |

## 5. About the LLM Backend

Every agent node calls a thin wrapper in `llm_provider.py` rather than a
specific vendor SDK directly. By default it runs in **offline mode**: a
deterministic, rule-based "brain" that performs the same jobs (intent
classification, response drafting, supervisor review) the LLM call is asked
to do, using keyword/pattern matching. This means the whole project —
routing, RAG, memory, human-in-the-loop, supervisor QA — is fully demoable
with **zero setup and no API key**.

To use a real hosted model instead, just set an environment variable before
running:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# or
export OPENAI_API_KEY=sk-...
python3 main.py
```

`llm_provider.py` detects the key automatically and routes every
`generate()` call to the real model — no other code changes needed. This
was a deliberate design choice: the graph topology, routing logic, RAG
pipeline, memory schema, and human-in-the-loop gate are all backend-agnostic.

## 6. Database Files

- `memory_db/memory.db` — human-readable conversation history (Task 7
  deliverable). Inspect with:
  ```bash
  sqlite3 memory_db/memory.db "SELECT * FROM conversation_history;"
  ```
- `memory_db/graph_checkpoints.db` — LangGraph's internal state checkpoints,
  required for `interrupt()`/resume to work across calls. Not meant to be
  human-read, but present as proof the graph is checkpointed.

Both files are created automatically on first run. Delete them to reset the
demo to a clean state.

## 7. Known Simplifications

- TF-IDF retrieval is used instead of a hosted embeddings API, for
  reproducibility without external dependencies. The retrieval pattern
  (chunk → index → cosine-similarity retrieve → inject into prompt) is the
  same one used in embedding-based RAG; only the vector representation
  differs.
- The offline LLM fallback uses keyword/pattern rules rather than a neural
  model. It is intentionally simple and documented as such — swapping in a
  real model via `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` is a one-line change.
- The human supervisor decision in `main.py`'s demo run is simulated
  (`decision: approve`) rather than coming from an interactive dashboard,
  since this is a CLI demonstration. The `human_approval_node` itself is a
  real LangGraph `interrupt()` and would work identically behind a real UI.
