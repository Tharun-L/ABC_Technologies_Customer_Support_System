"""
memory.py
---------
Task 7: SQLite-based memory to store and retrieve customer conversation history.

This is a dedicated, human-readable SQLite schema (separate from LangGraph's
own checkpointer, which we also use for graph-state persistence/interrupts).
This module is what satisfies the assignment's explicit requirement:
"The system should maintain customer conversation history using SQLite memory."

Example:
    Customer: "My name is David. I have a billing issue."
    Customer (later): "What was my previous issue?"
    -> the system retrieves the stored turn and answers from it.
"""

from __future__ import annotations
import sqlite3
import os
import datetime
from typing import List, Dict, Any, Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "memory_db", "memory.db")


SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id    TEXT PRIMARY KEY,
    customer_name  TEXT,
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conversation_history (
    turn_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id     TEXT NOT NULL,
    customer_name   TEXT,
    query           TEXT NOT NULL,
    intent          TEXT,
    department      TEXT,
    final_response  TEXT,
    approval_status TEXT,
    timestamp       TEXT NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
);
"""


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db():
    conn = _connect()
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def upsert_customer(customer_id: str, customer_name: Optional[str] = None):
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT customer_id, customer_name FROM customers WHERE customer_id = ?", (customer_id,))
        row = cur.fetchone()
        now = datetime.datetime.utcnow().isoformat()
        if row is None:
            cur.execute(
                "INSERT INTO customers (customer_id, customer_name, created_at) VALUES (?, ?, ?)",
                (customer_id, customer_name, now),
            )
        elif customer_name and not row[1]:
            # Backfill name if we learn it later in the conversation
            cur.execute(
                "UPDATE customers SET customer_name = ? WHERE customer_id = ?",
                (customer_name, customer_id),
            )
        conn.commit()
    finally:
        conn.close()


def save_turn(
    customer_id: str,
    query: str,
    intent: Optional[str] = None,
    department: Optional[str] = None,
    final_response: Optional[str] = None,
    approval_status: Optional[str] = None,
    customer_name: Optional[str] = None,
) -> int:
    """Persist one conversational turn and return its turn_id."""
    upsert_customer(customer_id, customer_name)
    conn = _connect()
    try:
        cur = conn.cursor()
        now = datetime.datetime.utcnow().isoformat()
        cur.execute(
            """INSERT INTO conversation_history
               (customer_id, customer_name, query, intent, department,
                final_response, approval_status, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (customer_id, customer_name, query, intent, department,
             final_response, approval_status, now),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_history(customer_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Return the most recent turns for a customer, oldest first."""
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute(
            """SELECT turn_id, query, intent, department, final_response,
                      approval_status, timestamp
               FROM conversation_history
               WHERE customer_id = ?
               ORDER BY turn_id DESC
               LIMIT ?""",
            (customer_id, limit),
        )
        rows = cur.fetchall()
        cols = ["turn_id", "query", "intent", "department", "final_response",
                "approval_status", "timestamp"]
        history = [dict(zip(cols, row)) for row in rows]
        return list(reversed(history))  # oldest first
    finally:
        conn.close()


def get_customer_name(customer_id: str) -> Optional[str]:
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT customer_name FROM customers WHERE customer_id = ?", (customer_id,))
        row = cur.fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def get_last_issue(customer_id: str) -> Optional[Dict[str, Any]]:
    """Used for 'What was my previous issue?' style memory-recall queries."""
    history = get_history(customer_id, limit=50)
    if not history:
        return None
    # Last issue = most recent turn before the current (recall) query
    return history[-1]


# Initialize schema on import so every module that uses memory.* is safe.
init_db()
