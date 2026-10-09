-- memory_db/schema.sql
-- Task 7 deliverable: SQLite schema for customer conversation memory.
-- This is the same schema created automatically by memory.py at runtime;
-- provided here standalone so it can be reviewed/run independently of
-- the Python application (e.g. `sqlite3 memory.db < schema.sql`).

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

-- Example queries used by the application (memory.py):

-- Insert / update a customer record
-- INSERT INTO customers (customer_id, customer_name, created_at)
-- VALUES ('cust_demo_david', 'David', '2026-06-28T00:00:00');

-- Save a conversational turn
-- INSERT INTO conversation_history
--   (customer_id, customer_name, query, intent, department,
--    final_response, approval_status, timestamp)
-- VALUES
--   ('cust_demo_david', 'David', 'I need a refund', 'Billing', 'Billing',
--    'Your refund is being reviewed.', 'pending', '2026-06-28T00:01:00');

-- Retrieve a customer's most recent history (oldest first)
-- SELECT turn_id, query, intent, department, final_response, approval_status, timestamp
-- FROM conversation_history
-- WHERE customer_id = 'cust_demo_david'
-- ORDER BY turn_id DESC
-- LIMIT 20;
