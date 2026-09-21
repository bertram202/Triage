import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import settings

DISPUTE_DAYS = 60  # сколько дней можно оспаривать операцию

SCHEMA = """
CREATE TABLE clients (
    id        INTEGER PRIMARY KEY,
    full_name TEXT NOT NULL,
    city      TEXT NOT NULL,
    phone     TEXT NOT NULL          -- маскированный, храним только последние цифры
);

CREATE TABLE cards (
    id           INTEGER PRIMARY KEY,
    client_id    INTEGER NOT NULL REFERENCES clients(id),
    last4        TEXT NOT NULL UNIQUE,
    kind         TEXT NOT NULL,      -- debit | credit
    status       TEXT NOT NULL,      -- active | blocked
    block_reason TEXT,               -- lost | stolen | fraud | client_request
    blocked_at   TEXT
);

CREATE TABLE transactions (
    id         INTEGER PRIMARY KEY,
    card_id    INTEGER NOT NULL REFERENCES cards(id),
    created_at TEXT NOT NULL,        -- 'YYYY-MM-DD HH:MM:SS'
    kind       TEXT NOT NULL,        -- purchase | atm_withdrawal | transfer_out | salary | fee | refund
    amount     REAL NOT NULL,        -- всегда положительная
    merchant   TEXT NOT NULL,
    status     TEXT NOT NULL         -- completed | disputed | refunded
);

CREATE TABLE tickets (
    id          TEXT PRIMARY KEY,
    channel     TEXT NOT NULL,       -- app (клиент известен) | email (клиента ищет агент)
    client_id   INTEGER,
    text        TEXT NOT NULL,
    status      TEXT NOT NULL,       -- processing | awaiting_review | approved | rejected | spam
    category    TEXT,
    urgency     TEXT,
    facts       TEXT,                -- JSON: что агент нашёл и о чём предупредил
    draft_reply TEXT,
    final_reply TEXT,
    actions     TEXT,                -- JSON: предложенные действия
    decision    TEXT,                -- approve | edit | reject
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE actions_log (
    id         INTEGER PRIMARY KEY,
    ticket_id  TEXT NOT NULL,
    action     TEXT NOT NULL,
    target     TEXT NOT NULL,
    operator   TEXT NOT NULL,
    status     TEXT NOT NULL,        -- executed | failed
    result     TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path or settings().db_path))
    conn.row_factory = sqlite3.Row
    return conn


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def days_ago(timestamp: str) -> int:
    return (datetime.now() - datetime.fromisoformat(timestamp)).days


def find_clients(conn, query: str) -> list[dict]:
    words = [w for w in re.findall(r"[а-яё]+", query.lower().replace("ё", "е")) if len(w) >= 3]
    if not words:
        return []
    found = []
    for row in conn.execute("SELECT id, full_name, city FROM clients"):
        name = row["full_name"].lower().replace("ё", "е").split()
        if all(any(part.startswith(w[:3]) for part in name) for w in words):
            found.append(dict(row))
    return found[:5]


def get_client(conn, client_id: int) -> dict | None:
    row = conn.execute("SELECT id, full_name, city FROM clients WHERE id = ?", (client_id,)).fetchone()
    return dict(row) if row else None


def get_cards(conn, client_id: int) -> list[dict]:
    rows = conn.execute(
        "SELECT last4, kind, status, block_reason, blocked_at FROM cards WHERE client_id = ? ORDER BY id",
        (client_id,),
    )
    return [dict(r) for r in rows]


def get_card(conn, last4: str) -> dict | None:
    row = conn.execute("SELECT * FROM cards WHERE last4 = ?", (str(last4),)).fetchone()
    return dict(row) if row else None


def get_transaction(conn, tx_id: int) -> dict | None:
    row = conn.execute(
        "SELECT t.*, c.last4 AS card_last4, c.client_id FROM transactions t "
        "JOIN cards c ON c.id = t.card_id WHERE t.id = ?",
        (tx_id,),
    ).fetchone()
    return dict(row) if row else None


def get_transactions(conn, client_id: int, *, card_last4=None, date_from=None, date_to=None,
                     min_amount=None, max_amount=None, kind=None, merchant=None, limit=20) -> dict:
    where = ["c.client_id = ?"]
    params: list[Any] = [client_id]
    for condition, value in [
        ("c.last4 = ?", card_last4),
        ("t.created_at >= ?", f"{date_from} 00:00:00" if date_from else None),
        ("t.created_at <= ?", f"{date_to} 23:59:59" if date_to else None),
        ("t.amount >= ?", min_amount),
        ("t.amount <= ?", max_amount),
        ("t.kind = ?", kind),
        ("LOWER(t.merchant) LIKE ?", f"%{merchant.lower()}%" if merchant else None),
    ]:
        if value is not None:
            where.append(condition)
            params.append(value)

    rows = [dict(r) for r in conn.execute(
        "SELECT t.id, t.created_at, t.kind, t.amount, t.merchant, t.status, c.last4 AS card_last4 "
        "FROM transactions t JOIN cards c ON c.id = t.card_id "
        f"WHERE {' AND '.join(where)} ORDER BY t.created_at DESC", params)]

    totals: dict[str, float] = {}
    for r in rows:
        r["days_ago"] = days_ago(r["created_at"])
        if r["status"] == "completed":
            totals[r["kind"]] = round(totals.get(r["kind"], 0) + r["amount"], 2)

    limit = min(max(limit, 1), 50)
    return {"found": len(rows), "totals_by_kind": totals, "transactions": rows[:limit]}


# запись(сюда только после одобрения оператором)

BLOCK_REASONS = {"lost": "утеря", "stolen": "кража", "fraud": "мошенничество", "client_request": "просьба клиента"}


def block_card(conn, last4: str, reason: str) -> dict:
    card = get_card(conn, last4)
    if card is None:
        raise ValueError(f"карта *{last4} не найдена")
    if card["status"] != "active":
        raise ValueError(f"карта *{last4} уже заблокирована")
    reason = reason if reason in BLOCK_REASONS else "client_request"
    conn.execute("UPDATE cards SET status = 'blocked', block_reason = ?, blocked_at = ? WHERE id = ?",
                 (reason, now_str(), card["id"]))
    return {"card": last4, "reason": BLOCK_REASONS[reason]}


def open_dispute(conn, tx_id: int) -> dict:
    conn.execute("UPDATE transactions SET status = 'disputed' WHERE id = ?", (tx_id,))
    return {"transaction": tx_id, "status": "оспаривается"}


def refund_fee(conn, tx_id: int) -> dict:
    """Комиссию помечаем возвращённой, а деньги возвращаем встречной операцией — как в выписке банка."""
    tx = get_transaction(conn, tx_id)
    if tx is None:
        raise ValueError(f"операция {tx_id} не найдена")
    conn.execute("INSERT INTO transactions (card_id, created_at, kind, amount, merchant, status) "
                 "VALUES (?, ?, 'refund', ?, ?, 'completed')",
                 (tx["card_id"], now_str(), tx["amount"], f"Возврат: {tx['merchant']}"))
    conn.execute("UPDATE transactions SET status = 'refunded' WHERE id = ?", (tx_id,))
    return {"transaction": tx_id, "amount": tx["amount"]}


def run_action(conn, action: dict, *, ticket_id: str, operator: str) -> dict:
    name = action["name"]
    target = action.get("card_last4") or action.get("transaction_id")
    try:
        if name == "block_card":
            result = block_card(conn, action["card_last4"], action.get("block_reason", "client_request"))
        elif name == "open_dispute":
            result = open_dispute(conn, action["transaction_id"])
        else:
            result = refund_fee(conn, action["transaction_id"])
        status = "executed"
    except (ValueError, TypeError, sqlite3.Error) as exc:
        result, status = {"error": str(exc)}, "failed"

    conn.execute("INSERT INTO actions_log (ticket_id, action, target, operator, status, result, created_at) "
                 "VALUES (?, ?, ?, ?, ?, ?, ?)",
                 (ticket_id, name, str(target), operator, status,
                  json.dumps(result, ensure_ascii=False), now_str()))
    conn.commit()
    return {**action, "status": status, "result": result}


TICKET_COLUMNS = ("id", "channel", "client_id", "text", "status", "category", "urgency", "facts",
                  "draft_reply", "final_reply", "actions", "decision", "created_at", "updated_at")


def save_ticket(conn, ticket: dict) -> None:
    placeholders = ", ".join("?" * len(TICKET_COLUMNS))
    conn.execute(f"INSERT OR REPLACE INTO tickets ({', '.join(TICKET_COLUMNS)}) VALUES ({placeholders})",
                 [ticket.get(c) for c in TICKET_COLUMNS])
    conn.commit()
