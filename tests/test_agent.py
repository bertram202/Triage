"""Проверки действий и развилки графа. Модель здесь не нужна."""

import asyncio
from pathlib import Path

import pytest
from langchain.messages import AIMessage, HumanMessage

from app import db, seed
from app.agent import nodes


@pytest.fixture
def conn(tmp_path: Path):
    path = tmp_path / "bank.sqlite3"
    seed.build(path)
    connection = db.connect(path)
    yield connection
    connection.close()


def check(conn, action, client_id=1):
    return nodes.check_actions(conn, [action], client_id)


def test_block_own_active_card(conn):
    good, issues = check(conn, {"name": "block_card", "card_last4": "4821", "reason": "украли"})
    assert issues == []
    assert good[0]["about"] == "Заблокировать карту *4821"


def test_cannot_block_foreign_card(conn):
    good, issues = check(conn, {"name": "block_card", "card_last4": "3310", "reason": "украли"})
    assert good == [] and "у клиента нет" in issues[0]


def test_cannot_block_twice(conn):
    db.block_card(conn, "4821", "stolen")
    good, issues = check(conn, {"name": "block_card", "card_last4": "4821", "reason": "ещё раз"})
    assert good == [] and "уже заблокирована" in issues[0]


def test_dispute_only_within_window(conn):
    # У Марины есть покупка старше 60 дней — оспорить её нельзя.
    old = conn.execute("SELECT t.id FROM transactions t JOIN cards c ON c.id = t.card_id "
                       "WHERE c.last4 = '6120' ORDER BY t.created_at LIMIT 1").fetchone()["id"]
    good, issues = check(conn, {"name": "open_dispute", "transaction_id": old, "reason": "не я"}, client_id=5)
    assert good == [] and "срок оспаривания истёк" in issues[0]


def test_refund_only_for_fees(conn):
    purchase = conn.execute("SELECT t.id FROM transactions t JOIN cards c ON c.id = t.card_id "
                            "WHERE c.last4 = '4821' AND t.kind = 'purchase' LIMIT 1").fetchone()["id"]
    good, issues = check(conn, {"name": "refund_fee", "transaction_id": purchase, "reason": "верните"})
    assert good == [] and "не комиссия банка" in issues[0]


def test_nothing_runs_without_client(conn):
    good, issues = nodes.check_actions(conn, [{"name": "block_card", "card_last4": "4821"}], None)
    assert good == [] and "клиент не определён" in issues[0]


def test_spam_goes_straight_to_save():
    assert nodes.after_triage({"category": "spam"}) == "save"
    assert nodes.after_triage({"category": "fraud"}) == "gather"


def test_gather_stops_after_step_limit():
    asked = AIMessage(content="", tool_calls=[{"name": "search_policies", "args": {}, "id": "1"}])
    assert nodes.after_gather({"messages": [asked], "steps": 1}) == "tools"
    assert nodes.after_gather({"messages": [asked], "steps": 99}) == "draft"
    assert nodes.after_gather({"messages": [AIMessage("готово")], "steps": 1}) == "draft"


def test_rejected_ticket_skips_execute():
    assert nodes.after_review({"review": {"decision": "reject"}}) == "save"
    assert nodes.after_review({"review": {"decision": "approve"}}) == "execute"


def test_refund_creates_opposite_transaction(conn):
    fee = conn.execute("SELECT id, amount FROM transactions WHERE kind = 'fee' LIMIT 1").fetchone()
    db.refund_fee(conn, fee["id"])
    assert db.get_transaction(conn, fee["id"])["status"] == "refunded"
    back = conn.execute("SELECT amount FROM transactions WHERE kind = 'refund'").fetchone()
    assert back["amount"] == fee["amount"]


def test_failed_action_is_logged_not_raised(conn):
    db.block_card(conn, "4821", "stolen")
    result = db.run_action(conn, {"name": "block_card", "card_last4": "4821"},
                           ticket_id="T-1", operator="Тестов")
    assert result["status"] == "failed"
    logged = conn.execute("SELECT action, operator, status FROM actions_log").fetchone()
    assert tuple(logged) == ("block_card", "Тестов", "failed")


def test_ticket_text_survives_failed_triage(monkeypatch):
    async def broken(schema, messages):
        return None

    monkeypatch.setattr(nodes, "ask", broken)
    monkeypatch.setattr(nodes, "save_card", lambda *a, **k: None)
    out = asyncio.run(nodes.triage({"ticket_id": "T-1", "channel": "app", "client_id": 1,
                                    "text": "Украли карту 4821", "messages": []}))
    assert out["category"] == "question"
    assert "Украли карту 4821" in out["messages"][0].content


def test_ask_retries_when_model_answers_with_prose(monkeypatch):
    attempts = []

    class Runnable:
        async def ainvoke(self, history):
            attempts.append(history)
            if len(attempts) == 1:
                return {"parsed": None, "parsing_error": None}
            return {"parsed": nodes.Triage(category="fraud", urgency="critical", summary="кража"),
                    "parsing_error": None}

    class Model:
        def with_structured_output(self, schema, **kwargs):
            return Runnable()

    monkeypatch.setattr(nodes, "llm", lambda: Model())
    answer = asyncio.run(nodes.ask(nodes.Triage, [HumanMessage("украли деньги")]))
    assert answer.category == "fraud"
    assert len(attempts) == 2
    assert "строго по схеме" in attempts[1][-1].content


def test_ask_gives_up_after_retry(monkeypatch):
    class Runnable:
        async def ainvoke(self, history):
            return {"parsed": None, "parsing_error": None}

    class Model:
        def with_structured_output(self, schema, **kwargs):
            return Runnable()

    monkeypatch.setattr(nodes, "llm", lambda: Model())
    assert asyncio.run(nodes.ask(nodes.Triage, [HumanMessage("текст")])) is None
