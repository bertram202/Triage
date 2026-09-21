"""API без модели: очередь, карточка, коды ошибок, страница оператора."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import db, main, seed
from app.config import settings


def a_ticket(ticket_id: str, **changes) -> dict:
    ticket = {
        "id": ticket_id, "channel": "app", "client_id": 2, "text": "Списали 12 499 ₽, я ничего не покупал",
        "status": "awaiting_review", "category": "dispute", "urgency": "high",
        "facts": json.dumps({"summary": "нашлась операция №1", "policy_refs": ["dispute"], "issues": []}),
        "draft_reply": "черновик", "final_reply": None,
        "actions": json.dumps([{"name": "open_dispute", "transaction_id": 1, "about": "Открыть спор"}]),
        "decision": None, "created_at": "2026-09-20T10:00:00", "updated_at": "2026-09-20T10:00:05",
    }
    return {**ticket, **changes}


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "bank.sqlite3"))
    monkeypatch.setenv("CHECKPOINTS_PATH", str(tmp_path / "checkpoints.sqlite3"))
    monkeypatch.setenv("INDEX_PATH", str(tmp_path / "policies.sqlite3"))
    settings.cache_clear()
    seed.build(tmp_path / "bank.sqlite3")

    async def no_index(force=False):
        return 0  # индекс правил трогать не будем: для него нужна модель эмбеддингов

    monkeypatch.setattr(main.rag, "build_index", no_index)
    with db.connect(tmp_path / "bank.sqlite3") as conn:
        db.save_ticket(conn, a_ticket("T-wait"))
        db.save_ticket(conn, a_ticket("T-done", status="approved", final_reply="ответ", decision="approve"))
    with TestClient(main.app) as test_client:
        yield test_client
    settings.cache_clear()


def test_queue_puts_waiting_first(client):
    data = client.get("/tickets").json()
    assert data["total"] == 2
    assert data["tickets"][0]["id"] == "T-wait"
    assert client.get("/tickets?status=approved").json()["total"] == 1


def test_card_unpacks_json_fields(client):
    card = client.get("/tickets/T-wait").json()
    assert card["facts"]["policy_refs"] == ["dispute"]
    assert card["actions"][0]["about"] == "Открыть спор"
    assert card["pending"] is None  # обращение попало в базу напрямую, паузы в графе нет


def test_missing_ticket_and_wrong_review(client):
    assert client.get("/tickets/T-nope").status_code == 404
    assert client.post("/tickets/T-wait/review", json={"decision": "approve"}).status_code == 409


def test_empty_text_is_rejected(client):
    assert client.post("/tickets", json={"text": ""}).status_code == 422


def test_page_is_served(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "/tickets" in page.text
    assert "<script src" not in page.text  # страница без сборки и без внешних скриптов
