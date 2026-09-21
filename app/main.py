import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel, Field

from app import db, rag
from app.agent import graph as agent_graph
from app.config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("triage")

PAGE = Path(__file__).parent / "web" / "index.html"

STEPS = {
    "triage": "Разбираю обращение",
    "gather": "Решаю, что посмотреть",
    "tools": "Смотрю данные и правила",
    "draft": "Готовлю ответ",
    "execute": "Выполняю одобренное",
    "save": "Сохраняю карточку",
}
TOOL_STEPS = {
    "find_client": "Ищу клиента",
    "get_client_cards": "Смотрю карты клиента",
    "get_transactions": "Смотрю операции",
    "search_policies": "Читаю правила банка",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    built = await rag.build_index()
    log.info("индекс правил: %s", f"пересобран, {built} кусков" if built else "готов")
    async with agent_graph.open_graph() as compiled:
        app.state.graph = compiled
        yield


app = FastAPI(title="Триаж обращений", lifespan=lifespan)


class NewTicket(BaseModel):
    text: str = Field(min_length=5, max_length=4000)
    channel: str = "app"
    client_id: int | None = None


class Decision(BaseModel):
    decision: str = "approve"  # approve | edit | reject
    reply: str | None = None
    action_indexes: list[int] | None = None
    operator: str = "оператор"


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


async def run(graph, payload, config):
    """Гонит граф и рассказывает оператору, что происходит. Пауза приходит событием review."""
    try:
        async for chunk in graph.astream(payload, config, stream_mode="updates"):
            for node, update in chunk.items():
                if node == "__interrupt__":
                    yield sse("review", update[0].value)
                    return
                if node in STEPS:
                    yield sse("step", {"text": STEPS[node]})
                for message in (update or {}).get("messages", []):
                    for call in getattr(message, "tool_calls", None) or []:
                        yield sse("step", {"text": TOOL_STEPS.get(call["name"], call["name"])})
                if node == "save":
                    yield sse("done", {"status": update.get("status"),
                                       "ticket_id": config["configurable"]["thread_id"]})
    except Exception as exc:
        log.exception("обработка упала")
        yield sse("error", {"text": f"{type(exc).__name__}: {exc}"})


def stream(generator) -> StreamingResponse:
    return StreamingResponse(generator, media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/tickets")
async def create_ticket(body: NewTicket):
    """Принимает обращение и сразу стримит шаги агента."""
    ticket_id = f"T-{uuid.uuid4().hex[:6]}"
    state = {
        "ticket_id": ticket_id,
        "channel": body.channel,
        "client_id": body.client_id if body.channel == "app" else None,
        "text": body.text,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "messages": [],
    }
    return stream(run(app.state.graph, state, agent_graph.config(ticket_id)))


@app.post("/tickets/{ticket_id}/review")
async def review_ticket(ticket_id: str, body: Decision):
    """Решение оператора. Граф продолжается с паузы, ответ — такой же поток шагов."""
    snapshot = await app.state.graph.aget_state(agent_graph.config(ticket_id))
    if not snapshot.interrupts:
        raise HTTPException(409, f"Обращение {ticket_id} не ждёт решения")
    resume = Command(resume=body.model_dump())
    return stream(run(app.state.graph, resume, agent_graph.config(ticket_id)))


@app.get("/tickets")
async def list_tickets(status: str | None = None):
    """Очередь: сначала те, что ждут решения, внутри — срочные и старые выше."""
    order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    with db.connect() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT id, status, category, urgency, client_id, text, created_at FROM tickets")]
    if status:
        rows = [r for r in rows if r["status"] == status]
    rows.sort(key=lambda r: (r["status"] != "awaiting_review", order.get(r["urgency"], 9), r["created_at"]))
    return {"total": len(rows), "tickets": rows}


@app.get("/tickets/{ticket_id}")
async def get_ticket(ticket_id: str):
    """Карточка обращения. Если оно на паузе, отдаём и то, что ждёт решения."""
    with db.connect() as conn:
        row = conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
    if row is None:
        raise HTTPException(404, f"Обращение {ticket_id} не найдено")

    ticket = dict(row)
    ticket["facts"] = json.loads(ticket["facts"] or "{}")
    ticket["actions"] = json.loads(ticket["actions"] or "[]")
    snapshot = await app.state.graph.aget_state(agent_graph.config(ticket_id))
    ticket["pending"] = snapshot.interrupts[0].value if snapshot.interrupts else None
    return ticket


@app.get("/health")
async def health():
    """Живы ли модель, эмбеддинги и база. Отвечает всегда: детали в полях."""
    s = settings()
    result = {"model": False, "embeddings": False, "database": False}
    async with httpx.AsyncClient(timeout=5.0) as client:
        for name, url, headers in [
            ("model", f"{s.llm_base_url.rstrip('/')}/models", {"Authorization": f"Bearer {s.llm_api_key}"}),
            ("embeddings", f"{s.embed_base_url.rstrip('/')}/api/tags", {}),
        ]:
            try:
                result[name] = (await client.get(url, headers=headers)).status_code == 200
            except httpx.HTTPError as exc:
                result[name] = str(exc)
    try:
        with db.connect() as conn:
            result["database"] = conn.execute("SELECT COUNT(*) FROM clients").fetchone()[0] > 0
    except Exception as exc:
        result["database"] = str(exc)
    return result


@app.get("/", include_in_schema=False)
async def page():
    return FileResponse(PAGE, media_type="text/html; charset=utf-8")
