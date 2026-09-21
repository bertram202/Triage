import json
import logging
import re
from datetime import date, datetime
from functools import lru_cache
from typing import Annotated, Any, Literal, TypedDict

from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from app import db
from app.agent import prompts
from app.agent.tools import TOOLS
from app.config import settings

log = logging.getLogger(__name__)


class Ticket(TypedDict, total=False):
    ticket_id: str
    channel: str
    client_id: int | None
    text: str
    created_at: str

    category: str
    urgency: str | None
    summary: str

    messages: Annotated[list, add_messages]
    steps: int

    reply: str
    policy_refs: list[str]
    facts: str
    actions: list[dict]
    issues: list[str]

    review: dict
    final_reply: str | None
    executed: list[dict]
    status: str


class Triage(BaseModel):
    category: Literal["fraud", "dispute", "complaint", "request", "question", "spam"]
    urgency: Literal["low", "medium", "high", "critical"] = Field(
        description="critical — деньги под угрозой прямо сейчас, high — деньги уже списаны, "
                    "medium — жалоба или просьба, low — вопрос")
    card_last4: str | None = Field(None, description="4 цифры карты, если клиент их назвал, иначе null")
    summary: str = Field(description="Суть обращения одним предложением")


class Action(BaseModel):
    name: Literal["block_card", "open_dispute", "refund_fee"]
    card_last4: str | None = Field(None, description="Только для block_card")
    transaction_id: int | None = Field(None, description="Для open_dispute и refund_fee, id из get_transactions")
    block_reason: Literal["lost", "stolen", "fraud", "client_request"] | None = None
    reason: str = Field(description="Зачем это действие, одно предложение")


class Draft(BaseModel):
    client_id: int | None = Field(None, description="id клиента, о котором обращение")
    reply: str = Field(description="Ответ клиенту на «вы», 3–6 предложений, со ссылками вида [dispute]")
    policy_refs: list[str] = Field(default_factory=list, description="Правила, на которые опирается ответ")
    facts: str = Field(description="Для оператора: что нашлось в базе, 1–3 предложения")
    actions: list[Action] = Field(default_factory=list, description="Что сделать, пустой список — ничего")


@lru_cache
def llm():
    s = settings()
    return init_chat_model(f"openai:{s.llm_model}", base_url=s.llm_base_url, api_key=s.llm_api_key,
                           temperature=0, timeout=s.llm_timeout, max_retries=0)


async def ask(schema, messages, retries: int = 1):
    """Ответ модели строго по схеме. function_calling: с ним мой сервер ошибается реже, чем с json_schema.

    Модель иногда отвечает прозой вместо вызова функции. Тогда просим ещё раз, показав, что не так.
    """
    runnable = llm().with_structured_output(schema, method="function_calling", include_raw=True)
    history = list(messages)
    for attempt in range(retries + 1):
        try:
            result = await runnable.ainvoke(history)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        else:
            if result.get("parsed") is not None and result.get("parsing_error") is None:
                return result["parsed"]
            error = str(result.get("parsing_error") or "ответ пришёл текстом, функция не вызвана")
        log.warning("%s: попытка %s не прошла — %s", schema.__name__, attempt + 1, error[:200])
        history = [*messages, HumanMessage(prompts.RETRY_HINT.format(error=error[:300]))]
    return None


CITATION = re.compile(r" ?\[([a-z_]+)\]")
POLICY_TITLE = re.compile(r"^\[([a-z_]+)\] (.+)$", re.MULTILINE)
WORDS = {"completed": "проведена", "disputed": "уже оспаривается", "refunded": "уже возвращена"}


# узлы

async def triage(state: Ticket) -> dict:
    """Категория, срочность и цифры карты — одним вызовом модели."""
    save_card(state, "processing")  # чтобы обращение было видно в очереди, пока агент работает
    answer = await ask(Triage, [
        SystemMessage(prompts.TRIAGE.format(today=date.today())),
        HumanMessage(state["text"]),
    ])
    text = f"Обращение клиента:\n{state['text']}"
    if answer is None:
        return {"category": "question", "urgency": "medium", "summary": state["text"][:150],
                "messages": [HumanMessage(text)]}
    urgency = "critical" if answer.category == "fraud" else answer.urgency  # здесь ошибка дороже всего
    return {"category": answer.category, "urgency": urgency, "summary": answer.summary,
            "messages": [HumanMessage(f"{text}\n\nРазобрано: {answer.category}, "
                                      f"карта *{answer.card_last4 or '—'}")]}


def after_triage(state: Ticket) -> Literal["gather", "save"]:
    """Рекламу дальше не ведём: искать по ней клиента и правила незачем."""
    return "save" if state["category"] == "spam" else "gather"


async def gather(state: Ticket) -> dict:
    """Агент сам решает, что спросить у базы и у правил банка."""
    known = (f"обращение из приложения, клиент {state['client_id']}" if state.get("client_id")
             else "обращение письмом, клиент пока неизвестен")
    system = prompts.GATHER.format(today=date.today(), known=f"{known}; {state.get('summary', '')}")
    try:
        answer = await llm().bind_tools(TOOLS).ainvoke([SystemMessage(system), *state["messages"]])
    except Exception as exc:
        log.warning("сбор фактов прерван: %s", exc)
        return {"steps": settings().max_tool_steps}
    return {"messages": [answer], "steps": state.get("steps", 0) + 1}


def after_gather(state: Ticket) -> Literal["tools", "draft"]:
    """Попросил инструменты — выполняем. Кончились шаги — пишем черновик по тому, что есть."""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None) and state.get("steps", 0) < settings().max_tool_steps:
        return "tools"
    return "draft"


async def write_draft(state: Ticket) -> dict:
    facts = facts_text(state["messages"])
    answer = await ask(Draft, [
        SystemMessage(prompts.DRAFT.format(today=date.today())),
        HumanMessage(f"Обращение: {state['text']}\n\nКатегория: {state['category']}\n\n"
                     f"Собранные факты:\n{facts}"),
    ])
    if answer is None:
        return {"reply": "", "facts": "Черновик не получился, ответьте клиенту вручную.",
                "policy_refs": [], "actions": [], "issues": []}

    titles = dict(POLICY_TITLE.findall(facts))
    client_id = state.get("client_id") or answer.client_id
    with db.connect() as conn:
        actions, issues = check_actions(conn, [a.model_dump() for a in answer.actions], client_id)
    return {"client_id": client_id, "reply": CITATION.sub("", answer.reply).strip(),
            "policy_refs": [titles.get(ref, ref) for ref in answer.policy_refs],
            "facts": answer.facts, "actions": actions, "issues": issues}


def review(state: Ticket) -> dict:
    """Останавливает граф и ждёт человека.

    Карточку пишем до interrupt(): после возобновления узел начнётся заново, а обращение
    на паузе должно быть видно в очереди.
    """
    save_card(state, "awaiting_review")
    answer = interrupt({
        "ticket_id": state["ticket_id"],
        "reply": state.get("reply", ""),
        "facts": state.get("facts", ""),
        "policy_refs": state.get("policy_refs", []),
        "issues": state.get("issues", []),
        "actions": [{"index": i, **a} for i, a in enumerate(state.get("actions", []))],
    }) or {}

    decision = answer.get("decision", "approve")
    reply = answer.get("reply")
    if reply and decision == "approve":
        decision = "edit"
    indexes = answer.get("action_indexes")
    if indexes is None:
        indexes = list(range(len(state.get("actions", []))))
    return {"review": {"decision": decision, "reply": reply, "indexes": indexes,
                       "operator": answer.get("operator", "оператор")}}


def after_review(state: Ticket) -> Literal["execute", "save"]:
    return "save" if state["review"]["decision"] == "reject" else "execute"


async def execute(state: Ticket) -> dict:
    """Выполняет только то, что оператор отметил."""
    decision = state["review"]
    actions = state.get("actions", [])
    chosen = [actions[i] for i in decision["indexes"] if 0 <= i < len(actions)]
    done = []
    if chosen:
        with db.connect() as conn:
            for action in chosen:
                done.append(db.run_action(conn, action, ticket_id=state["ticket_id"],
                                          operator=decision["operator"]))
    return {"executed": done, "final_reply": decision["reply"] or state.get("reply"), "status": "approved"}


async def save(state: Ticket) -> dict:
    """Последний узел на всех путях: и спам, и отказ, и одобрение."""
    if state.get("category") == "spam":
        status = "spam"
    elif state.get("review", {}).get("decision") == "reject":
        status = "rejected"
    else:
        status = state.get("status", "approved")
    save_card(state, status)
    return {"status": status}


# помощники


def facts_text(messages: list) -> str:
    """Переписка агента простым текстом: в вызове draft инструменты не привязаны,
    и история с их вызовами сбивает часть серверов."""
    lines = []
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            lines += [f"Запрос {c['name']}({json.dumps(c['args'], ensure_ascii=False)})" for c in m.tool_calls]
        elif isinstance(m, ToolMessage):
            lines.append(f"Ответ {m.name}:\n{str(m.content)[:3000]}")
        elif isinstance(m, AIMessage) and m.content:
            lines.append(f"Вывод агента:\n{m.content}")
    return "\n\n".join(lines)


def label(tx: dict) -> str:
    return f"№{tx['id']} на {tx['amount']:.0f} ₽, {tx['merchant']}, {tx['created_at'][:10]}"


def check_actions(conn, actions: list[dict], client_id: int | None) -> tuple[list[dict], list[str]]:
    """Проверки, которые делает код, а не модель.

    Решение остаётся за оператором, но чужие карты и просроченные споры до него не доходят:
    вместо действия он видит причину отказа.
    """
    good, issues = [], []
    for action in actions:
        if client_id is None:
            issues.append("клиент не определён, выполнить действия нельзя")
            break

        if action["name"] == "block_card":
            card = db.get_card(conn, action.get("card_last4") or "")
            if card is None or card["client_id"] != client_id:
                issues.append(f"карты *{action.get('card_last4')} у клиента нет")
                continue
            if card["status"] != "active":
                issues.append(f"карта *{card['last4']} уже заблокирована")
                continue
            action["about"] = f"Заблокировать карту *{card['last4']}"
        else:
            tx = db.get_transaction(conn, action.get("transaction_id") or 0)
            if tx is None or tx["client_id"] != client_id:
                issues.append(f"операции {action.get('transaction_id')} у клиента нет")
                continue
            if tx["status"] != "completed":
                issues.append(f"операция №{tx['id']} {WORDS[tx['status']]}")
                continue
            if action["name"] == "open_dispute":
                if db.days_ago(tx["created_at"]) > db.DISPUTE_DAYS:
                    issues.append(f"операции №{tx['id']} больше {db.DISPUTE_DAYS} дней, срок оспаривания истёк")
                    continue
                action["about"] = f"Открыть спор по операции {label(tx)}"
            else:
                if tx["kind"] != "fee":
                    issues.append(f"операция №{tx['id']} — не комиссия банка, возвращать нечего")
                    continue
                action["about"] = f"Вернуть комиссию {label(tx)}"
        good.append(action)
    return good, issues


def save_card(state: Ticket, status: str) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    card: dict[str, Any] = {
        "id": state["ticket_id"],
        "channel": state["channel"],
        "client_id": state.get("client_id"),
        "text": state["text"],
        "status": status,
        "category": state.get("category"),
        "urgency": state.get("urgency"),
        "facts": json.dumps({"summary": state.get("facts"), "policy_refs": state.get("policy_refs", []),
                             "issues": state.get("issues", []), "executed": state.get("executed", [])},
                            ensure_ascii=False, default=str),
        "draft_reply": state.get("reply"),
        "final_reply": state.get("final_reply") if status == "approved" else None,
        "actions": json.dumps(state.get("actions", []), ensure_ascii=False),
        "decision": state.get("review", {}).get("decision"),
        "created_at": state.get("created_at", now),
        "updated_at": now,
    }
    with db.connect() as conn:
        db.save_ticket(conn, card)
