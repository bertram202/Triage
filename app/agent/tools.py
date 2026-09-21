import json
from typing import Annotated, Literal

from langchain.tools import tool
from langgraph.prebuilt import InjectedState

from app import db, rag


def _dump(data) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


def _not_my_client(state: dict, client_id: int) -> str | None:
    """Обращение из приложения — значит клиент известен, и чужие данные агенту не отдаём."""
    known = state.get("client_id")
    if known and client_id != known:
        return f"Отказано: обращение от клиента {known}, данные клиента {client_id} недоступны."
    return None


@tool
async def find_client(query: str, state: Annotated[dict, InjectedState]) -> str:
    """Ищет клиента банка по имени и фамилии.

    Нужен, когда обращение пришло письмом и клиент представился. Если обращение из приложения,
    клиент уже известен и искать его не нужно.

    Args:
        query: имя и фамилия в любом падеже, например «Татьяна Фёдорова».
    """
    with db.connect() as conn:
        found = db.find_clients(conn, query)
    return _dump({"clients": found})


@tool
async def get_client_cards(client_id: int, state: Annotated[dict, InjectedState]) -> str:
    """Возвращает карты клиента: последние 4 цифры, тип (debit или credit), статус (active или blocked),
    причину и дату блокировки.

    Смотри сюда, если клиент не назвал цифры карты и перед тем, как предлагать блокировку.

    Args:
        client_id: id клиента из обращения или из find_client.
    """
    if refusal := _not_my_client(state, client_id):
        return refusal
    with db.connect() as conn:
        client = db.get_client(conn, client_id)
        if client is None:
            return f"Клиента с id={client_id} нет."
        return _dump({"client": client, "cards": db.get_cards(conn, client_id)})


@tool
async def get_transactions(
    client_id: int,
    state: Annotated[dict, InjectedState],
    card_last4: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    min_amount: float | None = None,
    max_amount: float | None = None,
    kind: Literal["purchase", "atm_withdrawal", "transfer_out", "salary", "fee"] | None = None,
    merchant: str | None = None,
    limit: int = 20,
) -> str:
    """Возвращает операции клиента, новые сначала, с фильтрами.

    У каждой операции есть id — по нему предлагаются действия open_dispute и refund_fee.
    days_ago — сколько дней назад она прошла. Поле totals_by_kind содержит суммы по типам
    операций по всем найденным строкам: бери его, а не складывай суммы сам.

    Args:
        client_id: id клиента.
        card_last4: последние 4 цифры, если нужна одна карта.
        date_from: начало периода, ГГГГ-ММ-ДД.
        date_to: конец периода, ГГГГ-ММ-ДД.
        min_amount: минимальная сумма в рублях.
        max_amount: максимальная сумма в рублях.
        kind: purchase — покупка, atm_withdrawal — снятие наличных, transfer_out — перевод,
            fee — комиссия банка, salary — зарплата.
        merchant: часть названия получателя, например «OZON» или «ATM».
        limit: сколько показать, до 50.

    Например: get_transactions(client_id=2, min_amount=12000, max_amount=13000) — найти списание
    примерно на 12 500 ₽; get_transactions(client_id=3, kind="fee") — все комиссии клиента.
    """
    if refusal := _not_my_client(state, client_id):
        return refusal
    with db.connect() as conn:
        result = db.get_transactions(conn, client_id, card_last4=card_last4, date_from=date_from,
                                     date_to=date_to, min_amount=min_amount, max_amount=max_amount,
                                     kind=kind, merchant=merchant, limit=limit)
    return _dump(result)


@tool
async def search_policies(query: str) -> str:
    """Ищет правила банка «Демобанк» и возвращает до трёх документов целиком.

    Смотри сюда за условиями, сроками, лимитами и комиссиями, прежде чем отвечать клиенту
    или предлагать действие. В ответе ссылайся на правила по идентификатору в квадратных
    скобках, например [dispute].

    Args:
        query: суть ситуации своими словами, по-русски. Короткая фраза работает лучше,
            чем текст обращения целиком.
    """
    found = await rag.search(query)
    if not found:
        return "Подходящих правил не нашлось."
    return "Правила банка:\n\n" + "\n\n---\n\n".join(
        f"[{p['doc_id']}] {p['title']}\n{p['text']}" for p in found)


TOOLS = [find_client, get_client_cards, get_transactions, search_policies]
