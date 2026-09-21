import random
from datetime import datetime, timedelta
from pathlib import Path

from app import db
from app.config import settings

#Демо-данные банка генерятся через python -m app.seed

CLIENTS = [
    (1, "Анна Смирнова", "Москва", "+7 *** ***-12-34"),
    (2, "Игорь Петров", "Казань", "+7 *** ***-45-67"),
    (3, "Ольга Лебедева", "Москва", "+7 *** ***-89-01"),
    (4, "Дмитрий Волков", "Новосибирск", "+7 *** ***-23-45"),
    (5, "Марина Кузнецова", "Санкт-Петербург", "+7 *** ***-67-89"),
    (6, "Сергей Морозов", "Екатеринбург", "+7 *** ***-11-22"),
    (7, "Татьяна Фёдорова", "Самара", "+7 *** ***-33-44"),
    (8, "Павел Никитин", "Москва", "+7 *** ***-55-66"),
    (9, "Елена Зайцева", "Краснодар", "+7 *** ***-77-88"),
    (10, "Роман Соколов", "Пермь", "+7 *** ***-99-00"),
]

# клиент, последние 4 цифры, тип. Первая карта клиента - основная
CARDS = [
    (1, "4821", "debit"), (1, "9043", "credit"),
    (2, "3310", "debit"),
    (3, "8812", "debit"),
    (4, "5507", "debit"),
    (5, "6120", "debit"), (5, "2298", "credit"),
    (6, "7401", "debit"),
    (7, "7777", "debit"),
    (8, "1654", "debit"),
    (9, "9982", "credit"),
    (10, "3045", "debit"),
]

SHOPS = ["PYATEROCHKA", "LENTA", "YANDEX.EDA", "OZON", "WILDBERRIES", "AZBUKA VKUSA",
         "GAZPROMNEFT", "APTEKA STOLICHKI", "MVIDEO", "KOFEMANIYA"]


def at(days_ago: int, hour: int = 13) -> str:
    """Момент столько-то дней назад — в формате, в котором даты лежат в базе."""
    moment = datetime.now() - timedelta(days=days_ago)
    return moment.replace(hour=hour, minute=15, second=0, microsecond=0).strftime("%Y-%m-%d %H:%M:%S")


def first_of_month() -> str:
    return datetime.now().replace(day=1, hour=0, minute=5, second=0, microsecond=0).strftime("%Y-%m-%d %H:%M:%S")


def last_month_day(day: int) -> str:
    first = datetime.now().replace(day=1)
    return (first - timedelta(days=1)).replace(day=day, hour=12, minute=0, second=0,
                                               microsecond=0).strftime("%Y-%m-%d %H:%M:%S")


def planted_transactions() -> list[tuple]:
    """Ситуации, ради которых всё и затевалось. (карта, когда, тип, сумма, получатель).

    Именно их разбирают примеры на странице оператора, поэтому они заданы руками,
    а не выпали случайно.
    """
    return [
        # Игорь Петров: списание, которого он не совершал — кандидат на оспаривание.
        ("3310", at(3), "purchase", 12499.0, "APPLE.COM/BILL"),
        ("3310", at(3, hour=12), "purchase", 1.0, "APPLE.COM/BILL"),
        # Ольга Лебедева: в прошлом месяце потратила больше 10 000, а плату всё равно списали.
        ("8812", last_month_day(6), "purchase", 7300.0, "MVIDEO"),
        ("8812", last_month_day(17), "purchase", 4150.0, "LENTA"),
        ("8812", first_of_month(), "fee", 99.0, "Плата за обслуживание карты"),
        # Дмитрий Волков: снятие в чужом банкомате с комиссией — по правилам всё верно.
        ("5507", at(9), "atm_withdrawal", 15000.0, "ATM ALFA"),
        ("5507", at(9, hour=14), "fee", 150.0, "Комиссия за снятие в чужом банкомате"),
        # Марина Кузнецова: покупка старше срока оспаривания — проверка отсечёт её.
        ("6120", at(95), "purchase", 5400.0, "WILDBERRIES"),
    ]


def background(conn, card_id: int, rnd: random.Random) -> None:
    """Обычная жизнь карты: несколько покупок и зарплата, чтобы выписка не выглядела пустой."""
    for _ in range(rnd.randint(5, 9)):
        conn.execute(
            "INSERT INTO transactions (card_id, created_at, kind, amount, merchant, status) "
            "VALUES (?, ?, 'purchase', ?, ?, 'completed')",
            (card_id, at(rnd.randint(1, 60), rnd.randint(9, 21)),
             round(rnd.uniform(300, 6000), 2), rnd.choice(SHOPS)))
    conn.execute(
        "INSERT INTO transactions (card_id, created_at, kind, amount, merchant, status) "
        "VALUES (?, ?, 'salary', ?, 'ООО Работодатель', 'completed')",
        (card_id, at(rnd.randint(5, 25)), round(rnd.uniform(70000, 160000), 2)))


def build(path: str | Path) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)

    rnd = random.Random(20)
    conn = db.connect(path)
    conn.executescript(db.SCHEMA)
    conn.executemany("INSERT INTO clients (id, full_name, city, phone) VALUES (?, ?, ?, ?)", CLIENTS)
    conn.executemany("INSERT INTO cards (client_id, last4, kind, status) VALUES (?, ?, ?, 'active')", CARDS)

    card_ids = {r["last4"]: r["id"] for r in conn.execute("SELECT id, last4 FROM cards")}
    for last4, when, kind, amount, merchant in planted_transactions():
        conn.execute("INSERT INTO transactions (card_id, created_at, kind, amount, merchant, status) "
                     "VALUES (?, ?, ?, ?, ?, 'completed')", (card_ids[last4], when, kind, amount, merchant))
    for card_id in card_ids.values():
        background(conn, card_id, rnd)

    conn.commit()
    counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ("clients", "cards", "transactions")}
    conn.close()
    return counts


if __name__ == "__main__":
    path = settings().db_path
    counts = build(path)
    print(f"База готова: {path}")
    print(f"  клиентов {counts['clients']}, карт {counts['cards']}, операций {counts['transactions']}")
