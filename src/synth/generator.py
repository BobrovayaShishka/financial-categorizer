from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass
class MerchantTemplate:
    description: str
    category: str
    amount_range: tuple[float, float]
    mcc: str | None = None


# Типовые мерчанты с эталонной категорией
KNOWN_MERCHANTS: list[MerchantTemplate] = [
    MerchantTemplate("PYATEROCHKA 1234 MOSCOW", "groceries", (300, 4500), "5411"),
    MerchantTemplate("PEREKRESTOK SM", "groceries", (500, 6000), "5411"),
    MerchantTemplate("MAGNIT MM 456", "groceries", (200, 3000), "5411"),
    MerchantTemplate("LENTA HYPER", "groceries", (800, 8000), "5411"),
    MerchantTemplate("STARBUCKS COFFEE", "dining", (350, 1200), "5814"),
    MerchantTemplate("DODO PIZZA", "dining", (600, 2500), "5812"),
    MerchantTemplate("YANDEX EDA", "dining", (500, 3500), "5812"),
    MerchantTemplate("YANDEX GO", "transport", (150, 2500), "4121"),
    MerchantTemplate("DELIMOBIL", "transport", (200, 4000), "4121"),
    MerchantTemplate("LUKOIL AZS", "transport", (1500, 6000), "5541"),
    MerchantTemplate("NETFLIX.COM", "subscriptions", (599, 999), "4899"),
    MerchantTemplate("SPOTIFY", "subscriptions", (169, 269), "4899"),
    MerchantTemplate("MTS MOBILE", "subscriptions", (300, 1200), "4814"),
    MerchantTemplate("MEGAFON", "subscriptions", (400, 1500), "4814"),
    MerchantTemplate("RIGLA APTEKA", "health", (200, 3500), "5912"),
    MerchantTemplate("INVITRO", "health", (800, 12000), "8099"),
    MerchantTemplate("OZON.RU", "shopping", (500, 25000), "5399"),
    MerchantTemplate("WILDBERRIES", "shopping", (300, 15000), "5399"),
    MerchantTemplate("LAMODA", "shopping", (1500, 12000), "5651"),
    MerchantTemplate("CINEMA PARK", "entertainment", (400, 2000), "7832"),
    MerchantTemplate("STEAM PURCHASE", "entertainment", (100, 5000), "5816"),
    MerchantTemplate("ZHILKOMUSLUGI", "housing", (2000, 15000), "4900"),
    MerchantTemplate("IPOTEKA PAYMENT", "housing", (15000, 80000), "6513"),
    MerchantTemplate("SKILLBOX", "education", (3000, 50000), "8299"),
    MerchantTemplate("NETOLOGY", "education", (5000, 80000), "8299"),
    MerchantTemplate("SBP P2P TRANSFER", "finance", (500, 50000), "6012"),
    MerchantTemplate("BANK FEE", "finance", (50, 500), "6012"),
    MerchantTemplate("TILDA.CC SUBSCRIPTION", "business", (500, 3000), "7372"),
    MerchantTemplate("FIGMA INC", "business", (900, 2500), "7372"),
    MerchantTemplate("NOTION LABS", "business", (800, 2000), "7372"),
    MerchantTemplate("GITHUB INC", "business", (400, 1500), "7372"),
    MerchantTemplate("YANDEX DIRECT", "business", (1000, 50000), "7311"),
    MerchantTemplate("REG.RU DOMAIN", "business", (200, 5000), "7372"),
    MerchantTemplate("TIMEWEB HOSTING", "business", (300, 3000), "7372"),
]

# «Трудные» операции — без очевидных ключевых слов, для проверки LLM
HARD_CASES: list[MerchantTemplate] = [
    MerchantTemplate("OOO ALFA SERVICE", "business", (5000, 25000)),
    MerchantTemplate("IP IVANOV A.V.", "business", (3000, 30000)),
    MerchantTemplate("CLOUD PAYMENT", "business", (1500, 12000)),
    MerchantTemplate("DESIGN STUDIO LLC", "business", (8000, 40000)),
    MerchantTemplate("MARKETING AGENCY", "business", (10000, 80000)),
    MerchantTemplate("ONLINE SCHOOL PRO", "education", (4000, 35000)),
    MerchantTemplate("FIT CLUB PREMIUM", "health", (3000, 15000)),
    MerchantTemplate("ENTERTAINMENT HALL", "entertainment", (1500, 8000)),
    MerchantTemplate("TRANSFER TO CARD", "finance", (1000, 100000)),
    MerchantTemplate("SERVICE PAYMENT", "other", (200, 5000)),
]


def _random_date(start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


def generate_statement(
    count: int = 250,
    hard_ratio: float = 0.12,
    seed: int = 42,
) -> list[dict]:
    random.seed(seed)
    end = date.today()
    start = end - timedelta(days=90)
    rows: list[dict] = []

    for i in range(count):
        is_hard = random.random() < hard_ratio
        pool = HARD_CASES if is_hard else KNOWN_MERCHANTS
        merchant = random.choice(pool)
        amount = -round(random.uniform(*merchant.amount_range), 2)
        tx_date = _random_date(start, end)

        bank_cat = None
        if not is_hard and random.random() < 0.35:
            bank_cat = ""

        rows.append(
            {
                "id": f"tx_{i + 1:04d}",
                "date": tx_date.isoformat(),
                "amount": amount,
                "description": merchant.description,
                "mcc": merchant.mcc or "",
                "bank_category": bank_cat if bank_cat is not None else "",
                "ground_truth": merchant.category,
            }
        )

    random.shuffle(rows)
    return rows


def save_raw_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "date", "amount", "description", "mcc", "bank_category"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_labeled_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id", "date", "amount", "description", "mcc",
        "bank_category", "ground_truth",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))
