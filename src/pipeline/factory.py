from __future__ import annotations

from src.domain.models import Transaction
from src.domain.taxonomy import load_taxonomy
from src.engines.llm import LLMCategorizer
from src.engines.merchant_lookup import MerchantKnowledgeBase
from src.engines.rules import RuleEngine
from src.pipeline.categorizer import ExpenseCategorizer
from src.settings import Settings


def build_categorizer(settings: Settings | None = None, use_llm: bool = True) -> ExpenseCategorizer:
    settings = settings or Settings()
    taxonomy = load_taxonomy()
    merchant_kb = MerchantKnowledgeBase(taxonomy)
    rule_engine = RuleEngine(taxonomy)
    llm = LLMCategorizer(
        taxonomy=taxonomy,
        host=settings.ollama_host,
        model=settings.ollama_model,
        batch_size=settings.llm_batch_size,
        temperature=settings.llm_temperature,
        enable_cache=settings.enable_llm_cache,
    )
    return ExpenseCategorizer(
        taxonomy=taxonomy,
        merchant_kb=merchant_kb,
        rule_engine=rule_engine,
        llm=llm,
        use_llm=use_llm,
    )


def rows_to_transactions(rows: list[dict]) -> list[Transaction]:
    return [
        Transaction(
            id=row["id"],
            date=row["date"],
            amount=float(row["amount"]),
            description=row["description"],
            mcc=row.get("mcc") or None,
            bank_category=row.get("bank_category") or None,
        )
        for row in rows
    ]
