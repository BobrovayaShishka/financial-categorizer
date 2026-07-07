from __future__ import annotations

from src.domain.models import (
    CategorizationResult,
    CategorizationSource,
    CategorizedTransaction,
    Taxonomy,
    TokenUsage,
    Transaction,
)
from src.engines.llm import LLMCategorizer
from src.engines.merchant_lookup import MerchantKnowledgeBase
from src.engines.rules import RuleEngine


class ExpenseCategorizer:
    """
    Трёхуровневый пайплайн (экономия токенов):
    1. База мерчантов (0 токенов)
    2. Правила по ключевым словам (0 токенов)
    3. LLM батчами только для оставшихся
    """

    def __init__(
        self,
        taxonomy: Taxonomy,
        merchant_kb: MerchantKnowledgeBase,
        rule_engine: RuleEngine,
        llm: LLMCategorizer | None = None,
        use_llm: bool = True,
    ):
        self.taxonomy = taxonomy
        self.merchant_kb = merchant_kb
        self.rule_engine = rule_engine
        self.llm = llm
        self.use_llm = use_llm and llm is not None

    async def categorize(self, transactions: list[Transaction]) -> CategorizationResult:
        result = CategorizationResult(transactions=[])
        pending_llm: list[tuple[str, str, Transaction]] = []

        for tx in transactions:
            if tx.amount >= 0:
                result.transactions.append(
                    CategorizedTransaction(
                        **tx.model_dump(),
                        category_id="income",
                        category_label="Доход",
                        confidence=1.0,
                        source=CategorizationSource.SKIPPED,
                    )
                )
                result.skipped += 1
                continue

            if tx.bank_category:
                cat_id = self._map_bank_category(tx.bank_category)
                if cat_id:
                    result.transactions.append(
                        CategorizedTransaction(
                            **tx.model_dump(),
                            category_id=cat_id,
                            category_label=self.taxonomy.label_for(cat_id),
                            confidence=0.95,
                            source=CategorizationSource.BANK,
                        )
                    )
                    continue

            merchant_cat = self.merchant_kb.match(tx.description)
            if merchant_cat:
                result.transactions.append(
                    CategorizedTransaction(
                        **tx.model_dump(),
                        category_id=merchant_cat,
                        category_label=self.taxonomy.label_for(merchant_cat),
                        confidence=0.98,
                        source=CategorizationSource.MERCHANT_KB,
                    )
                )
                result.merchant_kb_matched += 1
                continue

            rule_cat = self.rule_engine.match(tx.description)
            if rule_cat:
                result.transactions.append(
                    CategorizedTransaction(
                        **tx.model_dump(),
                        category_id=rule_cat,
                        category_label=self.taxonomy.label_for(rule_cat),
                        confidence=0.85,
                        source=CategorizationSource.RULES,
                    )
                )
                result.rules_matched += 1
                continue

            pending_llm.append((tx.id, tx.description, tx))

        if pending_llm and self.use_llm:
            llm_items = [(tx_id, desc) for tx_id, desc, _ in pending_llm]
            llm_results, usage = await self.llm.categorize_batch(llm_items)
            result.usage.add(usage)

            llm_map = {r["id"]: r for r in llm_results}
            for tx_id, _, tx in pending_llm:
                llm_item = llm_map.get(tx_id, {"category": "other", "confidence": 0.3})
                cat_id = llm_item["category"]
                result.transactions.append(
                    CategorizedTransaction(
                        **tx.model_dump(),
                        category_id=cat_id,
                        category_label=self.taxonomy.label_for(cat_id),
                        confidence=float(llm_item.get("confidence", 0.7)),
                        source=CategorizationSource.LLM,
                    )
                )
                result.llm_matched += 1
        else:
            for _, _, tx in pending_llm:
                result.transactions.append(
                    CategorizedTransaction(
                        **tx.model_dump(),
                        category_id="other",
                        category_label=self.taxonomy.label_for("other"),
                        confidence=0.0,
                        source=CategorizationSource.LLM if self.use_llm else CategorizationSource.RULES,
                    )
                )

        return result

    def _map_bank_category(self, bank_category: str) -> str | None:
        text = bank_category.lower()
        mapping = {
            "продукт": "groceries",
            "ресторан": "dining",
            "кафе": "dining",
            "транспорт": "transport",
            "такси": "transport",
            "связь": "subscriptions",
            "подписк": "subscriptions",
            "аптек": "health",
            "медиц": "health",
            "одежд": "shopping",
            "маркет": "shopping",
            "развлеч": "entertainment",
            "жкх": "housing",
            "коммунал": "housing",
            "образован": "education",
            "перевод": "finance",
            "комисс": "finance",
        }
        for key, cat_id in mapping.items():
            if key in text:
                return cat_id
        return None
