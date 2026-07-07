from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class CategorizationSource(str, Enum):
    MERCHANT_KB = "merchant_kb"
    RULES = "rules"
    LLM = "llm"
    BANK = "bank"
    SKIPPED = "skipped"


class Transaction(BaseModel):
    id: str
    date: str
    amount: float
    description: str
    mcc: str | None = None
    bank_category: str | None = None


class CategorizedTransaction(Transaction):
    category_id: str
    category_label: str
    confidence: float = 1.0
    source: CategorizationSource = CategorizationSource.RULES


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    cached_hits: int = 0

    def add(self, other: TokenUsage) -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        self.llm_calls += other.llm_calls
        self.cached_hits += other.cached_hits


class CategorizationResult(BaseModel):
    transactions: list[CategorizedTransaction]
    usage: TokenUsage = Field(default_factory=TokenUsage)
    rules_matched: int = 0
    merchant_kb_matched: int = 0
    llm_matched: int = 0
    skipped: int = 0


class CategoryInfo(BaseModel):
    id: str
    label: str
    keywords: list[str] = Field(default_factory=list)


class Taxonomy(BaseModel):
    categories: dict[str, CategoryInfo]

    def label_for(self, category_id: str) -> str:
        cat = self.categories.get(category_id)
        return cat.label if cat else category_id

    def category_ids(self) -> list[str]:
        return list(self.categories.keys())

    def prompt_list(self) -> str:
        lines = []
        for cid, info in self.categories.items():
            if cid == "other":
                continue
            lines.append(f"{cid}: {info.label}")
        return "\n".join(lines)
