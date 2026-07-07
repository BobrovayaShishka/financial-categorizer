from __future__ import annotations

from src.domain.models import Taxonomy


class RuleEngine:
    """Keyword-based categorization — zero LLM tokens."""

    def __init__(self, taxonomy: Taxonomy):
        self.taxonomy = taxonomy

    def match(self, description: str) -> str | None:
        text = description.lower()
        best_category: str | None = None
        best_score = 0

        for cid, info in self.taxonomy.categories.items():
            if cid == "other":
                continue
            score = sum(1 for kw in info.keywords if kw in text)
            if score > best_score:
                best_score = score
                best_category = cid

        return best_category if best_score > 0 else None
