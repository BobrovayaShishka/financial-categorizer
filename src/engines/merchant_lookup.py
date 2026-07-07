from __future__ import annotations

from pathlib import Path

import yaml

from src.domain.models import Taxonomy
from src.domain.taxonomy import CONFIG_DIR


class MerchantKnowledgeBase:
    """Exact merchant lookup — zero LLM tokens."""

    def __init__(self, taxonomy: Taxonomy, path: Path | None = None):
        self.taxonomy = taxonomy
        self._entries: list[tuple[list[str], str]] = []
        config_path = path or CONFIG_DIR / "merchants.yaml"
        with open(config_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        for entry in raw.get("merchants", []):
            patterns = [str(p).lower() for p in entry["patterns"]]
            self._entries.append((patterns, entry["category"]))

    def match(self, description: str) -> str | None:
        text = description.lower()
        for patterns, category_id in self._entries:
            for pattern in patterns:
                if pattern in text:
                    if category_id in self.taxonomy.categories:
                        return category_id
        return None
