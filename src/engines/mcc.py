from __future__ import annotations

from pathlib import Path

import yaml

from src.domain.models import Taxonomy
from src.domain.taxonomy import CONFIG_DIR


class MccMapper:
    """MCC-код → категория (0 токенов)."""

    def __init__(self, taxonomy: Taxonomy, path: Path | None = None):
        self.taxonomy = taxonomy
        self._map: dict[str, str] = {}
        config_path = path or CONFIG_DIR / "mcc.yaml"
        with open(config_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        for mcc, category_id in raw.get("mcc", {}).items():
            mcc_key = str(mcc).strip()
            if category_id in taxonomy.categories:
                self._map[mcc_key] = category_id

    def match(self, mcc: str | None) -> str | None:
        if not mcc:
            return None
        code = str(mcc).strip()
        return self._map.get(code)
