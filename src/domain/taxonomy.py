from __future__ import annotations

from pathlib import Path

import yaml

from src.domain.models import CategoryInfo, Taxonomy

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def load_taxonomy(path: Path | None = None) -> Taxonomy:
    config_path = path or CONFIG_DIR / "categories.yaml"
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    categories = {}
    for cid, data in raw["categories"].items():
        categories[cid] = CategoryInfo(
            id=cid,
            label=data["label"],
            keywords=[str(k).lower() for k in data.get("keywords", [])],
        )
    return Taxonomy(categories=categories)
