"""
Заготовка для дообучения модели, если accuracy < 95%.

План:
1. Накопить размеченные выписки (CSV с ground_truth)
2. Экспортировать в формат instruction-tuning:
   {"instruction": "...", "input": "описание операции", "output": "category_id"}
3. Fine-tune qwen2.5:3b через LoRA (unsloth / axolotl / ollama create)
4. Заменить OLLAMA_MODEL в .env на fine-tuned версию
"""

from __future__ import annotations

import json
from pathlib import Path


def export_training_pairs(labeled_csv: Path, output_jsonl: Path) -> int:
    """Конвертирует labeled CSV в JSONL для fine-tuning."""
    import csv

    from src.domain.taxonomy import load_taxonomy

    taxonomy = load_taxonomy()
    count = 0
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with open(labeled_csv, encoding="utf-8") as f_in, open(
        output_jsonl, "w", encoding="utf-8"
    ) as f_out:
        for row in csv.DictReader(f_in):
            if not row.get("ground_truth") or float(row["amount"]) >= 0:
                continue
            record = {
                "instruction": taxonomy.prompt_list(),
                "input": row["description"],
                "output": row["ground_truth"],
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count
