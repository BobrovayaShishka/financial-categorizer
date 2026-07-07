from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from src.domain.models import CategorizationResult, CategorizedTransaction


def compute_metrics(
    predicted: list[CategorizedTransaction],
    ground_truth: dict[str, str],
) -> dict:
    """Accuracy и per-category метрики для расходов."""
    y_true: list[str] = []
    y_pred: list[str] = []
    sources: Counter[str] = Counter()

    for tx in predicted:
        if tx.amount >= 0:
            continue
        truth = ground_truth.get(tx.id)
        if not truth:
            continue
        y_true.append(truth)
        y_pred.append(tx.category_id)
        sources[tx.source.value] += 1

    total = len(y_true)
    if total == 0:
        return {"accuracy": 0.0, "total_expenses": 0}

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / total

    categories = sorted(set(y_true) | set(y_pred))
    per_category: dict[str, dict] = {}
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for t, p in zip(y_true, y_pred):
        confusion[t][p] += 1

    for cat in categories:
        tp = confusion[cat][cat]
        fp = sum(confusion[other][cat] for other in categories if other != cat)
        fn = sum(confusion[cat][other] for other in categories if other != cat)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_category[cat] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(confusion[cat].values()),
        }

    other_rate = y_pred.count("other") / total

    return {
        "accuracy": round(accuracy, 4),
        "total_expenses": total,
        "correct": correct,
        "other_rate": round(other_rate, 4),
        "per_category": per_category,
        "sources": dict(sources),
        "meets_95_target": accuracy >= 0.95,
    }


def build_report(
    result: CategorizationResult,
    ground_truth: dict[str, str],
    mode: str = "rules+llm",
) -> dict:
    metrics = compute_metrics(result.transactions, ground_truth)
    category_totals: Counter[str] = Counter()
    for tx in result.transactions:
        if tx.amount < 0:
            category_totals[tx.category_label] += abs(tx.amount)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "metrics": metrics,
        "token_usage": result.usage.model_dump(),
        "pipeline_stats": {
            "merchant_kb": result.merchant_kb_matched,
            "rules": result.rules_matched,
            "llm": result.llm_matched,
            "skipped_income": result.skipped,
        },
        "spending_by_category": dict(category_totals),
    }


def save_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
