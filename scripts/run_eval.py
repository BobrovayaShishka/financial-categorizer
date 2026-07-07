"""Evaluate categorizer accuracy against labeled data."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics.evaluate import build_report, save_report
from src.pipeline.factory import build_categorizer, rows_to_transactions
from src.settings import Settings
from src.synth.generator import load_csv

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


async def run_eval(use_llm: bool, labeled_path: Path) -> None:
    settings = Settings()
    rows = load_csv(labeled_path)
    ground_truth = {r["id"]: r["ground_truth"] for r in rows if r.get("ground_truth")}
    transactions = rows_to_transactions(rows)

    categorizer = build_categorizer(settings, use_llm=use_llm)
    mode = "rules+merchant_kb+llm" if use_llm else "rules+merchant_kb"
    result = await categorizer.categorize(transactions)

    report = build_report(result, ground_truth, mode=mode)
    report_path = DATA_DIR / "reports" / "latest.json"
    save_report(report, report_path)

    m = report["metrics"]
    print(f"Mode: {mode}")
    print(f"Accuracy: {m['accuracy']:.1%} ({m['correct']}/{m['total_expenses']})")
    print(f"Other rate: {m['other_rate']:.1%}")
    print(f"95% target: {'PASS' if m['meets_95_target'] else 'FAIL'}")
    print(f"Sources: {m['sources']}")
    print(f"Tokens: {report['token_usage']}")
    print(f"Report: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate categorizer")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Evaluate rules+merchant_kb only (no Ollama needed)",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=DATA_DIR / "labeled" / "statement_labeled.csv",
    )
    args = parser.parse_args()
    asyncio.run(run_eval(use_llm=not args.no_llm, labeled_path=args.file))


if __name__ == "__main__":
    main()
