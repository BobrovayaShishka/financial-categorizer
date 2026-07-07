"""Generate synthetic bank statements."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.synth.generator import generate_statement, save_labeled_csv, save_raw_csv

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic bank statement")
    parser.add_argument("--count", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hard-ratio", type=float, default=0.12)
    args = parser.parse_args()

    rows = generate_statement(count=args.count, hard_ratio=args.hard_ratio, seed=args.seed)
    raw_path = DATA_DIR / "raw" / "statement.csv"
    labeled_path = DATA_DIR / "labeled" / "statement_labeled.csv"

    save_raw_csv(rows, raw_path)
    save_labeled_csv(rows, labeled_path)

    print(f"Generated {len(rows)} transactions")
    print(f"  Raw:     {raw_path}")
    print(f"  Labeled: {labeled_path}")


if __name__ == "__main__":
    main()
