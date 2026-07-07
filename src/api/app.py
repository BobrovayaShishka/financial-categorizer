from __future__ import annotations

import io
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.domain.models import Transaction
from src.metrics.evaluate import build_report
from src.pipeline.factory import build_categorizer, rows_to_transactions
from src.settings import Settings
from src.synth.generator import load_csv

settings = Settings()
app = FastAPI(title="Expense Categorizer", version="0.1.0")

STATIC_DIR = Path(__file__).parent / "static"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")


class CategorizeRequest(BaseModel):
    transactions: list[Transaction]
    use_llm: bool = True


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model": settings.ollama_model}


@app.get("/categories")
async def categories() -> dict:
    from src.domain.taxonomy import load_taxonomy
    taxonomy = load_taxonomy()
    return {
        cid: info.label for cid, info in taxonomy.categories.items()
    }


@app.post("/categorize")
async def categorize(body: CategorizeRequest) -> JSONResponse:
    categorizer = build_categorizer(settings, use_llm=body.use_llm)
    result = await categorizer.categorize(body.transactions)
    return JSONResponse(
        {
            "transactions": [t.model_dump() for t in result.transactions],
            "usage": result.usage.model_dump(),
            "stats": {
                "merchant_kb": result.merchant_kb_matched,
                "rules": result.rules_matched,
                "llm": result.llm_matched,
                "skipped": result.skipped,
            },
        }
    )


@app.post("/categorize/upload")
async def categorize_upload(
    file: UploadFile = File(...),
    use_llm: bool = True,
) -> JSONResponse:
    content = await file.read()
    text = content.decode("utf-8-sig")
    import csv
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    transactions = rows_to_transactions(rows)
    categorizer = build_categorizer(settings, use_llm=use_llm)
    result = await categorizer.categorize(transactions)
    return JSONResponse(
        {
            "transactions": [t.model_dump() for t in result.transactions],
            "usage": result.usage.model_dump(),
            "stats": {
                "merchant_kb": result.merchant_kb_matched,
                "rules": result.rules_matched,
                "llm": result.llm_matched,
                "skipped": result.skipped,
            },
        }
    )


@app.post("/demo/evaluate")
async def demo_evaluate(use_llm: bool = True) -> JSONResponse:
    labeled_path = Path(__file__).resolve().parents[2] / "data" / "labeled" / "statement_labeled.csv"
    if not labeled_path.exists():
        return JSONResponse({"error": "Run scripts/generate_data.py first"}, status_code=404)

    rows = load_csv(labeled_path)
    ground_truth = {r["id"]: r["ground_truth"] for r in rows if r.get("ground_truth")}
    transactions = rows_to_transactions(rows)
    categorizer = build_categorizer(settings, use_llm=use_llm)
    result = await categorizer.categorize(transactions)
    report = build_report(result, ground_truth)
    return JSONResponse(report)
