from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

import httpx

from src.domain.models import Taxonomy, TokenUsage

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Ты классификатор банковских расходов физлица. "
    "Ответ — только JSON-массив без markdown. "
    "Примеры: OOO/Marketing/Cloud/Design Studio → business; "
    "Online School → education; Fit Club/Gym → health; "
    "Entertainment Hall → entertainment; Transfer to Card → finance. "
    "other — только если точно не подходит ни одна категория."
)


class LLMCategorizer:
    """Batch LLM categorization via Ollama — used only for unresolved items."""

    def __init__(
        self,
        taxonomy: Taxonomy,
        host: str,
        model: str,
        batch_size: int = 15,
        temperature: float = 0.0,
        enable_cache: bool = True,
    ):
        self.taxonomy = taxonomy
        self.host = host.rstrip("/")
        self.model = model
        self.batch_size = batch_size
        self.temperature = temperature
        self.enable_cache = enable_cache
        self._cache: dict[str, dict[str, Any]] = {}

    def _cache_key(self, description: str) -> str:
        normalized = " ".join(description.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()

    def _build_prompt(self, items: list[tuple[str, str]]) -> str:
        cats = self.taxonomy.prompt_list()
        lines = [f'Категории:\n{cats}\n\nОперации (id|описание):']
        for tx_id, desc in items:
            safe_desc = desc.replace("|", "/").strip()[:120]
            lines.append(f"{tx_id}|{safe_desc}")
        lines.append(
            '\nВерни JSON: [{"id":"...","category":"<id>","confidence":0.0-1.0}]'
            '\nother — только в крайнем случае. '
            'OOO/LLC/Agency/Cloud/Studio → business; School → education.'
        )
        return "\n".join(lines)

    async def categorize_batch(
        self, items: list[tuple[str, str]]
    ) -> tuple[list[dict[str, Any]], TokenUsage]:
        usage = TokenUsage()
        results: list[dict[str, Any]] = []
        uncached: list[tuple[str, str]] = []

        for tx_id, desc in items:
            key = self._cache_key(desc)
            if self.enable_cache and key in self._cache:
                cached = dict(self._cache[key])
                cached["id"] = tx_id
                results.append(cached)
                usage.cached_hits += 1
            else:
                uncached.append((tx_id, desc))

        for i in range(0, len(uncached), self.batch_size):
            batch = uncached[i : i + self.batch_size]
            batch_results, batch_usage = await self._call_ollama(batch)
            usage.add(batch_usage)

            for item in batch_results:
                tx_id, desc = next(t for t in batch if t[0] == item["id"])
                key = self._cache_key(desc)
                if self.enable_cache:
                    self._cache[key] = {
                        "category": item["category"],
                        "confidence": item.get("confidence", 0.7),
                    }
                results.append(item)

        return results, usage

    async def _call_ollama(
        self, batch: list[tuple[str, str]]
    ) -> tuple[list[dict[str, Any]], TokenUsage]:
        usage = TokenUsage(llm_calls=1)
        prompt = self._build_prompt(batch)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature},
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(f"{self.host}/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            logger.warning("Ollama unavailable: %s", exc)
            return self._fallback(batch), usage

        message = data.get("message", {})
        content = message.get("content", "")
        eval_count = data.get("eval_count", 0)
        prompt_eval_count = data.get("prompt_eval_count", 0)
        usage.prompt_tokens = prompt_eval_count
        usage.completion_tokens = eval_count
        usage.total_tokens = prompt_eval_count + eval_count

        parsed = self._parse_response(content, batch)
        return parsed, usage

    def _parse_response(
        self, content: str, batch: list[tuple[str, str]]
    ) -> list[dict[str, Any]]:
        valid_ids = {cid for cid in self.taxonomy.category_ids()}
        try:
            data = json.loads(content)
            if isinstance(data, dict) and "items" in data:
                data = data["items"]
            if not isinstance(data, list):
                raise ValueError("Expected JSON array")
        except (json.JSONDecodeError, ValueError):
            return self._fallback(batch)

        results = []
        seen = set()
        for item in data:
            tx_id = str(item.get("id", ""))
            category = str(item.get("category", "other"))
            if category not in valid_ids:
                category = "other"
            confidence = float(item.get("confidence", 0.7))
            results.append({"id": tx_id, "category": category, "confidence": confidence})
            seen.add(tx_id)

        for tx_id, _ in batch:
            if tx_id not in seen:
                results.append({"id": tx_id, "category": "other", "confidence": 0.3})

        return results

    def _fallback(self, batch: list[tuple[str, str]]) -> list[dict[str, Any]]:
        return [{"id": tx_id, "category": "other", "confidence": 0.0} for tx_id, _ in batch]
