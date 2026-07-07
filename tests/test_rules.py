from __future__ import annotations

import pytest

from src.domain.taxonomy import load_taxonomy
from src.engines.merchant_lookup import MerchantKnowledgeBase
from src.engines.rules import RuleEngine


@pytest.fixture
def taxonomy():
    return load_taxonomy()


@pytest.fixture
def merchant_kb(taxonomy):
    return MerchantKnowledgeBase(taxonomy)


@pytest.fixture
def rule_engine(taxonomy):
    return RuleEngine(taxonomy)


class TestMerchantKB:
    def test_tilda_is_business(self, merchant_kb):
        assert merchant_kb.match("TILDA.CC SUBSCRIPTION") == "business"

    def test_figma_is_business(self, merchant_kb):
        assert merchant_kb.match("FIGMA INC SAN FRANCISCO") == "business"

    def test_pyaterochka_is_groceries(self, merchant_kb):
        assert merchant_kb.match("PYATEROCHKA 1234") == "groceries"

    def test_unknown_returns_none(self, merchant_kb):
        assert merchant_kb.match("RANDOM PAYMENT XYZ") is None


class TestRuleEngine:
    def test_taxi_keyword(self, rule_engine):
        assert rule_engine.match("Payment TAXI SERVICE") == "transport"

    def test_no_match(self, rule_engine):
        assert rule_engine.match("XYZ UNKNOWN") is None
