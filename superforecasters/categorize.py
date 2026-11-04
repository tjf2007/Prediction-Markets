"""Map markets to Finance/Economics sub-topics using event tags + title keywords."""
from __future__ import annotations

import re

TOPICS: dict[str, list[str]] = {
    "fed_rates": [r"\bfed\b", r"fomc", r"rate (cut|hike)", r"interest rate", r"powell", r"fed funds", r"\bbps\b",
                  r"basis points", r"\becb\b", r"\bboe\b", r"bank of (japan|england|canada)", r"\bboj\b", r"rba"],
    "inflation": [r"\bcpi\b", r"\bpce\b", r"\bppi\b", r"inflation", r"core prices"],
    "labor": [r"payrolls?", r"\bnfp\b", r"jobs report", r"unemployment", r"jobless", r"jolts", r"nonfarm"],
    "growth": [r"\bgdp\b", r"recession", r"\bism\b", r"\bpmi\b", r"retail sales", r"consumer (sentiment|confidence)"],
    "rates_bonds": [r"treasury", r"10[- ]?year", r"\byields?\b", r"2[- ]?year", r"\bbond"],
    "equities": [r"s&p", r"\bspx\b", r"\bspy\b", r"nasdaq", r"\bdow\b", r"stock", r"\bipo\b", r"earnings", r"market cap",
                 r"\bnvda\b|nvidia", r"apple|microsoft|tesla|amazon|meta|alphabet|google"],
    "fx": [r"\busd\b", r"dollar index", r"\bdxy\b", r"\beur\b|euro", r"\byen\b", r"yuan|renminbi", r"exchange rate"],
    "commodities": [r"\boil\b", r"crude", r"\bwti\b", r"brent", r"gold", r"silver", r"natural gas", r"gas prices", r"copper"],
    "fiscal_trade": [r"tariff", r"trade deal", r"debt ceiling", r"shutdown", r"deficit", r"tax", r"stimulus", r"budget"],
    "housing": [r"housing", r"home (prices|sales)", r"mortgage", r"case[- ]shiller"],
}
FIN_ECON_TAGS = {"economy", "economics", "finance", "fed", "fed rates", "inflation", "stocks", "business", "markets",
                 "macro", "interest rates", "jobs", "gdp", "tariffs", "commodities", "earnings", "ipos", "financials",
                 "economic policy", "trade war", "recession", "equities", "forex"}
_COMPILED = {k: re.compile("|".join(v), re.I) for k, v in TOPICS.items()}


def topics_for(title: str | None, tags: list[str] | None = None) -> list[str]:
    text = (title or "") + " " + " ".join(tags or [])
    return [k for k, rx in _COMPILED.items() if rx.search(text)]


def is_fin_econ(title: str | None, tags: list[str] | None = None) -> bool:
    if tags and FIN_ECON_TAGS.intersection(t.lower() for t in tags):
        return True
    return bool(topics_for(title, tags))


def primary_topic(title: str | None, tags: list[str] | None = None) -> str:
    t = topics_for(title, tags)
    return t[0] if t else "other"
