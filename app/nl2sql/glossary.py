"""Schema + business-glossary prompt builder for NL-to-SQL (data design §7.1).

The glossary (config/glossary.yaml) is operator-editable (PRD AC-6.8). It is
loaded once and injected into the LLM prompt so the model uses real column names
and the org's domain vocabulary instead of hallucinating.
"""
from __future__ import annotations

from functools import lru_cache

import yaml

from app.config import CONFIG_DIR

GLOSSARY_PATH = CONFIG_DIR / "glossary.yaml"

# Tables the NL-to-SQL surface may touch (mirrors validator whitelist, §7.2).
ALLOWED_TABLES = {"collections", "visits", "attorney_aging", "settlements", "lop"}


@lru_cache
def load_glossary() -> dict:
    with GLOSSARY_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def schema_prompt() -> str:
    """Compressed schema + glossary text injected into the LLM system prompt."""
    g = load_glossary()
    tables = g.get("tables", {})
    lines = ["Tables available:"]
    for name, cols in tables.items():
        if name in ALLOWED_TABLES:
            lines.append(f"- {name}({', '.join(cols)})")
    glossary = g.get("glossary", [])
    if glossary:
        lines.append("")
        lines.append("Business glossary:")
        for item in glossary:
            lines.append(f'- "{item["term"]}" = {item["meaning"]}')
    return "\n".join(lines)
