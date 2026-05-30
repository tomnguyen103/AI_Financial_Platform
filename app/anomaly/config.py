"""Per-entity threshold config (YAML), AC-4.7 / data design §5.

Merges entity-specific overrides over defaults so a volatile facility can have
wider bands without code changes.
"""
from __future__ import annotations

import copy
from functools import lru_cache

import yaml

from app.config import CONFIG_DIR

_THRESHOLDS_PATH = CONFIG_DIR / "thresholds.yaml"


@lru_cache
def _load() -> dict:
    return yaml.safe_load(_THRESHOLDS_PATH.read_text(encoding="utf-8"))


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def thresholds_for(entity_id: str) -> dict:
    cfg = _load()
    base = cfg.get("defaults", {})
    override = cfg.get("overrides", {}).get(entity_id, {})
    return _deep_merge(base, override)
