"""Config loading and deterministic pseudo-randomness helpers."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from statistics import NormalDist

import yaml

ROOT = Path(__file__).resolve().parents[1]
_ENV = re.compile(r"\$\{([A-Z0-9_]+)\}")


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config" / "tiers.yaml"
    text = path.read_text()
    # Unset env vars become empty strings; LiteLLM mode checks for them explicitly.
    text = _ENV.sub(lambda m: os.environ.get(m.group(1), ""), text)
    return yaml.safe_load(text)


def load_cases(path: str | Path | None = None) -> list[dict]:
    path = Path(path) if path else ROOT / "data" / "golden.jsonl"
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def hash01(*parts) -> float:
    """Deterministic uniform in [0, 1) keyed by the given parts."""
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def hash_normal(*parts) -> float:
    u = min(max(hash01(*parts), 1e-12), 1 - 1e-12)
    return NormalDist().inv_cdf(u)


Z95 = NormalDist().inv_cdf(0.95)  # 1.6448536...


def lognormal_params(median: float, p95: float) -> tuple[float, float]:
    """mu, sigma of a lognormal with the given median and 95th percentile."""
    if p95 <= median:
        raise ValueError("p95 must exceed median")
    return math.log(median), (math.log(p95) - math.log(median)) / Z95


def estimate_tokens(text: str) -> int:
    """~4 characters per token. Used only when a provider reports no usage."""
    return max(1, round(len(text) / 4))
