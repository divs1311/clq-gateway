"""The dashboard's JS simulator must reproduce the Python model exactly."""
import json
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from clq.economics import compare, default_params
from clq.util import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.parametrize("tweak", [{}, {"recall": 0.9, "cache_on": False}, {"mix": {"easy": 0.5, "medium": 0.35, "hard": 0.15}}])
def test_js_matches_python(tweak):
    p = deepcopy(default_params(load_config()))
    p.update(tweak)
    p["mc_samples"] = 5000
    py = compare(p)
    js = json.loads(subprocess.run(
        ["node", "-e", f"const s=require({json.dumps(str(ROOT / 'dashboard' / 'sim.js'))});"
                       f"console.log(JSON.stringify(s.compare({json.dumps(p)})))"],
        capture_output=True, text=True, check=True).stdout)
    for pol in py:
        for k in ("pass_rate", "silent_error_rate", "handoff_rate", "model_cost_per_1k", "fully_loaded_per_1k"):
            assert js[pol][k] == pytest.approx(py[pol][k], abs=1e-9)
        for q in ("p50", "p90", "p95", "p99"):
            assert js[pol]["latency_ms"][q] == pytest.approx(py[pol]["latency_ms"][q], abs=1e-6)
