# CLQ Gateway

A cost, latency and quality (CLQ) routing gateway and eval harness for an LLM support copilot. The copilot
helps merchants on a fictional multichannel order-management platform (Omnistock) with integration problems:
failed order imports, webhook errors, unmatched SKUs, stock drift and how-to questions.

The project answers one product question with working code: **which routing policy should take production
traffic, and what has to be true before it does?**

```
request -> semantic cache -> risk rules -> complexity router -> small | mid | frontier -> validator
                                                                           ^                 | flagged
                                                                           +---- escalate ---+--> human
```

## What's in the repo

| Path | What it is |
|---|---|
| `data/golden.jsonl` | 200 synthetic cases: 120 routine, 50 multi-hop, 30 adversarial. Each has a deterministic contract (root cause, exact tool call, escalation, citations, forbidden patterns) and a 0/1/2 judge rubric |
| `clq/router.py` | TF-IDF + logistic-regression complexity router, a RouteLLM-style alpha threshold, and a deterministic risk pre-filter. Scored on out-of-fold probabilities |
| `clq/cache.py` | Semantic cache that only stores how-to answers, never merchant-specific diagnoses |
| `clq/guardrails.py` | Runtime validator: Pydantic schema, tool allow-list and signatures, tenant isolation, security regex, citations, confidence |
| `clq/cascade.py` | The gateway. Cost and latency accumulate across escalations |
| `clq/providers.py` | `LiteLLMProvider` for real models; `MockProvider` for offline runs |
| `clq/evaluate.py`, `clq/judge.py`, `clq/stats.py` | Harness, LLM judge with position-swap pairwise, Wilson CIs, exact McNemar, Cohen's kappa with bootstrap CI, production re-weighting, ablations |
| `clq/economics.py`, `dashboard/sim.js` | Unit-economics model (exact path enumeration + seeded Monte-Carlo latency). The JS port matches Python to 1e-9 (`tests/test_parity.py`) |
| `docs/PRD.md` | Engineering PRD: objective function, latency budgets, escalation rules, SLOs, kill criteria, rollout |
| `reports/` | Generated `summary.md`, `ablations.md`, `error_analysis.md`, raw results |
| `dashboard/` | Interactive console: simulator, eval results, trace explorer, ablations |

## Run it

```bash
pip install -e ".[dev]"
make all              # build data -> mock eval -> 32 tests -> dashboard/index.html
python -m clq.cli simulate --silent-cost 4 --recall 0.85
```

**With real models** (any LiteLLM-supported provider, including a local vLLM or Ollama endpoint for the small tier):

```bash
pip install -e ".[live]"
export CLQ_SMALL_MODEL=...  CLQ_MID_MODEL=...  CLQ_FRONTIER_MODEL=...  CLQ_JUDGE_MODEL=...
# plus the provider API keys LiteLLM expects
make eval-live
python -m clq.cli label --policy cascade+cache   # hand-grade replies
python -m clq.cli calibrate                      # judge kappa vs your labels
```

Then update `config/tiers.yaml` (prices, latency, validator rates, routing matrix) from the measured run so the
simulator and the PRD reflect real numbers.

## Read this before quoting any number

The default run uses `MockProvider`. It answers correctly with the probabilities in `config/tiers.yaml >
mock_priors` and fails in realistic ways: truncated JSON, wrong tool arguments, ungrounded claims, followed
injections, missed escalations. The grading, routing, validation, statistics and economics code is real, but
the model quality is assumed. The prices are illustrative. Every report and the dashboard carry a SIMULATED
banner for this reason.

## Findings (simulated v0)

1. **Token cost is the wrong objective.** Model spend is 0.6% of the fully-loaded cost of frontier-only
   ($3.33 of $527 per 1k tickets). Human hand-offs and wrong answers are the rest.
2. **The routed cascade is 3x cheaper per token and 4x faster at the median, and it matches frontier quality on
   the golden set (McNemar p = 0.81). It still loses on fully-loaded cost**, because confident cheap-tier
   mistakes get past the validator.
3. **The decision turns on two numbers.** The cascade wins if a wrong answer costs under **$4.40**, or if
   validator recall rises from 61% to **90%**. So the next investment is error detection, not the router.
4. **The LLM judge failed calibration** (kappa 0.65 at 94% raw agreement). Decisions use human labels instead.
5. **The classifier alone mis-routes attacks.** Without risk rules it sends 10 of 30 adversarial cases below the
   frontier tier.

The full reasoning, SLOs and rollout plan are in [`docs/PRD.md`](docs/PRD.md).

## Design choices worth defending in an interview

- **Fully-loaded cost per ticket instead of gross margin.** A support copilot is a cost-to-serve feature, so
  margin would need invented revenue.
- **A true cascade.** Escalations pay for every attempt. A static traffic split hides that cost.
- **Production re-weighting.** The golden set oversamples hard cases (60/25/15) on purpose. Cost and latency are
  re-weighted to the expected production mix (70/22/8).
- **The P95 comes from the latency distribution, not the mean.** If 15% of traffic takes 3 s, P95 cannot be 1 s.
- **The validator never sees ground truth.** It uses only what production has.
- **No leakage.** The router is scored out of fold. Thresholds tuned on the golden set are labelled as
  hypotheses for fresh data.
