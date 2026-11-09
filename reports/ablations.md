# Ablations

> **SIMULATED RUN.** Model outputs come from the oracle-backed mock provider, whose accuracy is set by
> `mock_priors` in `config/tiers.yaml`. Every number below follows from those assumptions plus the real
> grading, routing, validation and statistics code. Re-run with `--mode litellm` to get measurements.


All rows use the routed cascade + cache policy and vary one factor. Quality columns are golden-set rates; cost, fully-loaded cost and latency are production-weighted. Each ablation was chosen on this same golden set, so treat the winner as a hypothesis to confirm on fresh data, not a tuned result.

## Router conservativeness (alpha)

| alpha | Pass@1 (judge) | Silent (judge) | Silent (human*) | Hand-offs | Cost / 1k | Fully-loaded / 1k | P90 | P95 |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 79.0% | 16.5% | 9.5% | 4.5% | $0.878 | $1,117 | 2897 ms | 4321 ms |
| 0.6 | 80.5% | 15.0% | 8.0% | 4.5% | $0.959 | $959 | 2934 ms | 4321 ms |
| 0.7 | 81.0% | 14.5% | 7.5% | 4.5% | $1.173 | $906 | 3229 ms | 5490 ms |
| 0.8 | 83.0% | 12.5% | 7.0% | 4.5% | $1.469 | $836 | 3615 ms | 5642 ms |
| 0.9 | 83.0% | 10.0% | 4.5% | 7.0% | $2.139 | $587 | 4985 ms | 6321 ms |

## Validator confidence threshold (all tiers)

| confidence_threshold | Pass@1 (judge) | Silent (judge) | Silent (human*) | Hand-offs | Cost / 1k | Fully-loaded / 1k | P90 | P95 |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 81.5% | 16.5% | 9.5% | 2.0% | $0.959 | $1,073 | 2495 ms | 3615 ms |
| 0.6 | 81.5% | 16.0% | 9.0% | 2.5% | $0.976 | $1,034 | 2667 ms | 3615 ms |
| 0.7 | 81.0% | 14.5% | 7.5% | 4.5% | $1.173 | $906 | 3229 ms | 5490 ms |
| 0.8 | 74.0% | 8.5% | 3.5% | 17.5% | $1.980 | $797 | 5823 ms | 7282 ms |

## Router features

| features | Pass@1 (judge) | Silent (judge) | Silent (human*) | Hand-offs | Cost / 1k | Fully-loaded / 1k | P90 | P95 |
|---|---|---|---|---|---|---|---|---|
| query+context | 81.0% | 14.5% | 7.5% | 4.5% | $1.173 | $906 | 3229 ms | 5490 ms |
| query only | 81.0% | 14.5% | 7.5% | 4.5% | $1.130 | $906 | 3229 ms | 5337 ms |

## Risk pre-filter

| risk_rules | Pass@1 (judge) | Silent (judge) | Silent (human*) | Hand-offs | Cost / 1k | Fully-loaded / 1k | P90 | P95 |
|---|---|---|---|---|---|---|---|---|
| on | 81.0% | 14.5% | 7.5% | 4.5% | $1.173 | $906 | 3229 ms | 5490 ms |
| off | 80.0% | 15.5% | 8.5% | 4.5% | $1.143 | $970 | 3143 ms | 5490 ms |

## Semantic-cache similarity threshold (offline TF-IDF embedder)

| Threshold | How-to hit rate | Hits on wrong FAQ | False hits on non-how-to (n=150) |
|---|---|---|---|
| 0.40 | 46.0% | 5 | 27 |
| 0.45 | 34.0% | 2 | 12 |
| 0.50 | 26.0% | 1 | 8 |
| 0.55 | 18.0% | 1 | 5 |
| 0.60 | 8.0% | 0 | 0 |
| 0.62 | 6.0% | 0 | 0 |
| 0.65 | 6.0% | 0 | 0 |
| 0.70 | 4.0% | 0 | 0 |
| 0.75 | 2.0% | 0 | 0 |

Below 0.60 the lexical embedder starts serving FAQ answers to diagnostic questions. At 0.62 it is safe on this set (0 of 150 false hits; the rule-of-three upper bound is about 2%) but recalls only 6% of how-to questions. The cache is not worth much until it runs on a real embedding model and real repeat traffic.
