# Eval summary

> **SIMULATED RUN.** Model outputs come from the oracle-backed mock provider, whose accuracy is set by
> `mock_priors` in `config/tiers.yaml`. Every number below follows from those assumptions plus the real
> grading, routing, validation and statistics code. Re-run with `--mode litellm` to get measurements.


Golden set: 200 cases (120 routine / 50 multi-hop / 30 adversarial). Production re-weighting uses {'easy': 0.7, 'medium': 0.22, 'hard': 0.08} (weights per stratum: easy x1.167, medium x0.880, hard x0.533).

## Golden-set results (unweighted, n = 200 per policy)

| Policy | Pass@1 (judge) | 95% CI | Pass@1 (human*) | Silent errors (judge) | Silent errors (human*) | Validator hand-offs | Escalated | McNemar p vs frontier |
|---|---|---|---|---|---|---|---|---|
| Small only | 55.0% | 48.1%-61.7% | 59.5% | 20.5% | 16.0% | 24.5% | 0.0% | 0.000 |
| Mid only | 73.5% | 67.0%-79.1% | 78.5% | 9.0% | 4.0% | 17.5% | 0.0% | 0.006 |
| Frontier only (baseline) | 82.0% | 76.1%-86.7% | 87.5% | 9.0% | 3.5% | 9.0% | 0.0% | - |
| Static router | 75.0% | 68.6%-80.5% | 81.0% | 12.5% | 6.5% | 12.5% | 0.0% | 0.016 |
| Cheap-first cascade | 72.0% | 65.4%-77.8% | 78.0% | 26.5% | 20.5% | 1.5% | 24.5% | 0.002 |
| Routed cascade | 81.0% | 75.0%-85.8% | 88.0% | 14.5% | 7.5% | 4.5% | 10.0% | 0.815 |
| Routed cascade + cache | 81.0% | 75.0%-85.8% | 88.0% | 14.5% | 7.5% | 4.5% | 10.0% | 0.815 |

*human = simulated human labels in mock mode.

## Production-weighted economics and latency

| Policy | Model cost / 1k | Fully-loaded / 1k | P50 | P90 | P95 | P99 | Final tier small/mid/frontier |
|---|---|---|---|---|---|---|---|
| Small only | $0.100 | $2,175 | 245 ms | 481 ms | 598 ms | 832 ms | 100% / 0% / 0% |
| Mid only | $1.015 | $765 | 913 ms | 1724 ms | 2086 ms | 2747 ms | 0% / 100% / 0% |
| Frontier only (baseline) | $3.327 | $549 | 2951 ms | 6174 ms | 7227 ms | 10077 ms | 0% / 0% / 100% |
| Static router | $0.824 | $1,054 | 592 ms | 2056 ms | 3124 ms | 6437 ms | 47% / 44% / 9% |
| Cheap-first cascade | $0.512 | $1,986 | 278 ms | 1694 ms | 3084 ms | 8236 ms | 80% / 14% / 6% |
| Routed cascade | $1.154 | $906 | 674 ms | 3206 ms | 5452 ms | 7840 ms | 44% / 39% / 18% |
| Routed cascade + cache | $1.173 | $906 | 691 ms | 3229 ms | 5490 ms | 7877 ms | 43% / 37% / 18% |

Latency percentiles from 200 weighted samples are noisy (P99 is roughly the 2nd-slowest request). Use the simulator for production-scale percentiles.

Fully-loaded cost = model cost + validator hand-offs x human cost + silent errors x silent-error cost ($3.00 and $12.00). Policy-required escalations (prompt injection, destructive or cross-tenant requests) are excluded because a correct system sends them to a human under every policy.

## Router

Out-of-fold accuracy (5-fold CV): query+context 87.0%, query only 88.5%.

Row-normalised confusion (query+context; rows = true easy/medium/hard, cols = predicted):

| true \ pred | easy | medium | hard |
|---|---|---|---|
| easy | 99.2% | 0.8% | 0.0% |
| medium | 20.0% | 80.0% | 0.0% |
| hard | 50.0% | 0.0% | 50.0% |

Without risk rules, the classifier at alpha=0.7 routes the 30 adversarial cases as small 4 / mid 6 / frontier 20: short ambiguous messages and destructive or cross-tenant requests look like simple questions. Deterministic risk rules flag 30/30 adversarial cases with 0/170 false positives and force them to the frontier tier. Caveat: Rules were written with knowledge of the golden-set attack categories; validate on a held-out red-team set.

## Validator (measured against mock ground truth)

| Tier | Recall on wrong answers | False-flag rate on correct | Top flags |
|---|---|---|---|
| small | 56.8% (n=74) | 5.6% (n=126) | LOW_CONFIDENCE 34, SCHEMA_INVALID 9, TENANT_VIOLATION 6, DISALLOWED_TOOL 6 |
| mid | 75.8% (n=33) | 6.0% (n=167) | LOW_CONFIDENCE 25, TENANT_VIOLATION 5, MISSING_CITATION 5, DISALLOWED_TOOL 5 |
| frontier | 53.3% (n=15) | 5.4% (n=185) | LOW_CONFIDENCE 15, MISSING_CITATION 2, DISALLOWED_TOOL 1, SCHEMA_INVALID 1 |

## Judge calibration

> SIMULATED judge vs SIMULATED human labels; replace with real labels before quoting.

Cohen's kappa (judge vs human, faithfulness pass/fail, n=312): **0.650** (95% bootstrap CI 0.487-0.785); raw agreement 94.2%.
Gate (kappa >= 0.8 and CI low >= 0.7): **FAIL**.
Judge false-fail rate on grounded replies: 5.8%. False-pass on ungrounded replies: short 0.0% (n=11), long 11.1% (n=9).

High raw agreement with a low kappa is the expected signature of an imbalanced label set: most replies are grounded, so agreeing on them is easy. Because the gate fails, ship decisions use deterministic checks plus human labels, not the judge.

## Pairwise comparison with position swap

Routed cascade + cache vs Frontier only (baseline) on 58 cases where both delivered different replies. Swap-consistent verdicts: 33 (56.9%). Among them: candidate wins 0, baseline wins 3, ties 30. Inconsistent verdicts are discarded because they reflect position bias, not quality.

## Simulator vs eval agreement (production-weighted)

The simulator uses pooled validator rates and the measured routing matrix, so small gaps are expected (per-tier validator recall differs from the pooled value, and the hard stratum has only 30 cases).

| Policy | Pass sim / eval | Silent sim / eval | Hand-off sim / eval | Model cost/1k sim / eval |
|---|---|---|---|---|
| Small only | 69.7% / 67.1% | 10.2% / 13.2% | 20.1% / 19.7% | $0.101 / $0.100 |
| Mid only | 83.6% / 82.5% | 4.5% / 2.7% | 11.9% / 14.9% | $1.010 / $1.015 |
| Frontier only (baseline) | 89.0% / 89.4% | 2.2% / 2.5% | 8.7% / 8.1% | $3.360 / $3.327 |
| Static router | 82.7% / 82.5% | 4.8% / 5.9% | 12.4% / 11.6% | $0.806 / $0.824 |
| Cheap-first cascade | 84.8% / 82.7% | 13.8% / 16.3% | 1.5% / 1.0% | $0.472 / $0.512 |
| Routed cascade + cache | 90.9% / 90.1% | 6.1% / 6.8% | 2.9% / 3.1% | $1.097 / $1.173 |

## Decision-relevant break-evens (simulator, default assumptions)

- The routed cascade beats frontier-only on fully-loaded cost only if a silent error costs less than **$4.40**.
- At the assumed $12.00 per silent error, the cascade needs validator recall of at least **90%** to break even (currently 61%).
