# PRD: CLQ Gateway (cost, latency and quality routing for a merchant support copilot)

**Status:** v0 design validated in simulation. **Not approved for production traffic.**
**Scope:** Omnistock (a fictional multichannel order-management platform) and its in-app support copilot.
**Owner:** Product (AI platform). **Reviewers:** Support Ops, Security, Inference Infra, Finance.

> Every result in this document comes from the offline mock run plus the economics model. The mock's model
> accuracy is an assumption (`config/tiers.yaml > mock_priors`). The grading, routing, validation and statistics
> code is real. Section 10 lists what must be re-measured with real models before any launch decision.

---

## 1. Problem

Merchants ask the copilot about channel integrations: failed order imports, webhook errors, unmatched SKUs,
stock drift and how-to questions. Today every request goes to a frontier model. At about 400k tickets a month
this raises three questions:

1. **Latency.** The frontier tier has a median of about 3 s and a P90 of about 5.8 s, which is too slow for an
   in-app assistant.
2. **Cost.** The routing project was proposed to cut token spend.
3. **Silent regressions.** Prompt and model changes ship without a gate, so regressions go unnoticed.

## 2. The decision this PRD makes

The obvious goal is to minimise token spend. **That is the wrong objective.** At the expected volume, model cost
is under 1% of what the support flow really costs:

| Cost component (frontier-only, simulator defaults) | Per 1k tickets | Share |
|---|---|---|
| Model + infra | $3.33 | 0.6% |
| Human hand-offs (8.6% x $3.00) | $259 | 49% |
| Silent errors (2.2% x $12.00) | $265 | 50% |
| **Fully-loaded** | **$527** | 100% |

**Primary objective:** minimise **fully-loaded cost per 1k tickets** = model cost + hand-offs x human cost +
silent errors x silent-error cost. The minimisation is subject to the SLOs in section 5.

**Why not gross margin?** The copilot is a cost-to-serve feature, not a line of revenue, so gross margin would
need an invented revenue number. Cost per resolved ticket is what Finance and Support Ops already track.

## 3. Goals and non-goals

**Goals**

- A routing gateway with four layers: cache, complexity and risk router, three model tiers, and a validator that
  escalates. It must have a measured Pareto position on cost, latency and quality.
- A release gate that blocks prompt and model changes that regress quality or safety.
- An economics model that Finance can audit and that agrees with the eval.

**Non-goals (v0)**

- Retrieval quality. The context chunks are fixed per case, so chunking and retriever ablations wait for v1.
- Fine-tuning.
- Autonomous write actions. All tools are read-only diagnostics.

## 4. System design

```
request -> [semantic cache] -> [risk rules] -> [complexity router] -> tier t -> [validator] --ok--> deliver
               hit: deliver        flagged: t = frontier                          | flagged
                                                                                  v
                                                     cascade: t+1 (until frontier), else human hand-off
```

| Layer | Implementation | Latency budget | Notes |
|---|---|---|---|
| Semantic cache | Embedding + cosine (offline: TF-IDF; prod: embedding model + pgvector) | median 20 ms, P95 45 ms | Stores **only** `HOW_TO_QUESTION` answers. Merchant-specific diagnoses are never cached. |
| Risk rules | Deterministic regex and entity checks | < 1 ms | Injection markers in the query **or retrieved text**, credential or bulk-delete requests, other merchant IDs, low-information queries. All go to the frontier tier. |
| Complexity router | TF-IDF + multinomial logistic regression | 30 ms | Picks the cheapest tier `t` with `P(difficulty <= t) >= alpha`, where `alpha = 0.70`. |
| Small tier | Quantized ~8B open-weight model | median 250 ms, P95 600 ms | |
| Mid tier | Fast hosted model | median 900 ms, P95 2.0 s | |
| Frontier tier | Frontier reasoning model | median 3.0 s, P95 7.0 s | |
| Validator | Pydantic schema, tool allow-list and signature, tenant check, security regex, citation check, confidence threshold | 5 ms | Uses only runtime information, never the golden answers. |
| Eval harness | Deterministic contract + LLM judge (rubric 0/1/2) + human labels | offline | |

**Escalation rules (exact)**

- Any validator flag makes the answer *untrusted*. The flags are `SCHEMA_INVALID`, `DISALLOWED_TOOL`,
  `BAD_TOOL_ARGS`, `TENANT_VIOLATION`, `FORBIDDEN_PATTERN`, `UNKNOWN_CITATION`, `MISSING_CITATION` and
  `LOW_CONFIDENCE` (confidence < 0.70).
- In a cascade, an untrusted answer goes to the next tier. An untrusted frontier answer becomes a **validator
  hand-off** to a human, with the attempts attached.
- If the model returns `escalate_to_human = true` and passes validation, that is a **policy hand-off**. This is
  correct for injection, destructive and cross-tenant requests. Policy hand-offs are excluded from policy
  comparisons because every correct policy produces them.
- Cost and latency accumulate across attempts. An escalated request pays for every tier it touched.

## 5. SLOs and release gates

| Metric | Target | Measured how |
|---|---|---|
| Silent-error rate (delivered and wrong) | <= 3.0% | Deterministic contract + human labels (judge only once it passes its gate) |
| Validator hand-off rate | <= 12% | Gateway logs |
| End-to-end latency P90 / P99 | <= 2.5 s / <= 12 s | Gateway logs (production); simulator (planning) |
| Security: forbidden pattern or tenant violation delivered | 0 | Validator + red-team suite |
| Judge trust: Cohen's kappa vs human on faithfulness | point >= 0.80 **and** 95% CI low >= 0.70 | >= 300 labelled replies, bootstrap CI |
| Quality regression on a change | Exact McNemar p < 0.05 **and** Pass@1 drop > 2 pts blocks release | Paired on the golden set |

## 6. Evaluation method

- **Golden set:** 200 synthetic cases. 120 are routine (50 how-to, 70 single-cause diagnostics), 50 are multi-hop
  (two chunks plus exact tool arguments), and 30 are adversarial (8 direct injection, 8 indirect injection in
  retrieved text, 6 destructive requests, 4 cross-tenant requests, 4 ambiguous).
- **Contract per case:** expected root cause, exact tool call, escalation flag, required citations and forbidden
  patterns. These are checked deterministically.
- **Judge:** a 0/1/2 rubric for faithfulness and helpfulness. A low-precision scale reduces judge noise. Pairwise
  comparisons run in both orders, and only verdicts that survive the swap count.
- **Statistics:** Wilson 95% CIs for Pass@1, exact McNemar for paired policy comparisons, and Cohen's kappa with a
  bootstrap CI for judge calibration.
- **Production re-weighting:** the golden set oversamples hard cases (60/25/15). Cost and latency are re-weighted to
  the expected production mix of 70/22/8. **Reporting golden-set cost as production cost would understate the
  savings from routing.**
- **No leakage:** the router is scored on 5-fold out-of-fold probabilities.

## 7. Results (simulated v0)

Simulator, 400k tickets a month, default assumptions. The eval agrees within 3 points on pass rate and within 8%
on model cost; see `reports/summary.md`.

| Policy | Pass | Silent | Hand-off | Model $/1k | Fully-loaded $/1k | Monthly fully-loaded | P50 | P90 | P95 | P99 |
|---|---|---|---|---|---|---|---|---|---|---|
| Frontier only (today) | 89.2% | 2.2% | 8.6% | $3.33 | $527 | $210,955 | 2.98 s | 5.80 s | 7.00 s | 9.96 s |
| Mid only | 83.7% | 4.5% | 11.8% | $1.02 | $889 | $355,771 | 0.91 s | 1.70 s | 2.02 s | 2.80 s |
| Static router | 83.0% | 4.8% | 12.2% | $0.83 | $940 | $376,028 | 0.59 s | 2.01 s | 3.02 s | 5.70 s |
| Cheap-first cascade | 84.9% | 13.7% | 1.4% | $0.50 | $1,685 | $673,888 | 0.32 s | 1.45 s | 2.62 s | 6.20 s |
| **Routed cascade** | **90.9%** | **6.1%** | **2.9%** | **$1.10** | **$825** | **$330,169** | **0.70 s** | **3.01 s** | **4.40 s** | **7.34 s** |

### Findings

1. **The routed cascade is 3x cheaper per token, 4x faster at the median, and has the highest pass rate. It still
   loses on fully-loaded cost** ($825 vs $527 per 1k tickets). The cheaper tiers make *confident* mistakes that the
   validator misses, which nearly triples the silent-error rate (6.1% vs 2.2%).
2. **The bottleneck is the validator, not the router.** The validator catches 61% of wrong answers overall: 57% on
   the small tier, where most cheap-tier traffic lands. The cascade breaks even at **90% recall**. At 90% recall
   both policies sit at about $375-381 per 1k, and the cascade keeps its latency and hand-off advantages.
3. **The decision depends on what a wrong answer costs.** The cascade wins whenever a silent error costs less than
   **$4.40**. Support Ops must sign off on the $12 assumption. It is the most important number in this PRD.
4. **The classifier alone is unsafe.** Out of fold at alpha = 0.70, it routed 10 of the 30 adversarial cases below
   the frontier tier (4 to the small tier). These were short ambiguous messages and destructive or cross-tenant
   requests that read like simple questions. Risk rules fixed that on this set (30/30 caught, 0/170 false
   positives), but they were written knowing the attack types, so that score proves nothing about production.
5. **The LLM judge is not trustworthy yet.** Kappa is 0.65 (CI 0.49-0.78) despite 94% raw agreement, so the gate
   fails. Judge-graded silent errors for the cascade (14.5%) are double the human-graded figure (7.5%). Ship
   decisions in this PRD therefore use the deterministic contract plus human labels.
6. **No configuration meets the P90 latency SLO** (2.5 s). The cascade's P90 is 3.0 s, driven by the roughly 11% of
   requests that escalate.
7. **The cache barely matters with lexical embeddings.** At the only safe threshold (0.62) it recalls 6% of how-to
   questions. Below 0.60 it starts answering diagnostic questions with FAQ text. Its value cannot be estimated until
   it runs on a real embedding model with real repeat traffic.

## 8. Recommendation and rollout

Do **not** route production traffic with v0. Build in this order:

| Phase | Exit criterion |
|---|---|
| 0. Real eval: run `--mode litellm` on three real models; label 300+ replies; re-fit the simulator from measured rates | Simulator within 3 pts of eval on pass rate |
| 1. Detection: claim-level grounding check plus tool-argument cross-check on small- and mid-tier answers (see `reports/error_analysis.md`) | Validator recall on cheap-tier errors >= 0.90 with false-flag rate <= 8%, on a held-out set |
| 2. Fix the judge: stricter rubric, length-blind instructions, reference-guided grading | Kappa point >= 0.80 and CI low >= 0.70 |
| 3. Latency: stream the reply after header fields validate; add an "investigating" message for escalations; consider a speculative mid-tier call | Simulated P90 <= 2.5 s |
| 4. Shadow mode: the cascade runs beside frontier-only on 100% of traffic; 1% of delivered replies are audited by humans | Fully-loaded cost lower with 95% confidence over 2 weeks |
| 5. Canary: 5%, then 25%, then 100% | Kill criteria below never fire |

### Kill criteria (automatic rollback to frontier-only)

- Any delivered answer with a forbidden pattern or a tenant violation.
- Audited silent-error rate (rolling, n >= 400) with a 95% Wilson upper bound above 3.0%.
- Validator hand-off rate above 12% for 2 consecutive hours.
- P90 latency above 2.5 s for 30 minutes, or P99 above 12 s for 30 minutes.
- Golden-set gate on any prompt or model change: McNemar p < 0.05 with a Pass@1 drop greater than 2 points, or
  any new failure on the adversarial stratum.

## 9. Risks

| Risk | Mitigation |
|---|---|
| Synthetic golden set is easier than real tickets, and template phrasing inflates router accuracy | Phase 0 adds 300 anonymised real tickets with labels before any launch decision |
| Risk rules overfit to known attacks | Separate red-team set written by someone who has not seen the rules; track rule recall as an SLO |
| Cheap tiers drift after a provider update | Golden-set gate on every model version; weekly calibration of validator recall |
| Mock nesting assumption (if a cheap tier is right, stronger tiers are too) | Real eval measures joint tier outcomes; the simulator accepts measured rates |

## 10. Assumptions register (all must be replaced by measurements)

| Assumption | Value | Source |
|---|---|---|
| Model accuracy by tier and difficulty | `mock_priors.accuracy` | Assumption; replace with the real eval |
| Token prices | $0.10/$0.40, $1/$4, $3/$15 per 1M input/output tokens | Illustrative only; use current provider price sheets |
| Tokens per request | 570 in / 110 out | Measured on the synthetic golden set; real contexts are likely longer. At 1,500/300 the frontier model cost rises to $8.87 per 1k, but fully-loaded cost moves by only about 1% |
| Tier latency | Log-normal with the medians and P95s in section 4 | Assumption; replace with gateway logs |
| Human cost per hand-off | $3.00 | Assumption; Support Ops to confirm |
| Silent-error cost | $12.00 | Assumption; this decides the policy (break-even $4.40) |
| Production difficulty mix | 70/22/8 | Assumption; label a random sample of 500 tickets |
| Volume | 400k tickets a month | Assumption |
