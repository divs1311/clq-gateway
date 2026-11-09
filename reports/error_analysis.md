# Error analysis

> **SIMULATED RUN.** Model outputs come from the oracle-backed mock provider, whose accuracy is set by
> `mock_priors` in `config/tiers.yaml`. Every number below follows from those assumptions plus the real
> grading, routing, validation and statistics code. Re-run with `--mode litellm` to get measurements.


Silent errors = answers delivered to the merchant that are wrong. Because the judge failed its calibration gate, the breakdown uses the deterministic contract plus human labels; the judge-graded count is shown for comparison and its excess is judge false-fails. Validator flags = problems caught before delivery (they trigger escalation or a hand-off).

| Policy | Silent errors (human) | Breakdown | Judge-graded silent errors (of which judge false-fails) | Validator flags |
|---|---|---|---|---|
| Small only | 32 | UNGROUNDED_REPLY 15 (47%), WRONG_TOOL_ARGS 7 (22%), ESCALATION_ERROR 6 (19%), WRONG_ROOT_CAUSE 4 (12%) | 41 (10) | LOW_CONFIDENCE 34, SCHEMA_INVALID 9, TENANT_VIOLATION 6, DISALLOWED_TOOL 6, MISSING_CITATION 5, FORBIDDEN_PATTERN 2 |
| Mid only | 8 | UNGROUNDED_REPLY 4 (50%), ESCALATION_ERROR 3 (38%), WRONG_ROOT_CAUSE 1 (12%) | 18 (10) | LOW_CONFIDENCE 25, TENANT_VIOLATION 5, MISSING_CITATION 5, DISALLOWED_TOOL 5, SCHEMA_INVALID 2, FORBIDDEN_PATTERN 1 |
| Frontier only (baseline) | 7 | ESCALATION_ERROR 4 (57%), WRONG_TOOL_ARGS 2 (29%), UNGROUNDED_REPLY 1 (14%) | 18 (11) | LOW_CONFIDENCE 15, MISSING_CITATION 2, DISALLOWED_TOOL 1, SCHEMA_INVALID 1, TENANT_VIOLATION 1 |
| Static router | 13 | UNGROUNDED_REPLY 6 (46%), ESCALATION_ERROR 4 (31%), WRONG_ROOT_CAUSE 2 (15%), WRONG_TOOL_ARGS 1 (8%) | 25 (12) | LOW_CONFIDENCE 17, SCHEMA_INVALID 4, TENANT_VIOLATION 3, MISSING_CITATION 2, DISALLOWED_TOOL 1 |
| Cheap-first cascade | 41 | UNGROUNDED_REPLY 18 (44%), ESCALATION_ERROR 10 (24%), WRONG_TOOL_ARGS 8 (20%), WRONG_ROOT_CAUSE 5 (12%) | 53 (13) | LOW_CONFIDENCE 44, TENANT_VIOLATION 11, SCHEMA_INVALID 10, DISALLOWED_TOOL 10, MISSING_CITATION 9, FORBIDDEN_PATTERN 2 |
| Routed cascade | 15 | UNGROUNDED_REPLY 7 (47%), ESCALATION_ERROR 4 (27%), WRONG_TOOL_ARGS 2 (13%), WRONG_ROOT_CAUSE 2 (13%) | 29 (14) | LOW_CONFIDENCE 23, SCHEMA_INVALID 4, TENANT_VIOLATION 4, MISSING_CITATION 2, DISALLOWED_TOOL 1 |
| Routed cascade + cache | 15 | UNGROUNDED_REPLY 7 (47%), ESCALATION_ERROR 4 (27%), WRONG_TOOL_ARGS 2 (13%), WRONG_ROOT_CAUSE 2 (13%) | 29 (14) | LOW_CONFIDENCE 23, SCHEMA_INVALID 4, TENANT_VIOLATION 4, MISSING_CITATION 2, DISALLOWED_TOOL 1 |

## What to fix, by failure class

| Failure class | Why the validator misses it | Fix | How to verify |
|---|---|---|---|
| UNGROUNDED_REPLY | Schema-valid and confident; only meaning is wrong | Claim-level grounding check (NLI or a small judge) on cheap-tier replies before delivery | Recall of the grounding check on held-out ungrounded replies >= 0.85 |
| WRONG_TOOL_ARGS | Arguments are well-formed, just wrong (channel, SKU, date) | Cross-check args against entities extracted from the query and the order record | Arg exact-match on multi-hop >= 95% |
| ESCALATION_ERROR | Escalation flag is a single boolean with no evidence | Deterministic policy rules decide escalation; the model only drafts the reply | 0 missed escalations on the red-team set |
| WRONG_ROOT_CAUSE | Enum is valid | Constrain the root-cause enum to the codes in the retrieved chunks | Root-cause accuracy per stratum |
| SCHEMA_CORRUPTION / SECURITY | Caught by Pydantic and regex guards | Keep; add constrained decoding where supported | Already 100% caught in this run |
