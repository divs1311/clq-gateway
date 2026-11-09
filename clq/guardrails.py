"""Runtime validator. Uses ONLY information available in production (never the golden contract).

A response is delivered only if it raises no flags; otherwise the gateway escalates
to the next tier (cascade) or hands off to a human (last tier / non-escalating policy).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from pydantic import ValidationError

from .schemas import TOOL_SIGNATURES, TriageOutput

# Global security patterns, applied to the whole serialized output.
SECURITY_PATTERNS = [r"sk_live_[A-Za-z0-9]{8,}", r"(?i)delete_orders\s*\("]

# Root causes where a reply legitimately has no supporting KB chunk.
NO_CITATION_NEEDED = {"PROMPT_INJECTION", "DESTRUCTIVE_ACTION_REQUEST", "CROSS_TENANT_REQUEST", "AMBIGUOUS_NEEDS_INFO"}


@dataclass
class Validation:
    ok: bool
    flags: list[str] = field(default_factory=list)
    output: Optional[TriageOutput] = None


def validate(raw: str, case_input: dict, confidence_threshold: float,
             allowed_tools: Optional[list[str]] = None) -> Validation:
    flags: list[str] = []
    allowed = set(allowed_tools or TOOL_SIGNATURES)

    if any(re.search(p, raw) for p in SECURITY_PATTERNS):
        flags.append("FORBIDDEN_PATTERN")

    try:
        out = TriageOutput.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError, TypeError):
        flags.append("SCHEMA_INVALID")
        return Validation(False, flags, None)

    tc = out.tool_call
    if tc is not None:
        if tc.name not in allowed:
            flags.append("DISALLOWED_TOOL")
        elif sorted(tc.args) != sorted(TOOL_SIGNATURES[tc.name]):
            flags.append("BAD_TOOL_ARGS")
        if "merchant_id" in tc.args and tc.args["merchant_id"] != case_input["requester_merchant_id"]:
            flags.append("TENANT_VIOLATION")

    ctx_ids = {c["id"] for c in case_input["context_chunks"]}
    if any(c not in ctx_ids for c in out.cited_chunks):
        flags.append("UNKNOWN_CITATION")
    if not out.cited_chunks and out.root_cause_code.value not in NO_CITATION_NEEDED:
        flags.append("MISSING_CITATION")

    if out.confidence < confidence_threshold:
        flags.append("LOW_CONFIDENCE")

    return Validation(not flags, flags, out)
