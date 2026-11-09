"""Typed contracts for golden cases and model outputs."""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class RootCause(str, Enum):
    HOW_TO_QUESTION = "HOW_TO_QUESTION"
    AUTH_TOKEN_EXPIRED = "AUTH_TOKEN_EXPIRED"
    WEBHOOK_SCHEMA_MISMATCH = "WEBHOOK_SCHEMA_MISMATCH"
    RATE_LIMITED = "RATE_LIMITED"
    SKU_MAPPING_MISSING = "SKU_MAPPING_MISSING"
    INVENTORY_SYNC_LAG = "INVENTORY_SYNC_LAG"
    INVENTORY_SYNC_STALLED = "INVENTORY_SYNC_STALLED"
    CATEGORY_ATTRIBUTE_MISSING = "CATEGORY_ATTRIBUTE_MISSING"
    CURRENCY_MISMATCH = "CURRENCY_MISMATCH"
    PROMPT_INJECTION = "PROMPT_INJECTION"
    DESTRUCTIVE_ACTION_REQUEST = "DESTRUCTIVE_ACTION_REQUEST"
    CROSS_TENANT_REQUEST = "CROSS_TENANT_REQUEST"
    AMBIGUOUS_NEEDS_INFO = "AMBIGUOUS_NEEDS_INFO"


TOOL_SIGNATURES: dict[str, list[str]] = {
    "check_token_validity": ["merchant_id", "channel"],
    "query_webhook_logs": ["merchant_id", "channel", "since"],
    "get_order_sync_status": ["merchant_id", "channel"],
    "inspect_listing_mapping": ["merchant_id", "channel", "sku"],
    "get_inventory_sync_report": ["merchant_id", "warehouse_id"],
}


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    args: dict[str, str]


class TriageOutput(BaseModel):
    """The JSON contract every tier must return."""
    model_config = ConfigDict(extra="forbid")
    root_cause_code: RootCause
    tool_call: Optional[ToolCall] = None
    escalate_to_human: bool
    customer_reply: str = Field(min_length=1)
    cited_chunks: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


class ContextChunk(BaseModel):
    id: str
    text: str


class CaseInput(BaseModel):
    user_query: str
    requester_merchant_id: str
    merchant_tier: Literal["standard", "enterprise_sla"]
    context_chunks: list[ContextChunk]


class Contract(BaseModel):
    expected_root_cause: RootCause
    expected_tool_call: Optional[ToolCall]
    must_escalate: bool
    allowed_tools: list[str]
    required_citations: list[str]
    forbidden_patterns_regex: list[str]


class GoldenCase(BaseModel):
    case_id: str
    stratum: Literal["routine", "multi_hop", "adversarial"]
    difficulty: Literal["easy", "medium", "hard"]
    ground_truth_complexity_tier: int = Field(ge=1, le=3)
    scenario: str
    input: CaseInput
    contract: Contract
    reference_reply: str
    judge_rubric: dict


STRATUM_TO_DIFFICULTY = {"routine": "easy", "multi_hop": "medium", "adversarial": "hard"}
DIFFICULTY_TO_TIER = {"easy": 1, "medium": 2, "hard": 3}
