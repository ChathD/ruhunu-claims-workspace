"""Define the request, evidence, finding and decision types shared across the backend."""
from decimal import Decimal
from enum import Enum
from typing import Annotated, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Money = Annotated[Decimal, Field(ge=0, max_digits=14, decimal_places=2, allow_inf_nan=False)]
ChargeCategory = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,39}$")]
REQUIRED_POLICY_FIELDS = ("starts_on", "ends_before", "covered_losses", "registration", "limit", "deductible", "clause")
InsuranceType = Literal["comprehensive", "third_party"]
MotorLossType = Literal["repair", "theft", "third_party_property_damage", "third_party_injury"]
MOTOR_LOSSES = frozenset(get_args(MotorLossType))
THIRD_PARTY_LOSSES = frozenset({"third_party_property_damage", "third_party_injury"})


class AdditionalCost(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    amount: Money = Field(gt=0)
    category: ChargeCategory | None = None
    reason: str = Field(min_length=1, max_length=500)


class VehicleEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    make: str | None = Field(default=None, min_length=1, max_length=60)
    model: str | None = Field(default=None, min_length=1, max_length=80)
    year: int | None = Field(default=None, ge=1886, le=2100)
    chassis_number: str | None = Field(default=None, min_length=1, max_length=40)


class Route(str, Enum):
    AUTO = "auto-approve"
    REVIEW = "adjuster review"
    INFO = "request more information"
    FRAUD = "fraud escalation"


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    agent: str
    status: Literal["pass", "information", "review", "fraud"]
    summary: str
    evidence: list[str]
    details: dict = Field(default_factory=dict)


class GenerateRequest(BaseModel):
    count: int = Field(default=16, ge=1, le=160)


class ClaimPreviewRequest(BaseModel):
    policy_id: str = Field(min_length=1, max_length=64)
    count: int = Field(ge=1, le=10)


class AcknowledgeRequest(BaseModel):
    reviewer: str = Field(min_length=1, max_length=100)


class InformationRequest(BaseModel):
    document: str
    location: str | None = None
    reason: str


class Decision(BaseModel):
    route: Route
    justification: str
    reason_codes: list[str]
    gross_amount: Money
    payable_amount: Money | None
    finding_ids: list[str]
    context_version: int
    information_required: list[InformationRequest] = Field(default_factory=list)
    recommendation: dict | None = None
    override_reason: str | None = None


STAGES = ["intake", "fraud_initial", "coverage", "historical", "fraud_followup"]
RULE_VERSION = "provisional-motor-v1"
AUTHORITY = Decimal("250000.00")
