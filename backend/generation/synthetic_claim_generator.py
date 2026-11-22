"""Plan and draft fictional claims against a selected policy, then validate and save approved previews."""
import base64
import json
import re
import sqlite3
import uuid
from contextlib import closing
from copy import deepcopy
from datetime import date
from decimal import Decimal
from difflib import SequenceMatcher
from itertools import islice
from types import SimpleNamespace
from typing import Annotated, Literal

from pydantic import Field, computed_field, model_validator
from pydantic_core import PydanticCustomError
from sqlalchemy import func, select

from backend.agents.historical_comparer import cost_document_amount
from backend.agents.intake_specialist import cost_breakdown, cost_evidence
from backend.storage.sqlite_store import Assessment, Claim, InvestigationRevision, Person, Policy, SyntheticPolicyHolder, digest, now
from backend.generation.claude_generation_client import BATCH_SIZE, GeneratedData, GenerationError, GenerationValidationError, generation_client, local_date, request_batch
from backend.generation.synthetic_policy_generator import SupportingDocumentType, list_people_policies
from backend.assurance.execution_manifest import execution_manifest
from backend.claim_schemas import AUTHORITY, AdditionalCost, ChargeCategory, Money, MotorLossType, Route, VehicleEvidence
from backend.evidence.required_claim_evidence import (
    SUPPORTING_DOCUMENT_LABELS, coverage_document_checklist, policy_applies_to_loss, supporting_document_rules,
)
from backend.model_settings import settings
from backend.workflow.specialist_context import generated_policy_context
from backend.generation.policy_copy_checks import copied_fields, draft_text, plan_text, policy_phrases, stored_cost_text, words
from backend.evidence.policy_wording_index import generated_wording_for_policy
from backend.generation.claim_scenario_rules import (
    COMPLETE_SCENARIOS, CONTRADICTORY_NARRATIVE, INCOMPLETE_NARRATIVE, ESTIMATE_NARRATIVE_MISMATCH, MISSING_SUPPORTING_DOCUMENT,
    NARRATIVE_SCENARIOS, OMISSIONS, REPAIR_SUBLIMIT_EXCEEDED, REUSED_COST_DOCUMENT, available_scenarios,
    baseline_limit, classify_scenario, is_complete_reference_claim, omit_required_evidence, repair_estimate_sublimits,
)
from backend.evidence.synthetic_cost_history import prior_cost_documents

POLICY_CLAIM_LIMIT = 10


GENERATION_TOOL = "submit_synthetic_claims"
PLANNING_TOOL = "plan_synthetic_claims"
MIN_COMPARABLES = 5
MAX_PLAN_COMPARABLES = 12
DIRECT_POLICY_EXCEPTIONS = {"uncovered_loss", "before_policy_start", "after_policy_end", "above_policy_limit"}
UNDAMAGED_DESCRIPTION = re.compile(
    r"\b(?:undamaged|untouched|unaffected|unmarked|unscathed|intact|no (?:signs? of )?damage|"
    r"no (?:scratches or dents|dents or scratches)|(?:was|were) (?:fine|not damaged)|(?:wasn|weren) t damaged)\b"
)

GENERATION_EXAMPLES = json.loads("""{
  "cases": [
    {
      "scenario": "Rear bumper repair after a parking collision",
      "conditions": "The policy covers the bumper repair and the separately listed tow without a receipt. The repair estimate totals LKR 87500.00.",
      "expected_route": "adjuster review",
      "claim": {
        "person_id": "PER-EX-REPAIR",
        "policy_id": "POL-EX-REPAIR",
        "vehicle": {"make": "Toyota", "model": "Vitz", "year": 2018, "chassis_number": "SYN-CH-REPAIR"},
        "incident_key": "example-repair",
        "loss_type": "repair",
        "loss_date": "2025-09-03",
        "reported_date": "2025-09-05",
        "gross_amount": "92500.00",
        "additional_costs": [{"category": "towing", "amount": "5000.00", "reason": "Towing to the repair garage because the bumper dragged on the road."}],
        "narrative": "A parking collision damaged the rear bumper. The garage quoted replacement parts and fitting.",
        "documents": [{
          "type": "repair_estimate",
          "name": "Bumper repair estimate.json",
          "garage_name": "Example Motor Works",
          "garage_address": "18 Example Road, Galle",
          "registration": "SYN-EX-REPAIR",
          "date": "2025-09-04",
          "total": "87500.00",
          "line_items": [
            {"description": "Rear bumper and brackets", "amount": "62000.00"},
            {"description": "Fitting and paint labour", "amount": "25500.00"}
          ]
        }]
      }
    },
    {
      "scenario": "Stolen motorcycle with police report and valuation",
      "conditions": "The theft policy requires a police report. The report is attached and the valuation supports the LKR 185000.00 claimed.",
      "expected_route": "adjuster review",
      "claim": {
        "person_id": "PER-EX-THEFT",
        "policy_id": "POL-EX-THEFT",
        "vehicle": {"make": "Honda", "model": "Dio", "year": 2021, "chassis_number": "SYN-CH-THEFT"},
        "incident_key": "example-theft",
        "loss_type": "theft",
        "loss_date": "2025-08-18",
        "reported_date": "2025-08-21",
        "gross_amount": "185000.00",
        "additional_costs": [],
        "narrative": "The motorcycle was stolen overnight from a locked courtyard. The owner submitted a valuation after reporting the theft.",
        "documents": [{
          "type": "vehicle_valuation",
          "name": "Motorcycle valuation.json",
          "registration": "SYN-EX-THEFT",
          "date": "2025-08-20",
          "total": "185000.00",
          "line_items": [
            {"description": "Pre-loss motorcycle market value", "amount": "185000.00"}
          ]
        }, {
          "type": "police_report",
          "name": "Police report.json",
          "registration": "SYN-EX-THEFT",
          "date": "2025-08-18",
          "station": "Fictional Lake Road Police Station",
          "summary": "The owner reported the motorcycle missing from the locked courtyard. The report records the vehicle registration and the overnight loss."
        }]
      }
    }
  ]
}""")["cases"]


class PolicySelectionError(ValueError):
    """Stored people and policies are unavailable or changed since preview."""


class LineItem(GeneratedData):
    description: str = Field(min_length=1, max_length=180)
    amount: Money


class RepairSublimitTest(GeneratedData):
    category: Literal["labour", "painting"]
    line_number: int = Field(ge=1, le=10, description="One-based position of the repair estimate charge exceeding its cap.")


class CostDocument(GeneratedData):
    name: str = Field(min_length=1, max_length=150)
    registration: str | None = Field(min_length=1, max_length=40)
    date: date
    total: Money | None = Field(default=None, description="Supply an explicit total when testing an arithmetic discrepancy.")
    line_items: list[LineItem] = Field(max_length=10)


class RepairEstimate(CostDocument):
    type: Literal["repair_estimate"]
    garage_name: str = Field(min_length=1, max_length=150)
    garage_address: str = Field(min_length=1, max_length=300)


class Document(CostDocument):
    type: Literal["vehicle_valuation", "third_party_demand"]
    chassis_number: str | None = Field(default=None, min_length=1, max_length=40)


class PoliceReport(GeneratedData):
    type: Literal["police_report"]
    name: str = Field(min_length=1, max_length=150)
    registration: str = Field(min_length=1, max_length=40)
    date: date
    station: str = Field(min_length=1, max_length=120)
    summary: str = Field(min_length=10, max_length=450)


class AdditionalCostReceipt(GeneratedData):
    type: Literal["additional_cost_receipt"]
    category: ChargeCategory
    name: str = Field(min_length=1, max_length=150)
    registration: str = Field(min_length=1, max_length=40)
    date: date
    total: Money = Field(gt=0)
    description: str = Field(min_length=10, max_length=450)


class SupportingDocument(GeneratedData):
    type: SupportingDocumentType
    name: str = Field(min_length=1, max_length=150)
    registration: str = Field(min_length=1, max_length=40)
    date: date
    author: str = Field(min_length=1, max_length=120)
    summary: str = Field(min_length=10, max_length=450)


IncidentNarrative = Annotated[str, Field(min_length=10, description="Concise factual account of the incident and resulting loss.")]


class ClaimDraft(GeneratedData):
    person_id: str = Field(min_length=1, max_length=64)
    policy_id: str = Field(min_length=1, max_length=64)
    vehicle: VehicleEvidence | None = None
    incident_key: str = Field(min_length=1, max_length=160, description="Temporary label for grouping repeated incidents within this response")
    loss_type: MotorLossType
    loss_date: date | None
    reported_date: date
    gross_amount: Money | None = Field(default=None, description="Supply an explicit gross amount when testing a discrepancy in the amount claimed.")
    additional_costs: list[AdditionalCost] = Field(max_length=10)
    narrative: IncidentNarrative
    documents: list[RepairEstimate | Document | PoliceReport | AdditionalCostReceipt | SupportingDocument] = Field(max_length=5)
    reused_cost_document_id: str | None = Field(default=None, min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_dates(self):
        if self.reported_date > local_date() or (self.loss_date and self.loss_date > self.reported_date):
            raise ValueError("Claim dates must be chronological and at or before today")
        if any(document.date > local_date() for document in self.documents):
            raise ValueError("Document dates must be at or before today")
        return self


class GeneratedCase(GeneratedData):
    scenario: str = Field(min_length=1, description="Brief title for the scenario.")
    conditions: str = Field(min_length=1, description="Brief description of the submitted evidence and any deliberate gaps or inconsistencies.")
    expected_route: Route
    claim: ClaimDraft


class GeneratedCases(GeneratedData):
    cases: list[GeneratedCase] = Field(min_length=1, max_length=BATCH_SIZE)


class EstimateDiscrepancy(GeneratedData):
    narrative_excerpt: str = Field(min_length=10, description="Exact sentence stating a vehicle part was undamaged.")
    unaffected_part: str = Field(min_length=3, description="Part named in the excerpt and the conflicting estimate description.")
    line_number: int = Field(ge=1, le=10)
    estimate_description: str = Field(min_length=1, max_length=180, description="Replacement estimate line quoting work on the undamaged part.")


class NarrativeVariation(GeneratedData):
    reference_sentence: str = Field(min_length=10, max_length=500)
    conflicting_sentence: str | None = Field(default=None, min_length=10, max_length=500)

    def apply(self, incident):
        if self.conflicting_sentence:
            return f"{incident} {self.conflicting_sentence}"
        before, _, after = incident.partition(self.reference_sentence)
        return " ".join(part for part in (before.rstrip(), after.lstrip()) if part)


class ClaimPlan(GeneratedData):
    scenario: str = Field(min_length=1, description="Brief title for the scenario.")
    incident: IncidentNarrative
    loss_type: MotorLossType
    loss_date: date
    cost_items: list[LineItem] = Field(min_length=1, max_length=10)
    additional_costs: list[AdditionalCost] = Field(max_length=10)
    policy_basis: str = Field(min_length=1, description="Brief explanation of the applicable policy terms and any deliberate exception.")
    required_evidence: list[str] = Field(min_length=1, max_length=8)
    comparison_ids: list[str] = Field(description="Distinct IDs of the supplied comparisons for this loss type.")
    scenario_type: str = Field(min_length=1, max_length=60)
    repair_sublimit_test: RepairSublimitTest | None = None
    estimate_discrepancy: EstimateDiscrepancy | None = None
    narrative_variation: NarrativeVariation | None = None
    include_optional_police_report: bool = False
    optional_supporting_documents: list[SupportingDocumentType] = Field(default_factory=list, max_length=2)
    omitted_supporting_document: SupportingDocumentType | None = None

    @model_validator(mode="before")
    @classmethod
    def discard_supplied_gross_amount(cls, data):
        if isinstance(data, dict) and "gross_amount" in data:
            return {key: value for key, value in data.items() if key != "gross_amount"}
        return data

    @property
    def cost_document_total(self) -> Decimal:
        return sum((item.amount for item in self.cost_items), Decimal("0.00"))

    @computed_field
    @property
    def gross_amount(self) -> Decimal:
        return self.cost_document_total + sum((item.amount for item in self.additional_costs), Decimal("0.00"))

    @model_validator(mode="after")
    def validate_repair_sublimit_test(self):
        test = self.repair_sublimit_test
        if self.scenario_type != REPAIR_SUBLIMIT_EXCEEDED:
            if test is not None:
                raise PydanticCustomError("invalid_repair_sublimit", "Use repair_sublimit_test only for repair_sublimit_exceeded")
            return self
        if not test or self.loss_type != "repair" or len(self.cost_items) < 2 or self.additional_costs:
            raise PydanticCustomError("invalid_repair_sublimit",
                                      "Use a repair loss with at least two estimate lines, the capped line reference and no separate charges")
        if test.line_number > len(self.cost_items):
            raise PydanticCustomError("invalid_repair_sublimit", "Select an existing estimate line, counting from 1")
        return self

    @model_validator(mode="after")
    def validate_narrative_variation(self):
        variation = self.narrative_variation
        error_type = "invalid_narrative_variation"
        if self.scenario_type not in NARRATIVE_SCENARIOS:
            if variation or self.include_optional_police_report:
                raise PydanticCustomError(error_type, "Use narrative variation options only for a narrative scenario")
            return self
        if not variation or self.incident.count(variation.reference_sentence) != 1:
            raise PydanticCustomError(error_type, "Give one exact incident sentence for the narrative variation")
        conflict = variation.conflicting_sentence
        if self.scenario_type == CONTRADICTORY_NARRATIVE:
            if not conflict or self.incident.find(conflict) >= 0 or words(conflict) == words(variation.reference_sentence):
                raise PydanticCustomError(error_type, "Supply a second, distinct sentence contradicting the same incident fact")
        elif conflict or len(variation.apply(self.incident)) < 10:
            raise PydanticCustomError(error_type, "Omit one fact and leave a readable account of the loss")
        return self

    @model_validator(mode="after")
    def validate_estimate_discrepancy(self):
        error_type = "invalid_estimate_discrepancy"
        discrepancy = self.estimate_discrepancy
        if self.scenario_type != ESTIMATE_NARRATIVE_MISMATCH:
            if discrepancy is not None:
                raise PydanticCustomError(error_type, "Use estimate_discrepancy only for estimate_narrative_mismatch")
            return self
        if discrepancy is None or self.loss_type != "repair":
            raise PydanticCustomError(error_type, "Use a repair claim and supply estimate_discrepancy")
        if discrepancy.line_number > len(self.cost_items):
            raise PydanticCustomError(error_type, "Set line_number to an existing line in cost_items, counting from 1")
        excerpt = " ".join(words(discrepancy.narrative_excerpt))
        if not excerpt or f" {excerpt} " not in f" {' '.join(words(self.incident))} ":
            raise PydanticCustomError(error_type, "Copy narrative_excerpt from the incident description")
        part = f" {' '.join(words(discrepancy.unaffected_part))} "
        if part not in f" {excerpt} " or not UNDAMAGED_DESCRIPTION.search(excerpt):
            raise PydanticCustomError(error_type, "Quote the named part being explicitly described as undamaged")
        if part not in f" {' '.join(words(discrepancy.estimate_description))} ":
            raise PydanticCustomError(error_type, "Name the same undamaged part in the replacement estimate description")
        original = self.cost_items[discrepancy.line_number - 1].description
        if part in f" {' '.join(words(original))} ":
            raise PydanticCustomError(error_type, "Choose a baseline estimate line for the damaged area")
        return self


class ClaimPlans(GeneratedData):
    cases: list[ClaimPlan] = Field(min_length=1, max_length=BATCH_SIZE)


def similar_incident(left, right):
    left, right = " ".join(words(left)), " ".join(words(right))
    return left == right or (min(len(left), len(right)) >= 60
                             and SequenceMatcher(None, left, right, autojunk=False).ratio() >= 0.88)


def exceeded_repair_sublimit(policy, test, items):
    term = repair_estimate_sublimits(policy).get(test.category)
    if not term or test.line_number > len(items):
        return None
    basis, limit = term
    item = items[test.line_number - 1]
    description = set(words(item.description))
    category_words = {"labour", "labor"} if test.category == "labour" else {"paint", "painting", "respray", "repainting"}
    if description.isdisjoint(category_words) or item.amount <= limit:
        return None
    return f"{item.description} is estimated at LKR {item.amount:,.2f}. The policy's {basis.replace('_', ' ')} {test.category} limit is LKR {limit:,.2f}, leaving LKR {item.amount - limit:,.2f} above that limit."


def request_language_checked_batch(client, model, prompt, tool, schema, count, text_fields, excerpts, *,
                                   plan_amount_range=None, discarded_incidents=(), policy=None, require_sublimit=False):
    for attempt in range(2):
        try:
            records = request_batch(client, model, prompt, tool, schema, count)
        except GenerationValidationError as error:
            if not attempt and tool == PLANNING_TOOL and (
                    error.issue_type == "invalid_repair_sublimit" or "repair_sublimit_test" in error.location):
                prompt += f"\nCorrect the capped repair estimate plan. {error} Return the complete batch again."
                continue
            if not attempt and tool == PLANNING_TOOL and (error.issue_type == "invalid_narrative_variation"
                    or "narrative_variation" in error.location):
                prompt += (f"\nCorrect the narrative variation. {error} Return the complete batch again. "
                           "Quote the exact reference sentence and make any contradictory sentence disagree on the same fact.")
                continue
            if (attempt or tool != PLANNING_TOOL or (error.issue_type != "invalid_estimate_discrepancy"
                    and "estimate_discrepancy" not in error.location)):
                raise
            prompt += (
                f"\nCorrect the estimate discrepancy plan. {error} "
                "Return the complete batch again. Copy the exact same part name from the narrative into unaffected_part and the replacement line, "
                "including its side and singular or plural form. "
                "For this retry you may choose a different unused scenario if you cannot plan a clear contradiction."
            )
            continue
        copied = sorted({f"case {index} {field}" for index, record in enumerate(records, 1)
                         for field in copied_fields(text_fields(record), excerpts)})
        amount_issues = []
        if plan_amount_range is not None:
            deductible, maximum, policy_limit = plan_amount_range
            for index, plan in enumerate(records, 1):
                if any(item.amount <= 0 for item in plan.cost_items):
                    amount_issues.append(f"Case {index} has a nonpositive estimate or valuation line.")
                if plan.scenario_type not in DIRECT_POLICY_EXCEPTIONS and not deductible < plan.gross_amount <= maximum:
                    amount_issues.append(f"Case {index} has an itemised total of LKR {plan.gross_amount:.2f}.")
                if plan.repair_sublimit_test and not exceeded_repair_sublimit(policy, plan.repair_sublimit_test, plan.cost_items):
                    amount_issues.append(f"Case {index} must select an itemised labour or painting line priced above its applicable cap.")
            if require_sublimit and not any(plan.repair_sublimit_test for plan in records):
                amount_issues.append(f"Include one {REPAIR_SUBLIMIT_EXCEEDED} scenario in the batch.")
        repeated = [
            index for index, plan in enumerate(records, 1)
            if any(plan.loss_type == prior["loss_type"] and similar_incident(plan.incident, prior["incident"])
                   for prior in discarded_incidents)
        ]
        if not copied and not amount_issues and not repeated:
            return records
        if repeated:
            if attempt:
                raise GenerationError("Claude repeated a discarded claim incident. Generate a new preview.")
            prompt += (
                f"\nCases {', '.join(map(str, repeated))} repeated discarded incidents. Return the full batch. "
                "Use a different event with a new setting and materially different circumstances and evidence. "
                "Changing wording, dates or identifiers alone is insufficient."
            )
        if amount_issues:
            requirement = (f"Use positive charges. A covered reference total must exceed LKR {deductible:.2f} and be at most LKR {maximum:.2f}. "
                           f"Direct policy exceptions may use a different total, including an amount above LKR {policy_limit:.2f} to test the limit.")
            if attempt:
                raise GenerationError(f"The plan did not pass the cost checks. {' '.join(amount_issues)} {requirement}")
            prompt += f"\nCorrect the itemised amounts. {' '.join(amount_issues)} {requirement} Return the complete batch in the same order."
        if copied:
            prompt += (
                f"\nSome submitted text copied policy wording in {', '.join(copied)}. "
                "Return the complete batch again in the same order. Rewrite those fields in fresh, natural language "
                "for the same events, evidence and scenario types. Keep amounts unless an itemised correction is requested."
            )
    raise GenerationError("The generated claim or document text repeated policy wording after a rewrite. Try generating a new preview.")


def request_plans(client, model, count, entry, remaining, comparisons, reusable_documents, today, excerpts=(), *,
                  recent_incidents=(), discarded_incidents=()):
    policy = entry["policy"]
    deductible, maximum = Decimal(policy["deductible"]), baseline_limit(policy)
    policy_limit = Decimal(policy["limit"])
    caps = {category: {"basis": basis, "limit": str(limit)}
            for category, (basis, limit) in repair_estimate_sublimits(policy).items()}
    require_sublimit = count >= 3 and REPAIR_SUBLIMIT_EXCEEDED in remaining
    sublimit_instruction = ""
    if REPAIR_SUBLIMIT_EXCEEDED in remaining:
        sublimit_instruction = (
            f"Available repair estimate caps: {json.dumps(caps)}. "
            + (f"Include one {REPAIR_SUBLIMIT_EXCEEDED} case in this batch. " if require_sublimit else "")
            + f"For {REPAIR_SUBLIMIT_EXCEEDED}, plan a covered repair with complete evidence, no separate additional costs, and at least two estimate lines. "
            "For one named panel, clearly itemise labour or painting as its own line. Price that line strictly above the selected cap. "
            "Keep the other work and the combined estimate within policy terms and the reference gross maximum. "
            "For example, for a painting cap of LKR 8000 per panel, plan an affected bumper paint line at LKR 9500 and an eligible bumper repair line at LKR 36000. "
            "Set repair_sublimit_test to the eligible category and the one-based line_number. Omit that field for other scenarios. "
        )
    prompt = (
        f"Plan exactly {count} distinct synthetic motor claims for this saved policy. Today is {today}. "
        "Choose one unused scenario_type for each case. You may choose either a policy-aligned submission or a deliberate exception. "
        "Invent a plausible own-vehicle repair, theft or third-party liability incident and give it a short title in scenario. For aligned cases and missing-evidence variants, "
        "plan the complete covered reference. For a deliberate policy exception such as an uncovered loss type, a date outside cover or "
        "a claim above the policy limit, you may plan the actual exception directly. "
        "Use the insured vehicle and holder and consistent dates except where the selected scenario tests one of those facts. The end date is exclusive. "
        "In policy_basis, explain the applicable term and any planned exception. Cite the relevant clauses and "
        "use short excerpts where needed. Put itemised costs and document requirements in their dedicated fields. "
        "Plan the claim form, matching itemised repair estimate, valuation or third-party demand and dates. "
        "A repair estimate needs the garage name and street address, registration and itemised proposed repairs. Keep VIN or chassis off it. "
        "For a policy-aligned scenario, choose a permitted vehicle use and an incident covered for that use. "
        "Describe the purpose of the trip or how the vehicle was being used in the claimant's own words. "
        "For repair, choose a cause from the relevant repair benefit and work expressly covered for that cause. Vary non-collision incidents when supported. "
        "Plan any mandatory police report and the reasons, limits and receipts for eligible additional costs if claimed. "
        "For a liability loss, use a claimant other than the insured for the damaged property or injury. Plan an itemised third_party_demand with "
        "documented third-party costs and evidence establishing the insured driver's responsibility. Use the insured vehicle registration to link the incident. "
        "Respect repair, labour, paint, towing and other charge exclusions. For the covered reference incident, use flood damage only when "
        "the policy expressly covers it, and plan work that fits the listed repair scope. "
        f"For a covered reference, choose positive itemised amounts that add up to more than LKR {deductible:.2f} and at most LKR {maximum:.2f}. "
        f"For a directly planned above_policy_limit scenario, the itemised total should instead exceed LKR {policy_limit:.2f}. "
        "In incident, describe what happened and the resulting damage or theft in a few factual sentences. "
        "Write in the claimant's own words. Vary length, how events are ordered, what the claimant personally observed, and how they discovered the damage or theft. "
        "Plan a different event for each case, with a distinct incident location, vehicle activity and sequence of events. "
        "Vary the cause, damage or liability facts where the policy and test scenario allow. "
        "Do not recreate a recent submitted or discarded incident by swapping the date, registration, or names. "
        "For incomplete_narrative, use a complete reference in incident and copy one important, non-repeated sentence to narrative_variation.reference_sentence "
        "for the server to remove. Choose a concrete fact such as where the vehicle was parked or how the damage occurred. Leave a readable account without it. "
        "For contradictory_narrative, quote one factual sentence from incident in reference_sentence and give a separate conflicting_sentence "
        "that changes the same fact. For example, the reference says the car was stationary at impact, and the conflicting sentence says it was moving. "
        "Plan attachments that corroborate the reference account. For a narrative scenario, set include_optional_police_report to true only if it is plausible "
        "that an optional police report establishes a relevant detail. Other scenarios omit narrative_variation and include_optional_police_report. "
        "Use practical workshop descriptions and incident-specific reasons for costs. Describe observations without copying policy terms, "
        "clause IDs or prior claim narratives. Any policy quotations belong only in policy_basis. "
        "Keep policy analysis, document requirements and cost breakdowns in their dedicated fields. "
        "Supply cost_items and additional_costs. The server calculates the claim total from those amounts. "
        "For covered reference claims, use similar staff-approved synthetic records where available. "
        "Identify only supplied records for the same loss type with comparable damage/work or valuation, "
        "and plan a cost document total within their observed cost range. Prefer at least five similar records "
        f"when available. Copy up to {MAX_PLAN_COMPARABLES} distinct IDs exactly as supplied into comparison_ids. "
        "Use [] if no suitable records exist. "
        "Do not invent history or copy identifying details or documents from those records. "
        "Mix policy-aligned cases, omissions and deliberate policy exceptions when the count and unused scenarios allow. For an omission choose a genuinely "
        "required item and include it in required_evidence so the server can remove it from the complete draft later. "
        "Include narrative variations in multi-claim batches when those types are available. "
        "Use the policy document checklist for the planned loss type and list the critical documents in required_evidence. "
        "In optional_supporting_documents list the optional incident_report or witness_statement types that should be attached, "
        "or use [] to omit them. For missing_required_supporting_document choose one applicable required document in "
        "omitted_supporting_document, and otherwise leave that field empty. "
        "Plan at most five attachments in a complete reference, counting the cost document, any police report, supporting evidence and receipts. "
        "For missing_loss_date, leave the exact date out of the narrative and attachment descriptions so the structured "
        "loss date is the only explicit statement of when the loss occurred. "
        "Other unused variants remain available for targeted coverage, financial and fraud testing. "
        + sublimit_instruction +
        "When estimate_narrative_mismatch is available and you plan at least three cases, include one such repair claim. "
        "For that variant, state in the claimant's own words which area was hit and name another part that was explicitly undamaged, "
        "such as saying it had no damage, was untouched or was intact. "
        "Keep cost_items aligned with the damaged area. In estimate_discrepancy, quote the sentence about the undamaged part, "
        "copy the exact part name into unaffected_part, and use that same phrase in the replacement estimate_description. "
        "Set line_number to the one-based position of a different item from the original estimate. For example, use 'left front door' "
        "in 'The left front door was untouched' and 'Replace left front door trim'. "
        "The server replaces just that estimate description. For other variants omit estimate_discrepancy. "
        "For example, if collision cover includes rear bumper repair, a complete reference could plan a dated repair claim "
        "with the insured chassis on the form, LKR 62,000 in replacement parts and LKR 25,500 in fitting and paint, and a matching repair estimate from a named garage with an address. "
        "If police evidence is required, include it in the plan even when choosing missing_required_police_report. "
        "Do not draft the submission, attachments, or outcome yet. Return one plan per case in generation order.\n"
        f"Unused scenario types: {json.dumps(remaining)}\n"
        f"Types that keep the complete draft: {', '.join(sorted(COMPLETE_SCENARIOS))}\n"
        f"Types that remove one required item: {json.dumps(OMISSIONS)}\n"
        f"Saved policy and holder ID: {json.dumps({**policy, 'person_id': entry['person']['person_id']})}\n"
        f"Policy document checklist by loss type: {json.dumps({loss: coverage_document_checklist(policy, loss) for loss in policy['covered_losses']})}\n"
        f"Recent submitted or discarded synthetic incidents to avoid repeating: {json.dumps([*recent_incidents, *discarded_incidents])}\n"
        f"Staff-approved synthetic comparisons: {json.dumps(list(comparisons.values()))}\n"
        f"Documents eligible for the dedicated reuse test: {json.dumps(reusable_documents)}"
    )
    plans = request_language_checked_batch(client, model, prompt, PLANNING_TOOL, ClaimPlans, count, plan_text, excerpts,
                                          plan_amount_range=(deductible, maximum, policy_limit), discarded_incidents=discarded_incidents,
                                          policy=policy, require_sublimit=require_sublimit)
    seen = set()
    for plan in plans:
        if plan.scenario_type not in remaining or plan.scenario_type in seen:
            raise GenerationError("Claude repeated a scenario type or selected one unavailable for this policy.")
        seen.add(plan.scenario_type)
        planned_date = plan.loss_date.isoformat()
        date_within_policy = policy["starts_on"] <= planned_date < policy["ends_before"]
        planned_date_exception = (plan.scenario_type == "before_policy_start" and planned_date < policy["starts_on"]
                                  or plan.scenario_type == "after_policy_end" and planned_date >= policy["ends_before"])
        if (plan.loss_date > today or not (date_within_policy or planned_date_exception)
                or (plan.loss_type not in policy["covered_losses"] and plan.scenario_type != "uncovered_loss")):
            raise GenerationError("The planned loss type or date does not match the selected scenario and policy.")
        plan.comparison_ids = [
            identifier for identifier in dict.fromkeys(plan.comparison_ids)
            if identifier in comparisons and comparisons[identifier]["loss_type"] == plan.loss_type
        ][:MAX_PLAN_COMPARABLES]
        report_required = plan.loss_type in (policy.get("police_report_required_for") or [])
        if (plan.scenario_type in {"missing_required_police_report", "required_police_report_attached"} and not report_required
                or plan.scenario_type in {"ordinary", "optional_police_report_attached"} and report_required):
            raise GenerationError("Claude planned a police report scenario that does not match the policy for this loss.")
        rules = {rule["document_type"]: rule["requirement"] for rule in supporting_document_rules(policy, plan.loss_type)}
        if (len(set(plan.optional_supporting_documents)) != len(plan.optional_supporting_documents)
                or any(rules.get(kind) != "optional" for kind in plan.optional_supporting_documents)):
            raise GenerationError("Claude selected a supporting document that the policy does not list as optional for this loss.")
        if plan.scenario_type == MISSING_SUPPORTING_DOCUMENT:
            if rules.get(plan.omitted_supporting_document) != "required":
                raise GenerationError("Choose a supporting document that the policy requires for the planned loss.")
        elif plan.omitted_supporting_document is not None:
            raise GenerationError("A required supporting document can be omitted only in its omission scenario.")
    return plans


def drafting_policy_facts(entry):
    policy = entry["policy"]
    facts = {key: policy[key] for key in ("policy_id", "insurance_type", "registration", "vehicle", "vehicle_type", "permitted_use", "starts_on", "ends_before",
             "covered_losses", "limit", "deductible", "police_report_required_for") if key in policy}
    facts["person_id"] = entry["person"]["person_id"]
    if repair := policy.get("repair_terms"):
        facts["repair_terms"] = {key: repair[key] for key in ("labour", "painting") if key in repair}
        charges = repair.get("additional_charges") or {}
        facts["repair_terms"]["eligible_charges"] = [
            {key: charge[key] for key in ("category", "limit_per_incident", "receipt_required")}
            for charge in charges.get("eligible_charges", [])
        ]
        facts["repair_terms"]["excluded_charge_categories"] = [charge["category"] for charge in charges.get("exclusions", [])]
    if theft := policy.get("theft_terms"):
        facts["theft_basis"] = theft["basis"]
    return facts


def document_draft_plan(plan, policy):
    result = []
    for rule in coverage_document_checklist(policy, plan.loss_type):
        kind = rule["document_type"]
        category = rule.get("category")
        if category and not any(cost.category == category for cost in plan.additional_costs):
            continue
        included = (rule["requirement"] == "required" or kind in plan.optional_supporting_documents
                    or (kind == "police_report" and (plan.scenario_type == "optional_police_report_attached" or plan.include_optional_police_report)))
        if kind == plan.omitted_supporting_document or (
                kind == "police_report" and plan.scenario_type == "missing_required_police_report") or (
                kind in {"repair_estimate", "vehicle_valuation", "third_party_demand"} and plan.scenario_type == "missing_cost_document"):
            included = False
        result.append({"document_type": kind, "category": category, "requirement": rule["requirement"], "included": included})
    return result


def request_cases(client, model, count, batch_id, offset, catalogue, *, plans, scenario_options=None, reusable_documents=None, excerpts=()):
    choices = [drafting_policy_facts(entry) for entry in catalogue]
    document_selections = [document_draft_plan(plan, catalogue[0]["policy"]) for plan in plans]
    prompt = (
        f"Generate exactly {count} invented Sri Lankan motor claim cases. Today is {local_date().isoformat()}. "
        f"Batch {batch_id}, starting case number {offset + 1}. Write one draft for each approved plan, in exactly the same order. "
        "Draft the selected scenario, whether it is policy-aligned or deliberately outside the policy. "
        "Keep the vehicle, loss and evidence consistent with that scenario. "
        "Use natural claimant language in the narrative. Write cost document items, report summaries, receipts and charge reasons "
        "as the claimant, repairer or reporting officer would. Vary expression and detail across cases. "
        "Keep contract clauses, copied policy passages and interpretations out of every part of the submission. "
        "For a policy-aligned, narrative-variation or estimate_narrative_mismatch scenario, write the complete reference claim with the planned loss date, type, "
        "amount and required evidence. Copy incident into claim.narrative, cost_items into the cost document's line_items, "
        "and additional_costs into claim.additional_costs. The server calculates the document total and gross amount from those verified items. "
        "For cost_document_arithmetic_mismatch supply the deliberately inconsistent document total. "
        "For unexplained_additional_cost or partial_amount supply the deliberately different gross_amount. "
        "Use the plan's scenario as the case title. "
        "For a missing-evidence scenario, keep the same reference facts and costs and omit only the designated field or document. "
        "You may also include that evidence for the server to remove. Describe the omission in conditions. "
        "For estimate_narrative_mismatch, leave the planned estimate descriptions unchanged and let the server apply the discrepancy. "
        "For repair_sublimit_exceeded, copy the approved repair estimate line items and their amounts exactly, even when one line exceeds a policy cap. "
        "The estimate shows what the garage quoted. Leave the cap and coverage interpretation out of the submission. "
        "For incomplete_narrative or contradictory_narrative, copy the approved incident exactly. The server changes only the narrative afterwards. "
        "Write supporting documents that corroborate the approved incident where they have firsthand evidence. "
        "For other dedicated coverage, amount discrepancy, or fraud tests, apply the specified variation. "
        f"Approved plans: {json.dumps([plan.model_dump(mode='json', exclude={'policy_basis', 'comparison_ids', 'estimate_discrepancy', 'narrative_variation', 'repair_sublimit_test'}) for plan in plans])}. "
        f"Policy document choices for each final claim, in the same order: {json.dumps(document_selections)}. "
        "Keep document choices appropriate to the submitted loss in targeted coverage variants. "
        f"Omission scenarios handled by the server: {json.dumps(OMISSIONS)}. "
        "For each claim, copy a policy_id and its associated person_id from the saved choices below. "
        "The policy ID is the only policy field in the submission. Keep policy terms in the saved policy record. "
        "Use loss_date <= reported_date <= today. In a complete draft, date attachments between the loss and report dates. "
        f"Planned scenario types and their requirements: {json.dumps(scenario_options or {})}. "
        f"Prior synthetic cost documents available for the reused_cost_document scenario: {json.dumps(reusable_documents or [])}. "
        "For that scenario, select one of those IDs as reused_cost_document_id. The server attaches the original document from a different vehicle. "
        "Use its loss type and total for an in-period claim. Supply any applicable police report in documents and leave the cost document out. "
        "Expect fraud escalation for that scenario. Omit reused_cost_document_id for all other scenarios. "
        "Use the policy vehicle registration for supporting documents unless the scenario deliberately tests inconsistent evidence. "
        "The server copies the vehicle registration into the claim. Do not generate people or policy data. "
        "Supply the vehicle make, model, year and chassis number in claim.vehicle from the selected saved vehicle for ordinary claims. "
        "If a scenario intentionally tests missing or inconsistent vehicle identity, provide only the evidence the claimant would report. "
        "Use the vehicle registration on an ordinary repair estimate, and include the issuing garage's fictional name and complete street address. "
        "There is no chassis_number or VIN field on the repair estimate. Keep the registration on valuations and other documents. "
        "Use the policy's structured labour and painting limits except for an approved deliberate sublimit exception. "
        "Name the panel separately for each panel-specific labour or paint charge. "
        "Use the planned work and structured limits to choose realistic charges. Where a scenario tests eligibility, consider excluded repair "
        "work, unlisted fittings, or charges beyond the limit. "
        "Use the vehicle valuation for the base theft loss. Itemise any separately claimed fittings as additional_costs. "
        "For theft with listed-fittings cover, identify each fitting by name. A valuation-only policy excludes separately claimed fittings. "
        "Keep loss narratives consistent unless the scenario explicitly requires a conflict. "
        "Use itemised repair estimates for repair and itemised vehicle valuations for theft. "
        "For third_party_property_damage or third_party_injury use an itemised third_party_demand. The name and descriptions identify the third party's "
        "losses in their own words, such as replacing their damaged fence panel or reimbursing a documented medical expense. The registration links the insured vehicle to the incident. "
        "For an in-period covered loss, a police report is mandatory only when its loss type is in the selected policy's police_report_required_for list. "
        "Include an optional police report only when requested in the document choices. "
        "A police_report uses an invented station and concise incident summary. Use the vehicle's registration and a report date on or after a readable loss date. "
        "A police report is incident evidence and carries no financial amounts. "
        "For witness_statement and incident_report use type, name, registration, date, author and summary. "
        "Use a fictional observer or incident recorder and describe the evidence in their own words. "
        "Attach only the selected optional supporting evidence and the required documents for this scenario. These records carry no financial amounts. "
        "Keep the repair estimate total tied to its line items. For charges outside the cost document, list each "
        "additional_costs entry with amount and an incident-specific reason. Use the exact policy category when it describes the service, "
        "including a category listed in the policy exclusions when testing an excluded charge. "
        "If a covered charge requires a receipt, attach an additional_cost_receipt with the same category, registration, date, amount and service details. "
        "This receipt supports the separate charge and does not add to the cost document total. "
        "Gross equals the cost document total plus those "
        "additional costs except in a targeted cost discrepancy scenario. Use [] when there are no extra costs. "
        "Use a short, distinct incident_key label for each scenario. "
        "The server creates the permanent incident ID from the label. "
        "Give a scenario title and conditions using the schema fields. The server derives the scenario type from the claim evidence. "
        "Provide expected_route as an evaluation label only. "
        "Verified fraud indicators and high-severity Fraud findings take precedence. Use fraud escalation when a repair estimate for the reported work "
        "is dated before the incident. Use request more information when missing claimant evidence blocks a required claim check, "
        "including when the gross amount exceeds LKR 250000. When claimant evidence is complete, use adjuster review for an uncovered loss, "
        "out-of-period date, a listed exclusion for a separately claimed charge, gross amount above LKR 250000, "
        "amount above the policy limit, or operational uncertainty. "
        "For a contradictory narrative choose fraud escalation only if the planned factual conflict is material enough to be a high-severity indicator. "
        "Incomplete narratives are not fraud indicators on their own. The server derives the expected route for complete and omission variants using the plan and available history. "
        "Do not generate historical claims, comparables or settlements. "
        "Use YYYY-MM-DD dates and plain decimal strings for LKR amounts, such as \"87500.00\". The policy end date is exclusive. "
        "Briefly describe the submitted evidence and any intentional gap or inconsistency in conditions. Report the expected route separately. "
        "The server assigns record IDs and fixed submission metadata.\n\n"
        "Each example below shows the shape of a single element of cases; use the approved plans for content. "
        "Use IDs only from the saved choices that follow, and return exactly the requested number of new cases.\n"
        f"Repair example: {json.dumps(GENERATION_EXAMPLES[0])}\n"
        f"Theft example: {json.dumps(GENERATION_EXAMPLES[1])}\n\n"
        f"Saved policy and person choices\n{json.dumps(choices)}\n\n"
        "Submit the requested number of new cases through the supplied tool."
    )
    return request_language_checked_batch(client, model, prompt, GENERATION_TOOL, GeneratedCases, count,
                                          lambda case: draft_text(case.claim), excerpts)


def saved_claims_for_policy(session, policy_id):
    return session.execute(select(Claim.payload, Assessment.execution).outerjoin(Assessment, Assessment.claim_id == Claim.id)
        .outerjoin(InvestigationRevision, InvestigationRevision.source_id == Assessment.id)
        .where(Claim.policy_id == policy_id, Claim.payload["source"].as_string() == "Synthetic generator",
               InvestigationRevision.source_id.is_(None))).all()


def scenario_history(saved_claims, policy):
    used = set()
    for payload, execution in saved_claims:
        saved_type = (execution or {}).get("generation_scenario")
        # Older submissions grouped covered and excluded charges under a single scenario.
        if saved_type == "itemised_additional_cost" and not execution.get("generation_plan"):
            used.add(classify_scenario(payload, policy))
        else:
            used.add(saved_type or classify_scenario(payload, policy))
    return used


def assert_capacity(saved_count, requested):
    remaining = max(0, POLICY_CLAIM_LIMIT - saved_count)
    if requested > remaining:
        raise PolicySelectionError(f"This policy can accept {remaining} more synthetic claims.")


def claim_policies(store):
    with store.session() as session:
        counts = dict(session.execute(select(Claim.policy_id, func.count())
                      .outerjoin(Assessment, Assessment.claim_id == Claim.id)
                      .outerjoin(InvestigationRevision, InvestigationRevision.source_id == Assessment.id)
                      .where(Claim.payload["source"].as_string() == "Synthetic generator", InvestigationRevision.source_id.is_(None))
                      .group_by(Claim.policy_id)).all())
    return [{**entry, "remaining_claims": max(0, POLICY_CLAIM_LIMIT - counts.get(entry["policy"]["policy_id"], 0))}
            for entry in list_people_policies(store) if entry["policy"]["starts_on"] <= local_date().isoformat()]


def planning_comparisons(store, session, policy):
    """Give the generator cost evidence from approved synthetic submissions."""
    records = {}
    received_at = now()
    for loss_type in policy["covered_losses"]:
        candidate = SimpleNamespace(id="", received_at=received_at, payload={
            "loss_date": policy["starts_on"], "loss_type": loss_type, "currency": "LKR",
        })
        history = store.history.records(session, candidate, source="Synthetic generator")
        usable = 0
        for item in reversed(history):
            amount = cost_document_amount(item)
            if amount is None or not item["narrative"] or any(doc.get("document_reference") for doc in item["documents"]):
                continue
            records[item["id"]] = {
                "id": item["id"], "loss_type": loss_type, "incident": item["narrative"][:450],
                "cost_items": [line for doc in item["documents"] for line in doc["line_items"]],
                "cost_document_total": str(amount),
            }
            usable += 1
            if usable == 24:
                break
    return records


def omission_reference(plan, claim, policy):
    """Restore only the planned omission on a private copy for reference checks."""
    reference = deepcopy(claim)
    scenario = plan.scenario_type
    documents, cost_type, name = cost_evidence(reference)
    insured_chassis = (policy.get("vehicle") or {}).get("chassis_number")
    if scenario == "missing_loss_date" and not reference["loss_date"]:
        reference["loss_date"] = plan.loss_date.isoformat()
    elif scenario == "missing_chassis" and not (reference.get("vehicle") or {}).get("chassis_number"):
        reference["vehicle"] = {**(reference.get("vehicle") or policy.get("vehicle") or {}), "chassis_number": insured_chassis}
    elif scenario == "missing_cost_document" and not documents:
        reference["documents"].append({"type": cost_type, "name": name, "date": plan.loss_date.isoformat(),
            "registration": policy["registration"],
            **({"garage_name": "Reference garage", "garage_address": "Reference address"} if cost_type == "repair_estimate" else {}),
            "total": str(plan.cost_document_total), "line_items": [item.model_dump(mode="json") for item in plan.cost_items]})
    elif scenario == "missing_itemisation" and len(documents) == 1 and not documents[0]["line_items"]:
        documents[0]["line_items"] = [item.model_dump(mode="json") for item in plan.cost_items]
    elif scenario in {"missing_required_police_report", MISSING_SUPPORTING_DOCUMENT}:
        kind = "police_report" if scenario == "missing_required_police_report" else plan.omitted_supporting_document
        if not any(doc["type"] == kind for doc in reference["documents"]):
            reference["documents"].append({"type": kind, "registration": policy["registration"], "date": plan.loss_date.isoformat()})
    return reference


def apply_plan(plan, claim, policy, comparisons, variant_route):
    scenario_type = plan.scenario_type
    if scenario_type == REUSED_COST_DOCUMENT or (scenario_type == "cost_document_before_loss" and claim["loss_type"] == "repair"):
        return Route.FRAUD, "The scenario introduces a conflicting cost document for fraud investigation."
    if (scenario_type not in COMPLETE_SCENARIOS and scenario_type not in OMISSIONS
            and not plan.estimate_discrepancy and not plan.narrative_variation and not plan.repair_sublimit_test):
        return (variant_route if variant_route != Route.AUTO else Route.REVIEW,
                "The scenario introduces a targeted coverage, cost or fraud exception.")
    omission = OMISSIONS.get(scenario_type)
    discrepancy = plan.estimate_discrepancy
    variation = plan.narrative_variation
    reference = claim
    if omission:
        reference = omission_reference(plan, claim, policy)
    elif discrepancy or variation:
        reference = deepcopy(claim)
    if variation and reference["narrative"] == variation.apply(plan.incident):
        reference["narrative"] = plan.incident
    if (reference["loss_type"] != plan.loss_type or reference["loss_date"] != plan.loss_date.isoformat()
            or reference["narrative"] != plan.incident):
        raise GenerationError(f"The draft loss type, date or incident differs from what was specified in the plan for {scenario_type}.")
    documents, _, _ = cost_evidence(reference)
    if discrepancy and len(documents) == 1:
        lines = documents[0]["line_items"]
        index = discrepancy.line_number - 1
        if index < len(lines) and lines[index]["description"] == discrepancy.estimate_description:
            lines[index]["description"] = plan.cost_items[index].description
    if (len(documents) != 1 or [LineItem.model_validate(line) for line in documents[0]["line_items"]] != plan.cost_items
            or [AdditionalCost.model_validate(cost) for cost in reference["additional_costs"]] != plan.additional_costs):
        raise GenerationError(f"The draft cost evidence differs from what was specified in the plan for {scenario_type}.")
    cost_document = documents[0]
    cost_document["total"] = str(plan.cost_document_total)
    reference["gross_amount"] = str(plan.gross_amount)
    if not is_complete_reference_claim(reference, policy):
        observed = classify_scenario(reference, policy).replace("_", " ")
        raise GenerationError(f"The draft for {scenario_type.replace('_', ' ')} failed its reference checks. Detected scenario: {observed}.")
    if plan.repair_sublimit_test:
        reason = exceeded_repair_sublimit(policy, plan.repair_sublimit_test, plan.cost_items)
        if cost_document["type"] != "repair_estimate" or reason is None:
            raise GenerationError("The completed repair estimate did not show the planned sublimit excess.")
        return Route.REVIEW, reason
    if omission or discrepancy or variation:
        claim["gross_amount"] = reference["gross_amount"]
        for submitted in cost_evidence(claim)[0]:
            submitted["total"] = cost_document["total"]
    if omission:
        if scenario_type == MISSING_SUPPORTING_DOCUMENT:
            omitted = plan.omitted_supporting_document
            claim["documents"] = [doc for doc in claim["documents"] if doc["type"] != omitted]
            omission = SUPPORTING_DOCUMENT_LABELS[omitted]
        else:
            omit_required_evidence(claim, scenario_type)
        return Route.INFO, f"The submitted claim deliberately omits required evidence. {omission}."
    if discrepancy:
        cost_evidence(claim)[0][0]["line_items"][discrepancy.line_number - 1]["description"] = discrepancy.estimate_description
        return Route.REVIEW, f"The repair estimate includes work on the {discrepancy.unaffected_part}, which the claimant says was undamaged."
    if variation:
        claim["narrative"] = variation.apply(plan.incident)
        if scenario_type == CONTRADICTORY_NARRATIVE:
            route = Route.FRAUD if variant_route == Route.FRAUD else Route.REVIEW
            return route, "The narrative gives conflicting accounts. Compare both with the attached documents and assess their significance."
    if plan.gross_amount > AUTHORITY:
        return Route.REVIEW, "The amount needed for a positive payable exceeds LKR 250,000 and requires human review."
    amounts = [Decimal(comparisons[identifier]["cost_document_total"]) for identifier in plan.comparison_ids]
    if len(amounts) < MIN_COMPARABLES:
        return Route.REVIEW, "The plan lacks five suitable staff-approved synthetic cost comparisons."
    if not min(amounts) <= Decimal(cost_document["total"]) <= max(amounts):
        return Route.REVIEW, "The cost document total is outside the range of the selected approved comparisons."
    return Route.AUTO, "The planned cost falls within the selected approved comparisons. Specialists will verify the claim after submission."


def complete_draft_amounts(claim, scenario_type):
    documents, _, _ = cost_evidence(claim)
    for document in documents:
        if document["total"] is None:
            if scenario_type == "cost_document_arithmetic_mismatch":
                raise GenerationError("The arithmetic mismatch scenario needs an explicit cost document total that differs from its line items.")
            document["total"] = str(sum((Decimal(item["amount"]) for item in document["line_items"]), Decimal("0.00")))
    if claim["gross_amount"] is None:
        if scenario_type in {"unexplained_additional_cost", "partial_amount"}:
            raise GenerationError("The claimed amount discrepancy scenario needs an explicit gross amount that differs from the itemised costs.")
        total, additional = cost_breakdown(claim, documents)
        claim["gross_amount"] = str(total + additional)


def classify_prepared_scenario(claim, policy, plan):
    scenario = classify_scenario(claim, policy)
    if (scenario in COMPLETE_SCENARIOS and plan.get("scenario_type") == REPAIR_SUBLIMIT_EXCEEDED
            and plan.get("repair_sublimit_test")):
        documents, _, _ = cost_evidence(claim)
        if (claim["loss_type"] == "repair" and len(documents) == 1 and documents[0]["type"] == "repair_estimate"
                and not claim.get("additional_costs") and is_complete_reference_claim(claim, policy)):
            test = RepairSublimitTest.model_validate(plan["repair_sublimit_test"])
            items = [LineItem.model_validate(item) for item in documents[0]["line_items"]]
            if len(items) >= 2 and exceeded_repair_sublimit(policy, test, items):
                return REPAIR_SUBLIMIT_EXCEEDED
    narrative_variation = plan.get("narrative_variation")
    if scenario in COMPLETE_SCENARIOS and narrative_variation and plan.get("scenario_type") in NARRATIVE_SCENARIOS:
        variation = NarrativeVariation.model_validate(narrative_variation)
        if claim["narrative"] == variation.apply(plan["incident"]):
            return plan["scenario_type"]
    discrepancy = plan.get("estimate_discrepancy")
    if discrepancy and scenario in COMPLETE_SCENARIOS and claim["loss_type"] == "repair":
        documents, _, _ = cost_evidence(claim)
        expected_items = [LineItem.model_validate(item) for item in plan["cost_items"]]
        index = discrepancy["line_number"] - 1
        expected_items[index].description = discrepancy["estimate_description"]
        if (claim["narrative"] == plan["incident"] and len(documents) == 1
                and [LineItem.model_validate(item) for item in documents[0]["line_items"]] == expected_items):
            return ESTIMATE_NARRATIVE_MISMATCH
    return scenario


def preview_batch(store, count, batch_id, policy_id, *, discarded_previews=()):
    """Validate an entire model-generated batch before making it available for review."""
    if not 1 <= count <= POLICY_CLAIM_LIMIT:
        raise PolicySelectionError("Generate between 1 and 10 claims for the selected policy.")
    entry = next((entry for entry in list_people_policies(store) if entry["policy"]["policy_id"] == policy_id), None)
    today = local_date()
    if entry is None or entry["policy"]["starts_on"] > today.isoformat():
        raise PolicySelectionError("Select a saved synthetic policy before generating claims.")
    person, policy = entry["person"], entry["policy"]
    with store.session() as session:
        saved = saved_claims_for_policy(session, policy_id)
        discarded_incidents = []
        if discarded_previews:
            preview_policies = {identifier for identifier, _ in discarded_previews}
            active_policies = set(session.scalars(select(Policy.id).where(Policy.id.in_(preview_policies))))
            discarded_incidents = [incident for identifier, incident in discarded_previews if identifier in active_policies]
        recent_incidents = [
            {"loss_type": payload["loss_type"], "incident": payload["narrative"][:500]}
            for payload in session.scalars(select(Claim.payload).where(Claim.payload["source"].as_string() == "Synthetic generator")
                                           .order_by(Claim.sequence.desc()).limit(12))
            if payload.get("narrative")
        ]
        wording = generated_wording_for_policy(session, policy_id)
        wording_text = base64.b64decode(wording.content).decode("utf-8-sig") if wording else ""
        excerpts = policy_phrases(policy, wording_text)
        prior_documents = prior_cost_documents(session, policy["covered_losses"], policy["registration"])
        safe_documents = ((row, doc) for row, doc in prior_documents
                          if not copied_fields(stored_cost_text(doc), excerpts))
        reusable = {doc["id"]: {"document": doc, "loss_type": row.payload["loss_type"]}
                    for row, doc in islice(safe_documents, 8)}
        comparisons = planning_comparisons(store, session, policy)
    assert_capacity(len(saved), count)
    used = scenario_history(saved, policy)
    options = available_scenarios(policy, today)
    if reusable:
        options[REUSED_COST_DOCUMENT] = "An in-period covered loss resubmits a stored cost document from a different vehicle."
    reuse_choices = [{"id": identifier, "loss_type": item["loss_type"], "total": item["document"]["total"]}
                     for identifier, item in reusable.items()]
    if len(options.keys() - used) < count:
        raise PolicySelectionError("The selected policy has too few unused scenario types for this preview.")
    fixtures = []
    incident_ids = set()
    with generation_client() as (client, model):
        for offset in range(0, count, BATCH_SIZE):
            size = min(BATCH_SIZE, count - offset)
            remaining = {key: description for key, description in options.items() if key not in used}
            reuse_options = reuse_choices if REUSED_COST_DOCUMENT in remaining else []
            plans = request_plans(client, model, size, entry, remaining, comparisons, reuse_options, today, excerpts,
                                  recent_incidents=recent_incidents, discarded_incidents=discarded_incidents)
            recent_incidents.extend({"loss_type": plan.loss_type, "incident": plan.incident[:500]} for plan in plans)
            cases = request_cases(client, model, size, batch_id, offset, [entry], plans=plans,
                                  scenario_options={plan.scenario_type: remaining[plan.scenario_type] for plan in plans},
                                  reusable_documents=reuse_options, excerpts=excerpts)
            for plan, case in zip(plans, cases, strict=True):
                if case.claim.policy_id != policy_id or case.claim.person_id != person["person_id"]:
                    raise GenerationError("The model returned a claim with a mismatched person and policy. Try generating again.")
                claim_id = str(uuid.uuid4())
                short = claim_id.replace("-", "").upper()
                incident_id = "INC-" + digest([batch_id, offset, policy["policy_id"], case.claim.incident_key])[:32].upper()
                if incident_id in incident_ids:
                    raise GenerationError("Claude assigned the same incident label to multiple claims in one group.")
                incident_ids.add(incident_id)
                prefixes = {"police_report": "PR", "additional_cost_receipt": "REC", "repair_estimate": "EST",
                            "incident_report": "IR", "witness_statement": "WS", "vehicle_valuation": "VAL",
                            "third_party_demand": "TPD"}
                documents = [{"id": f"{prefixes[document.type]}-{short}-{index}",
                              **document.model_dump(mode="json")}
                             for index, document in enumerate(case.claim.documents, 1)]
                reused_id = case.claim.reused_cost_document_id
                if reused_id:
                    prior = reusable.get(reused_id)
                    if (not prior or prior["loss_type"] != case.claim.loss_type or len(documents) >= 5
                            or any(document["type"] not in {"police_report", *SUPPORTING_DOCUMENT_LABELS} for document in documents)):
                        raise GenerationError("Claude must select an available cost document for this loss type and omit other cost documents.")
                    prefix = prefixes[prior["document"]["type"]]
                    documents.append({**prior["document"], "id": f"{prefix}-{short}-{len(documents) + 1}", "document_reference": reused_id})
                claim = {
                    "claim_id": claim_id, "revision": 1, "registration": policy["registration"],
                    "incident_id": incident_id,
                    "currency": "LKR", "source": "Synthetic generator", "documents": documents,
                    **case.claim.model_dump(mode="json", exclude={"documents", "incident_key", "reused_cost_document_id"}),
                }
                expected, expectation = apply_plan(plan, claim, policy, comparisons, case.expected_route)
                complete_draft_amounts(claim, plan.scenario_type)
                if plan.include_optional_police_report and not any(doc["type"] == "police_report" for doc in claim["documents"]):
                    raise GenerationError("The narrative scenario is missing the police report selected in its plan.")
                document_check_claim = ({**claim, "loss_date": plan.loss_date.isoformat()}
                                        if plan.scenario_type == "missing_loss_date" else claim)
                if claim["loss_type"] == plan.loss_type and policy_applies_to_loss(policy, document_check_claim):
                    expected_support = sorted(rule["document_type"] for rule in supporting_document_rules(policy, plan.loss_type)
                                              if rule["document_type"] != plan.omitted_supporting_document
                                              and (rule["requirement"] == "required"
                                                   or rule["document_type"] in plan.optional_supporting_documents))
                    actual_support = sorted(doc["type"] for doc in claim["documents"] if doc["type"] in SUPPORTING_DOCUMENT_LABELS)
                    if expected_support != actual_support:
                        raise GenerationError("Claude returned supporting documents that differ from the approved claim plan.")
                plan_record = plan.model_dump(mode="json")
                scenario_key = classify_prepared_scenario(claim, policy, plan_record)
                if reused_id and scenario_key != REUSED_COST_DOCUMENT:
                    raise GenerationError("Claude must use an in-period covered loss for a reused cost document.")
                if scenario_key != plan.scenario_type:
                    raise GenerationError("Claude returned evidence for a different scenario than the approved plan.")
                used.add(scenario_key)
                document_checklist = [
                    {**rule, "included": any(doc["type"] == rule["document_type"] and doc.get("category") == rule.get("category")
                                             for doc in claim["documents"])}
                    for rule in coverage_document_checklist(policy, claim["loss_type"])
                    if claim["loss_type"] in policy["covered_losses"]
                    and ("category" not in rule or any(cost.get("category") == rule["category"] for cost in claim["additional_costs"]))
                ]
                omission = SUPPORTING_DOCUMENT_LABELS[plan.omitted_supporting_document] if plan.omitted_supporting_document else OMISSIONS.get(scenario_key)
                plan_record.update(omission=omission, expected_route=expected.value, expectation=expectation, documents=document_checklist)
                fixtures.append({"claim": claim, "person": person, "scenario_key": scenario_key,
                                 "scenario": plan.scenario, "conditions": case.conditions, "expected": expected.value,
                                 "plan": plan_record})
    return {"model": model, "fixtures": fixtures, "policy_id": policy_id, "policy_hash": digest(policy),
            "wording_id": wording.id if wording else None}


def commit_batch(store, labels_path, batch_id, preview):
    """Save the reviewed payloads and keep generation labels out of workflow evidence."""
    manifest = {**execution_manifest(), "generation_model": preview["model"]}
    # Read the processing mode at commit, independently of generation mode.
    enabled, model = settings()
    manifest.update(workflow="claude-intake-first-v3", mode="anthropic-tools" if enabled else "deterministic-tools",
                    model=model if enabled else None)
    labels, ids = [], []
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(labels_path)) as label_db:
        label_db.execute("CREATE TABLE IF NOT EXISTS labels (assessment_id TEXT PRIMARY KEY,batch_id TEXT,expected TEXT,scenario TEXT,seed INTEGER)")
        if "conditions" not in {column[1] for column in label_db.execute("PRAGMA table_info(labels)")}:
            label_db.execute("ALTER TABLE labels ADD COLUMN conditions TEXT")
        with label_db, store.session.begin() as session:
            session.connection().exec_driver_sql("BEGIN IMMEDIATE")
            policy_id = preview["policy_id"]
            saved = saved_claims_for_policy(session, policy_id)
            assert_capacity(len(saved), len(preview["fixtures"]))
            saved_policy = session.get(Policy, policy_id)
            holder = session.get(SyntheticPolicyHolder, policy_id)
            if (saved_policy is None or digest(saved_policy.payload) != preview["policy_hash"]
                    or holder is None or session.get(Person, holder.person_id) is None):
                raise PolicySelectionError("A saved person or policy changed. Generate a new claim preview.")
            policy = saved_policy.payload
            wording = generated_wording_for_policy(session, policy_id)
            wording_id = wording.id if wording else None
            if wording_id != preview["wording_id"]:
                raise PolicySelectionError("The policy wording changed. Generate a new claim preview.")
            policy_context = generated_policy_context({"synthetic_wording_document_id": wording_id} if wording_id else {})
            used = scenario_history(saved, policy)
            offset = session.scalar(select(func.max(Claim.sequence))) or 0
            for sequence, fixture in enumerate(preview["fixtures"], offset + 1):
                claim = fixture["claim"]
                scenario_key = fixture["scenario_key"]
                if (claim["policy_id"] != policy_id or scenario_key in used
                        or scenario_key != classify_prepared_scenario(claim, policy, fixture["plan"])):
                    raise PolicySelectionError("A scenario in this preview is already saved for this policy. Generate a new preview.")
                used.add(scenario_key)
                if holder.person_id != claim["person_id"]:
                    raise PolicySelectionError("A saved person or policy changed. Generate a new claim preview.")
                identifier = str(uuid.uuid4())
                session.add(Claim(id=claim["claim_id"], sequence=sequence, batch_id=batch_id, policy_id=policy_id,
                                  payload=claim, payload_hash=digest(claim)))
                session.flush()
                session.add(Assessment(id=identifier, claim_id=claim["claim_id"],
                    execution={**manifest, "generation_scenario": scenario_key, "generation_plan": fixture["plan"]},
                    context={"findings": {}, "skipped": {}, "environment": "synthetic", "rules": "provisional-motor-v1",
                             "workflow": manifest["workflow"], **policy_context, "version": 0}))
                store.add_event(session, identifier, "submitted", {"status": "queued"})
                ids.append(identifier)
                labels.append((identifier, batch_id, fixture["expected"], fixture["scenario"], fixture["conditions"]))
            label_db.executemany("INSERT INTO labels (assessment_id,batch_id,expected,scenario,conditions) VALUES(?,?,?,?,?)", labels)
    return {"batch_id": batch_id, "count": len(ids), "assessment_ids": ids}
