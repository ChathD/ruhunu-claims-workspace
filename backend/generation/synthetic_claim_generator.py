"""Draft Sri Lankan incidents from basic reported identities and save claims for processing."""
import json
import random
import sqlite3
import uuid
from collections import Counter, defaultdict
from contextlib import closing
from datetime import date, timedelta
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import Field, model_validator
from sqlalchemy import func, literal_column, select

from backend.assurance.execution_manifest import execution_manifest
from backend.assurance.investigation_tracing import operation_trace
from backend.claim_schemas import AUTHORITY, SCENARIO_ROUTES, AdditionalCost, ChargeCategory, Money, MotorLossType, VehicleEvidence
from backend.evidence.motor_benefits import MotorBenefit
from backend.evidence.required_claim_evidence import cost_line_total
from backend.evidence.synthetic_claim_text import BLOCK_SEPARATOR, DOCUMENT_SOURCES, SOURCE_BY_DOCUMENT_TYPE, extract_claim_text, render_claim_text
from backend.generation.claude_generation_client import (
    BATCH_SIZE, GeneratedData, GenerationError, GenerationValidationError, generation_client, local_date, request_batch,
)
from backend.generation.policy_schedule_generator import identity_key
from backend.storage.postgres import begin_write
from backend.storage.sqlite_store import (
    Assessment, Claim, ClaimArchive, EvaluationLabel, PolicySchedule, SyntheticClaimAttachment, SyntheticClaimPreview, SyntheticClaimSchedule, WorkflowEvent, digest,
)
from backend.workflow.specialist_readiness import WORKFLOW

GENERATION_TOOL = "submit_synthetic_claims"
BELOW_THRESHOLD = "below_threshold"
ABOVE_THRESHOLD = "above_threshold"
SupportingDocumentType = Literal["incident_report", "witness_statement", "work_incapacity_evidence", "earnings_evidence"]
PROPERTY_COST_TYPES = {"repair": "repair_estimate", "third_party_property_damage": "third_party_demand"}
INCIDENT_PROFILES: dict[str, tuple[MotorLossType, str]] = {
    "parked_vehicle_damage": ("repair", "A parked vehicle is struck or damaged during manoeuvring."),
    "junction_collision": ("repair", "A junction collision damages the insured vehicle."),
    "rear_end_collision": ("repair", "A rear-end collision damages the insured vehicle in traffic."),
    "monsoon_flooding": ("repair", "Floodwater damages the insured vehicle during monsoon rain."),
    "windscreen_damage": ("repair", "Road debris damages the windscreen."),
    "falling_object_damage": ("repair", "A fallen branch or other object damages the vehicle."),
    "parked_vehicle_theft": ("theft", "A parked vehicle is missing when its driver returns."),
    "overnight_vehicle_theft": ("theft", "A vehicle is reported stolen overnight from a residential area."),
    "car_park_theft": ("theft", "A vehicle goes missing from a commercial car park."),
    "third_party_vehicle_damage": ("third_party_property_damage", "Another vehicle's owner claims damage after a collision."),
    "roadside_property_damage": ("third_party_property_damage", "The incident damages a third party's roadside property."),
    "parking_property_damage": ("third_party_property_damage", "A parking manoeuvre damages a third party's gate or wall."),
    "pedestrian_injury": ("third_party_injury", "A pedestrian reports injury following a vehicle incident."),
    "cyclist_injury": ("third_party_injury", "A cyclist reports injury following a vehicle incident."),
    "other_vehicle_occupant_injury": ("third_party_injury", "An occupant of another vehicle reports injury after a collision."),
}


class LineItem(GeneratedData):
    description: str = Field(min_length=1, max_length=180)
    amount: Money


class CostDocument(GeneratedData):
    name: str = Field(min_length=1, max_length=150)
    registration: str | None = Field(min_length=1, max_length=40)
    date: date
    total: Money | None = Field(default=None, description="Printed total, if visible on the document.")
    line_items: list[LineItem] = Field(default_factory=list, max_length=10,
                                       description="Visible itemised charges with a description and LKR amount per item.")


class RepairEstimate(CostDocument):
    type: Literal["repair_estimate"]
    garage_name: str = Field(min_length=1, max_length=150)
    garage_address: str = Field(min_length=1, max_length=300)


class Document(CostDocument):
    type: Literal["vehicle_valuation", "third_party_demand"]
    chassis_number: str | None = Field(default=None, min_length=1, max_length=40)


class MotorBenefitReceipt(CostDocument):
    type: Literal["motor_benefit_receipt"]
    service: MotorBenefit
    provider: str = Field(min_length=1, max_length=150)
    service_location: str = Field(min_length=1, max_length=150)
    payment_status: Literal["paid", "pro_forma"]
    beneficiary: str | None = Field(default=None, min_length=1, max_length=120)
    service_code: str | None = Field(default=None, min_length=1, max_length=40)
    asset_serial: str | None = Field(default=None, min_length=1, max_length=40)
    part_serial: str | None = Field(default=None, min_length=1, max_length=40)


class PurchaseReceipt(GeneratedData):
    type: Literal["purchase_receipt"]
    name: str = Field(min_length=1, max_length=150)
    registration: str = Field(min_length=1, max_length=40)
    date: date
    provider: str = Field(min_length=1, max_length=150)
    asset_serial: str = Field(min_length=1, max_length=40)
    part_serial: str = Field(min_length=1, max_length=40)
    description: str = Field(min_length=10, max_length=450)


class PoliceReport(GeneratedData):
    type: Literal["police_report"]
    name: str = Field(min_length=1, max_length=150)
    registration: str = Field(min_length=1, max_length=40)
    date: date
    incident_date: date | None = Field(default=None, exclude_if=lambda value: value is None,
                                       description="Date in the incident register, separate from the report issue date.")
    incident_location: str | None = Field(default=None, min_length=1, max_length=120, exclude_if=lambda value: value is None)
    commercial_trip_verified: bool | None = Field(default=None, exclude_if=lambda value: value is None,
        description="The police checked a passenger booking and recorded a paid trip underway at the incident.")
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
    summary: str = Field(min_length=10, max_length=650)


class IndependentAssessorReport(GeneratedData):
    type: Literal["independent_assessor_report"]
    name: str = Field(min_length=1, max_length=150)
    registration: str | None = Field(default=None, min_length=1, max_length=40)
    date: date
    inspection_date: date = Field(description="Date the assessor inspected the property or reviewed the evidence.")
    inspection_basis: Literal["physical_inspection", "photographs", "document_review"]
    author: str = Field(min_length=1, max_length=120)
    summary: str = Field(min_length=10, max_length=750,
                         description="Describe the evidence reviewed, the repair cost or pre-loss vehicle value findings, and any limitations.")
    assessed_amount_lkr: Money | None = Field(default=None,
        description="Independent estimate of supported repair costs or pre-loss vehicle value, when established.")
    prior_damage_recorded_on: date | None = Field(default=None, exclude_if=lambda value: value is None,
        description="Date of a condition record identifying old damage separately from damage in this incident.")


FiledDocument = Annotated[
    RepairEstimate | Document | MotorBenefitReceipt | PurchaseReceipt | PoliceReport | AdditionalCostReceipt | SupportingDocument | IndependentAssessorReport,
    Field(discriminator="type"),
]
IncidentNarrative = Annotated[str, Field(min_length=10, description="Concise factual account of the incident and resulting loss.")]


class IncidentDetails(GeneratedData):
    collision_type: str | None = Field(default=None, max_length=80)
    weather: str | None = Field(default=None, max_length=80)
    other_party: str | None = Field(default=None, max_length=120)
    at_fault_party: Literal["insured", "other_party", "shared"] | None = None
    liability_split_insured_pct: int | None = Field(default=None, ge=0, le=100)
    estimated_impact_speed_kmh: str | None = Field(default=None, pattern=r"^\d{1,3}(?:-\d{1,3})?$")
    airbags_deployed: bool | None = None
    ems_at_scene: bool | None = None
    initial_repair_estimate_lkr: Money | None = Field(default=None, gt=0)
    icd10_codes: list[str] = Field(default_factory=list, max_length=5)
    claimed_injuries: list[str] = Field(default_factory=list, max_length=5)
    medical_facility: str | None = Field(default=None, max_length=120)
    medical_intake_date: date | None = None
    medical_intake_filed: bool = Field(default=False, description="A medical intake record was actually filed with this claim.")
    adjuster_note_filed: bool = Field(default=False, description="The claim file includes a separate adjuster diary entry.")
    law_firm: str | None = Field(default=None, max_length=120)


class ClaimForm(GeneratedData):
    loss_type: MotorLossType
    loss_date: date | None
    reported_date: date
    gross_amount: Money = Field(gt=0)
    additional_costs: list[AdditionalCost] = Field(max_length=10)
    narrative: IncidentNarrative
    incident_details: IncidentDetails = Field(default_factory=IncidentDetails)

    @model_validator(mode="after")
    def validate_dates(self):
        if self.reported_date > local_date() or (self.loss_date and self.loss_date > self.reported_date):
            raise ValueError("Claim dates must be chronological and at or before today")
        if (seen := self.incident_details.medical_intake_date) and (seen > local_date() or self.loss_date and seen < self.loss_date):
            raise ValueError("Medical treatment must be at or after the loss and at or before today")
        details = self.incident_details
        if details.medical_intake_filed and not any((details.claimed_injuries, details.icd10_codes, details.medical_facility, details.medical_intake_date)):
            raise ValueError("A filed medical intake must include a detail from the record.")
        return self


class ClaimScenario(ClaimForm):
    documents: list[FiledDocument] = Field(max_length=5)

    @model_validator(mode="after")
    def validate_document_dates(self):
        for document in self.documents:
            if document.date > local_date():
                raise ValueError("Document dates must be at or before today")
            if isinstance(document, IndependentAssessorReport) and (
                    document.inspection_date > document.date or self.loss_date and document.inspection_date < self.loss_date):
                raise ValueError("The assessment date must be on or after the incident and on or before the report date")
        return self


class ClaimDraft(ClaimScenario):
    insured_name: str = Field(min_length=2, max_length=120)
    registration: str = Field(min_length=1, max_length=40)
    policy_id: str = Field(min_length=1, max_length=64)
    vehicle: VehicleEvidence | None = None


class GeneratedCase(GeneratedData):
    scenario: str = Field(min_length=1, description="Brief title for the scenario.")
    scenario_type: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    claim: ClaimDraft


class GeneratedCases(GeneratedData):
    cases: list[GeneratedCase] = Field(min_length=1, max_length=BATCH_SIZE)


class GeneratedIncident(GeneratedCase):
    claim: ClaimScenario


class GeneratedIncidents(GeneratedData):
    cases: list[GeneratedIncident] = Field(min_length=1, max_length=BATCH_SIZE)


def sample_claim_identities(store, count):
    with store.session() as session:
        schedules = session.scalars(select(PolicySchedule.payload)).all()
    if not schedules:
        raise ValueError("Generate a policy schedule before generating claims.")
    today = local_date()
    eligible = [schedule for schedule in schedules if date.fromisoformat(schedule["starts_on"]) <= today]
    if not eligible:
        raise ValueError("Claim generation needs a policy schedule whose period has already started.")
    return [{"insured_name": schedule["insured_name"], "registration": schedule["vehicle"]["registration"],
             "vehicle": {field: schedule["vehicle"][field] for field in ("make", "model", "year", "chassis_number")},
             "loss_date": sample_loss_date(date.fromisoformat(schedule["starts_on"]), date.fromisoformat(schedule["ends_on"]), today)}
            for schedule in random.choices(eligible, k=count)]


def sample_loss_date(inception, expiry, today):
    earliest = max(inception, today - timedelta(days=548))
    windows = []
    last_insured_day = min(expiry, today)
    if earliest <= last_insured_day:
        windows.append((earliest, last_insured_day))
    if expiry < today:
        windows.append((max(earliest, expiry + timedelta(days=1)), today))
    start, end = random.choice(windows)
    return (start + timedelta(days=random.randint(0, (end - start).days))).isoformat()


def claim_amount_band(amount):
    if amount is not None:
        value = Decimal(str(amount))
        if value < AUTHORITY:
            return BELOW_THRESHOLD
        if value > AUTHORITY:
            return ABOVE_THRESHOLD
    return None


def claim_scenario_history(store, discarded=()):
    with store.session() as session:
        if not store.database_url:
            session.connection().exec_driver_sql("BEGIN")
        records = session.execute(select(Claim.payload["insured_name"].as_string(), Claim.payload["loss_type"].as_string(),
            Assessment.execution["generation_scenario"].as_string(), Claim.payload["gross_amount"].as_string()).select_from(Claim).join(Assessment).where(
                Claim.payload["source"].as_string() == "Synthetic generator", Claim.payload["parent_claim_id"].as_string().is_(None),
                Claim.id.not_in(select(ClaimArchive.claim_id))).order_by(Claim.sequence)).all()
        order = SyntheticClaimPreview.batch_id if store.database_url else literal_column("rowid")
        previews = session.scalars(select(SyntheticClaimPreview.payload).order_by(order))
        pending = [item for preview in previews for item in preview["fixtures"]]
    history = [(name, loss, profile, claim_amount_band(amount)) for name, loss, profile, amount in records]
    history.extend((item["claim"]["insured_name"], item["claim"]["loss_type"], item["scenario_type"],
                    claim_amount_band(item["claim"]["gross_amount"])) for item in [*discarded, *pending])
    return history


def plan_claim_scenarios(identities, history=()):
    holder_losses, holder_profiles = defaultdict(Counter), defaultdict(Counter)
    batch_losses, batch_profiles = Counter(), Counter()
    last_loss = {}
    last_band = None
    for name, loss, profile, band in history:
        holder = identity_key(name)
        holder_losses[holder][loss] += 1
        holder_profiles[holder][profile] += 1
        last_loss[holder] = loss
        if band:
            last_band = band

    slots = []
    profiles = list(INCIDENT_PROFILES)
    include_theft_assessor = bool(random.getrandbits(1))
    for subject in identities:
        holder = identity_key(subject["insured_name"])
        band = ABOVE_THRESHOLD if last_band == BELOW_THRESHOLD else BELOW_THRESHOLD
        random.shuffle(profiles)

        def rank(profile):
            loss = INCIDENT_PROFILES[profile][0]
            return (loss == last_loss.get(holder), holder_losses[holder][loss], batch_losses[loss],
                    band == ABOVE_THRESHOLD and loss != "theft",
                    holder_profiles[holder][profile], batch_profiles[profile])

        eligible = [profile for profile in profiles if band == ABOVE_THRESHOLD or INCIDENT_PROFILES[profile][0] != "theft"]
        selected = min(eligible, key=rank)
        loss, details = INCIDENT_PROFILES[selected]
        slot = {"scenario_type": selected, "loss_type": loss, "incident": details, "amount_band": band}
        if loss == "theft":
            slot["include_assessor_report"] = include_theft_assessor
            include_theft_assessor = not include_theft_assessor
        slots.append(slot)
        holder_losses[holder][loss] += 1
        holder_profiles[holder][selected] += 1
        batch_losses[loss] += 1
        batch_profiles[selected] += 1
        last_loss[holder] = loss
        last_band = band
    return slots


def claim_labels(claim, insured_name, details):
    repair = next((item for item in claim["documents"] if item["type"] == "repair_estimate"), None)
    final = repair["total"] if repair else None
    initial = details.get("initial_repair_estimate_lkr") if repair else None
    if initial is not None and final is not None and Decimal(initial) == Decimal(final):
        initial = None
    flags = []
    loss_date = claim["loss_date"]
    if repair and loss_date and repair.get("date") and repair["date"] < loss_date:
        flags.append("cost_document_predates_loss")
    if loss_date and details.get("claimed_injuries") and details.get("medical_intake_date"):
        delay = (date.fromisoformat(details["medical_intake_date"]) - date.fromisoformat(loss_date)).days
        if delay >= 4:
            flags.append(f"delayed_treatment_{delay}_days_post_loss")
    vehicle = claim.get("vehicle") or {}
    fault = {"insured": insured_name, "other_party": details.get("other_party"), "shared": "shared"}
    return {
        "incident_date": loss_date, "loss_type": claim["loss_type"], "collision_type": details.get("collision_type"),
        "weather": details.get("weather"), "insured_party": insured_name, "submitted_policy_number": claim.get("policy_id"),
        "insured_vehicle": " ".join(str(vehicle[key]) for key in ("year", "make", "model") if vehicle.get(key)) or None,
        "at_fault_party": fault.get(details.get("at_fault_party")), "liability_split_insured_pct": details.get("liability_split_insured_pct"),
        "estimated_impact_speed_kmh": details.get("estimated_impact_speed_kmh"), "airbags_deployed": details.get("airbags_deployed"),
        "ems_at_scene": details.get("ems_at_scene"), "initial_repair_estimate_lkr": initial or final,
        "supplement_repair_estimate_lkr": final if initial is not None else None, "final_repair_estimate_lkr": final,
        "icd10_codes": details.get("icd10_codes", []), "claimed_injuries": details.get("claimed_injuries", []),
        "attorney_represented": bool(details.get("law_firm")), "law_firm": details.get("law_firm"),
        "siu_flag_count": len(flags), "fraud_siu_flags": flags,
    }


def render_submission(draft, documents, identity, noise_level):
    claim_id = str(uuid.uuid4())
    filings = []
    for index, item in enumerate(documents, 1):
        fields = item if isinstance(item, dict) else item.model_dump(mode="json")
        document = {**fields, "id": f"DOC-{claim_id}-{index}"}
        if isinstance(item, CostDocument) and item.total is None and item.line_items:
            document["total"] = str(cost_line_total(document["line_items"]))
        filings.append(document)
    claim = {"claim_id": claim_id, "incident_id": str(uuid.uuid4()), "revision": 1, "currency": "LKR",
             "source": "Synthetic generator", "documents": filings,
             **draft.model_dump(mode="json", exclude={"documents", "incident_details"}), **identity}
    details = draft.incident_details.model_dump(mode="json")
    labels = claim_labels(claim, claim["insured_name"], details)
    claim["unstructured_inputs"] = render_claim_text(claim, {"full_name": claim["insured_name"]}, details, noise_level)
    claim.update(extract_claim_text(claim["unstructured_inputs"]))
    return claim, labels


def preview_batch(count, batch_id, *, identities=None, scenario_plan=None, discarded_previews=(), noise_level=0.2, actor_id=None):
    if identities is not None:
        if len(identities) != count or scenario_plan is not None and len(scenario_plan) != count:
            raise ValueError("Provide a policyholder, vehicle and scenario for each claim.")
        if scenario_plan is None:
            scenario_plan = plan_claim_scenarios(identities)
    with operation_trace("generator.preview", batch_id, metadata={"batch_size": count}, actor_id=actor_id) as span:
        fixtures = []
        incidents = list(discarded_previews)
        with generation_client() as (client, model):
            for offset in range(0, count, BATCH_SIZE):
                size = min(BATCH_SIZE, count - offset)
                subjects = identities[offset:offset + size] if identities is not None else None
                schema = GeneratedIncidents if subjects is not None else GeneratedCases
                slots = scenario_plan[offset:offset + size] if scenario_plan is not None else None
                identity_instructions = ("Use the supplied people and vehicles in the given order, one incident each. "
                    "Use the assigned loss_date exactly when supplied, and use it consistently in the incident account and evidence. "
                    "Report the claim on or after the loss date, at or before today. Use dates in the past 18 months. "
                    "Generate each incident independently of any insurance or expected outcome. "
                    "Do not state policy terms, fabricate a policy number or decide coverage.\n\n"
                    f"People, vehicles and assigned loss dates: {json.dumps(subjects, ensure_ascii=False)}\n\n"
                    "For each entry in order, draft an incident consistent with the assigned profile. Copy its scenario_type and loss_type exactly. "
                    f"Match its amount_band. For {BELOW_THRESHOLD}, set the gross amount strictly below LKR {AUTHORITY:,.0f}. "
                    f"For {ABOVE_THRESHOLD}, set it strictly above LKR {AUTHORITY:,.0f}. Use realistic damage, injury severity and costs "
                    "for the selected scale and vehicle. Keep the gross amount, any additional charges, and any submitted cost records consistent "
                    "with the incident. Keep the amount band out of the claim text. "
                    f"Assigned scenarios: {json.dumps(slots, ensure_ascii=False)}\n\n") if subjects is not None else (
                    "Invent each claimant name, vehicle, registration and fictional policy reference. Do not decide coverage. "
                    "Sample parked vehicle damage, rear-end or junction collisions, monsoon flooding, theft and third-party losses.\n\n")
                prompt = (
                    f"Draft {size} independent fictional Sri Lankan motor claim submissions. Today is {local_date()}. "
                    + identity_instructions +
                    "Use distinct Sri Lankan localities and incident accounts. Use a short scenario title, snake_case scenario_type and one supported loss_type. "
                    "Use YYYY-MM-DD for submitted dates, LKR decimals for amounts and km/h for speed.\n\n"
                    "Work out each incident first, then decide what the claimant reports and what records actually accompany this claim. "
                    "Choose the documents and optional facts for that incident only. An empty documents list is valid. Leave irrelevant fields unset. "
                    "A parked vehicle theft may have no impact speed, repair quote, injury or medical facts. A collision can be reported with a quote and no police filing. "
                    "When you include evidence, check that its parties, dates, loss, damage and charges fit the incident. "
                    "A deliberate discrepancy may be part of the incident, but unrelated evidence is not. "
                    "Use the document type from the schema that matches each filed record. "
                    "For itemised documents put the entries in line_items, each with description and amount. "
                    "Leave line_items empty if the filed document shows only a total. Provide a fictional garage name and address for estimates. "
                    "Use station and summary for police reports, and author and summary for other supporting records. "
                    "Keep police and supporting document summaries under 400 characters, and independent assessor summaries under 700 characters. "
                    "When a repair quote or a third-party property damage demand is filed, include a separate independent_assessor_report. "
                    "Use a fictional assessor independent of the claimant and repairer. State what was inspected, using a physical inspection, "
                    "photographs or document review as appropriate. Identify the damaged vehicle by its registration or describe the damaged "
                    "property if there is no registration. Explain which proposed work matches the damage and whether the quantities, parts and "
                    "labour rates are reasonable. Give the assessed amount only if supported, and state any unverified work or costs. "
                    "This report is evidence about damage and costs and must not set a policy decision or add a charge to the claim. "
                    "Omit property assessment reports for injury-only scenarios. For theft, follow include_assessor_report when supplied. "
                    "When no assignment is supplied, vary whether theft claims have an independent_assessor_report. "
                    "A filed theft assessment is a separate independent review of the vehicle's pre-loss market value. "
                    "For an unrecovered vehicle, set inspection_basis to document_review or photographs. "
                    "Identify the vehicle and the evidence the assessor reviewed, explain the basis and limitations of the valuation, "
                    "and give an assessed amount only when the evidence supports one. The report adds no charge to the claim. "
                    "Supporting document summaries should reflect the issuer's records. For a lost-income claim, include work_incapacity_evidence "
                    "or earnings_evidence only if that evidence is filed, with the relevant days unable to work or the rate and basis of earnings. "
                    "List separate costs such as towing outside the main quote. Normally reconcile them to the claim total. Some cases may have missing documents or contradictory accounts. "
                    "Do not disclose test labels or suspicions in the source text.\n\n"
                    "Set medical_intake_filed only when a medical intake record accompanies the claim. Injuries can be reported without a medical record. "
                    "Supply facility, intake date and diagnostic codes only when available in the scenario. "
                    "Set adjuster_note_filed only when the file includes a separate adjuster entry. "
                    "Use fictional parties. For a filed repair estimate, use its current quotation total. "
                    "Set incident_details.initial_repair_estimate_lkr only when an earlier quotation had a different total.\n\n"
                    f"Recent synthetic accounts to avoid repeating: {json.dumps(incidents[-24:], ensure_ascii=False)}"
                )
                correction = ""
                for attempt in range(2):
                    try:
                        cases = request_batch(client, model, prompt + correction, GENERATION_TOOL, schema, size)
                    except GenerationValidationError as error:
                        if attempt:
                            raise
                        correction = f"\n\n{error} Return the complete batch with the schema error corrected."
                        continue
                    issues = []
                    if subjects is not None:
                        for index, (case, subject, slot) in enumerate(zip(cases, subjects, slots, strict=True), 1):
                            if case.scenario_type != slot["scenario_type"] or case.claim.loss_type != slot["loss_type"]:
                                issues.append(f"Case {index} must copy its assigned scenario_type and loss_type.")
                            assigned = subject.get("loss_date")
                            if assigned and (case.claim.loss_date is None or case.claim.loss_date.isoformat() != assigned):
                                issues.append(f"Case {index} must use the assigned loss_date {assigned} and keep its account and records consistent.")
                            band = slot["amount_band"]
                            if claim_amount_band(case.claim.gross_amount) != band:
                                direction = "below" if band == BELOW_THRESHOLD else "above"
                                issues.append(f"Case {index} must have a gross_amount strictly {direction} LKR {AUTHORITY:,.0f}. "
                                              "Adjust the incident severity and any cost records to fit the claimed amount.")
                            document_types = {document.type for document in case.claim.documents}
                            has_assessor_report = "independent_assessor_report" in document_types
                            if (PROPERTY_COST_TYPES.get(case.claim.loss_type) in document_types
                                    and not has_assessor_report):
                                issues.append(f"Case {index} must include an independent_assessor_report for the filed repair quote "
                                              "or property damage demand.")
                            if (include_assessor := slot.get("include_assessor_report")) is not None and include_assessor != has_assessor_report:
                                action = "include" if include_assessor else "omit"
                                issues.append(f"Case {index} must {action} the independent_assessor_report as assigned.")
                    if not issues:
                        break
                    if attempt:
                        raise GenerationError("The generated claims did not match their assigned incidents. " + issues[0])
                    correction = "\n\nReturn the complete batch in order. " + " ".join(issues)
                for index_in_batch, case in enumerate(cases):
                    draft = case.claim
                    if draft.narrative in incidents:
                        raise GenerationError("The generator repeated a claim narrative. Try generating again.")
                    incidents.append(draft.narrative)
                    identity = subjects[index_in_batch] if subjects is not None else {}
                    claim, labels = render_submission(draft, draft.documents, identity, noise_level)
                    fixtures.append({"claim": claim, "scenario": case.scenario, "scenario_type": case.scenario_type,
                                     "ground_truth_labels": labels})
        span.summary = {"status": "complete", "output_count": len(fixtures)}
        result = {"batch_id": batch_id, "model": model, "fixtures": fixtures}
        span.output = result
        return result


def commit_batch(store, labels_path, preview):
    """Store original submissions and ground truth separately for later processing."""
    with store.session.begin() as session:
        begin_write(session)
        return _save_batch(store, session, labels_path, preview)


def commit_pending_preview(store, labels_path, batch_id):
    with store.session.begin() as session:
        begin_write(session)
        pending = session.get(SyntheticClaimPreview, batch_id)
        if pending is not None:
            result = _save_batch(store, session, labels_path, pending.payload)
            session.delete(pending)
            return result
        ids = session.scalars(select(Assessment.id).join(Claim, Claim.id == Assessment.claim_id).where(
            Claim.batch_id == batch_id,
            select(WorkflowEvent.id).where(WorkflowEvent.assessment_id == Assessment.id,
                                          WorkflowEvent.kind == "synthetic_saved").exists(),
        ).order_by(Claim.sequence)).all()
        return _batch_result(batch_id, ids) if ids else None


def _batch_result(batch_id, ids):
    return {"batch_id": batch_id, "count": len(ids), "assessment_ids": ids, "status": "awaiting_policy"}


def _save_batch(store, session, labels_path, preview):
    batch_id = preview["batch_id"]
    ids, labels = [], []
    offset = session.scalar(select(func.max(Claim.sequence))) or 0
    execution = {**execution_manifest(), "workflow": WORKFLOW, "generation_model": preview["model"]}
    for sequence, fixture in enumerate(preview["fixtures"], offset + 1):
        claim = fixture["claim"]
        # Reuse the label key if a retry follows a labels commit and claims rollback.
        identifier = str(uuid.uuid5(uuid.NAMESPACE_OID, claim["claim_id"]))
        session.add(Claim(id=claim["claim_id"], sequence=sequence, batch_id=batch_id, policy_id=claim.get("policy_id") or "",
                          payload=claim, payload_hash=digest(claim)))
        session.flush()
        session.add(Assessment(id=identifier, claim_id=claim["claim_id"], status="awaiting_policy",
            execution={**execution, "generation_scenario": fixture["scenario_type"]},
            context={"findings": {}, "skipped": {}, "environment": "synthetic", "workflow": WORKFLOW, "version": 0}))
        store.add_event(session, identifier, "synthetic_saved", {"status": "awaiting_policy"})
        ids.append(identifier)
        truth = fixture["ground_truth_labels"]
        expected = SCENARIO_ROUTES[truth["route"]].value if "route" in truth else None
        if store.database_url:
            session.add(EvaluationLabel(assessment_id=identifier, batch_id=batch_id, expected=expected,
                                        scenario=fixture["scenario"], seed=truth.get("scenario_seed"), ground_truth=truth))
        else:
            labels.append((identifier, batch_id, expected, fixture["scenario"], truth.get("scenario_seed"),
                           json.dumps(truth, ensure_ascii=False)))
        if "policy_schedule" in fixture:
            save_artifacts(session, claim, fixture["policy_schedule"])
    session.flush()
    if store.database_url:
        return _batch_result(batch_id, ids)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(labels_path)) as label_db, label_db:
        label_db.execute("CREATE TABLE IF NOT EXISTS labels (assessment_id TEXT PRIMARY KEY,batch_id TEXT,expected TEXT,scenario TEXT,seed INTEGER)")
        if "ground_truth" not in {column[1] for column in label_db.execute("PRAGMA table_info(labels)")}:
            label_db.execute("ALTER TABLE labels ADD COLUMN ground_truth TEXT")
        label_db.executemany(
            "INSERT INTO labels (assessment_id,batch_id,expected,scenario,seed,ground_truth) VALUES(?,?,?,?,?,?) "
            "ON CONFLICT(assessment_id) DO UPDATE SET expected=excluded.expected, scenario=excluded.scenario, "
            "seed=excluded.seed, ground_truth=excluded.ground_truth", labels)
    return _batch_result(batch_id, ids)


def save_artifacts(session, claim, schedule):
    claim_id = claim["claim_id"]
    session.add(SyntheticClaimSchedule(claim_id=claim_id, schedule_id=schedule["id"], payload=schedule))
    inputs = claim["unstructured_inputs"]
    for source in DOCUMENT_SOURCES:
        if source not in inputs:
            continue
        documents = (doc for doc in claim["documents"] if SOURCE_BY_DOCUMENT_TYPE[doc["type"]] == source)
        for document, rendered in zip(documents, inputs[source].split(BLOCK_SEPARATOR), strict=True):
            session.add(SyntheticClaimAttachment(id=document["id"], claim_id=claim_id, payload={
                "fields": document, "text": rendered,
            }))
