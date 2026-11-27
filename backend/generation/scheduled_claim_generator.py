"""Generate a claim and its supporting records against a saved motor policy schedule."""

import json
import random
import re
import secrets
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal, ROUND_DOWN, ROUND_UP

from pydantic import Field
from sqlalchemy import select

from backend.agents.fraud_specialist import scheduled_cost_indicators
from backend.agents.intake_specialist import missing_cost_fields
from backend.assurance.investigation_tracing import operation_trace
from backend.claim_schemas import AUTHORITY, MOTOR_LOSSES, SCENARIO_ROUTES, Route, ScenarioRoute
from backend.evidence.motor_benefits import (
    BENEFIT_NAMES, COMMERCIAL_EXCLUSION, GLASS_WAIVER, MotorBenefit, commercial_use_exclusion, fixed_motor_benefit,
    mandatory_police_threshold, motor_benefit_excess, police_thresholds, quoted_terms, towing_excess_unresolved,
)
from backend.evidence.policy_schedule import fixed_repair_terms, scheduled_financials
from backend.evidence.required_claim_evidence import COST_DOCUMENT_TYPES, cost_line_total, has_police_report
from backend.evidence.synthetic_cost_history import prior_cost_documents
from backend.generation.claude_generation_client import (
    GeneratedData, GenerationError, GenerationValidationError, generation_client, local_date, request_tool,
)
from backend.generation.scheduled_claim_wording import linked_wording, normalized
from backend.generation.synthetic_claim_generator import (
    INCIDENT_PROFILES, ClaimForm, FiledDocument, IndependentAssessorReport, LineItem, MotorBenefitReceipt, PoliceReport, render_submission,
)
from backend.storage.sqlite_store import PolicySchedule
from backend.workflow.versions import V3, selected_pdf_workflow


CLAIM_TOOL = "submit_scheduled_claim"
ATTACHMENT_TOOL = "submit_claim_attachments"
BACKDATED = "cost_document_before_inception"
UNLISTED = "attachment_registration_mismatch"
INFLATED = "severity_inflation_and_phantom_items"
DISGUISED = "excluded_commercial_use_disguised"
REUSED = "duplicated_cost_document"
SUBLIMIT = "sublimit_or_depreciation_review"
DAMAGE_BOUNDARY = "prior_damage_boundary"
NEAR_LIMIT = "high_severity_near_policy_limit"
CONDITIONAL_DRIVER = "conditional_named_driver"
SHARED_LIABILITY = "shared_third_party_liability"
ITEMS_MISSING = "missing_cost_itemisation"
POLICE_MISSING = "missing_mandatory_police_report"
COST_REGISTRATION_MISSING = "missing_cost_document_registration"
CLERICAL = "clerical_amount_mismatch"
PAYMENT_MISSING = "missing_proof_of_payment"
AUTO_REPAIR = "straightforward_repair"
AUTO_MEDICAL = "covered_motor_outpatient_treatment"
AUTO_EQUIPMENT = "single_scheduled_equipment_loss"
AUTO_TOWING = "standard_towing_reimbursement"
AUTO_GLASS = "zero_excess_glass_repair"
SCENARIO_VARIANTS = {
    "auto_approval": (AUTO_REPAIR, AUTO_MEDICAL, AUTO_EQUIPMENT, AUTO_TOWING, AUTO_GLASS),
    "adjuster_review": (SHARED_LIABILITY, SUBLIMIT, DAMAGE_BOUNDARY, NEAR_LIMIT, CONDITIONAL_DRIVER),
    "need_information": (ITEMS_MISSING, POLICE_MISSING, COST_REGISTRATION_MISSING, CLERICAL, PAYMENT_MISSING),
    "fraud_escalation": (BACKDATED, UNLISTED, INFLATED, DISGUISED, REUSED),
}
MOTOR_BENEFIT_VARIANTS = {
    AUTO_MEDICAL: "motor_medical", AUTO_EQUIPMENT: "scheduled_equipment", AUTO_TOWING: "roadside_towing",
    AUTO_GLASS: "glass_waiver", PAYMENT_MISSING: "roadside_towing",
}
OWN_VEHICLE_LOSSES = {"repair", "theft"}
CONDITIONAL_COST = re.compile(r"\b(?:depreciat\w*|sub[ -]?limit|market value|wear and tear)\b", re.I)
COVERED_DRIVER = re.compile(r"\b(?:licen[sc]\w*|permission|consent|named|authori[sz]\w*|entitled to drive)\b", re.I)
NAMED_DRIVER_ONLY = re.compile(r"\b(?:only (?:the )?named drivers?|named drivers? only)\b", re.I)
OCCASIONAL_DRIVERS = ("Nimal Perera", "Malini Fernando", "Dinesh Silva", "Isuri Jayasinghe")
SPECIFIC_PROFILES = {
    AUTO_EQUIPMENT: ("theft_of_fitted_equipment", "Only one covered part fitted to the parked vehicle is stolen. The vehicle itself remains at the scene."),
    AUTO_MEDICAL: ("motor_outpatient_treatment", "An insured driver sustains a minor injury directly in an accident involving the insured car and receives same-day outpatient treatment."),
    AUTO_TOWING: ("accident_tow", "An accidental collision disables the insured vehicle. It is towed from the incident site to the nearest approved repairer."),
    PAYMENT_MISSING: ("accident_tow", "An accidental collision disables the insured vehicle. It is towed from the incident site to the nearest approved repairer."),
    AUTO_GLASS: ("glass_only_damage", "Debris breaks only the windscreen. There is no damage to the vehicle body or any other component."),
    INFLATED: ("minor_parking_scrape", "A minor low-speed parking scrape only scuffs one exterior panel. Airbags did not deploy and the car was driven normally afterward."),
    SHARED_LIABILITY: ("third_party_vehicle_damage", INCIDENT_PROFILES["third_party_vehicle_damage"][1]),
    NEAR_LIMIT: ("major_collision", "A serious collision causes extensive damage to the insured vehicle. Repair or total loss requires a substantial valuation."),
}
TOW_LOCATIONS = ("Galle", "Matara", "Colombo", "Negombo", "Kurunegala", "Kandy")
CENT = Decimal("0.01")
LABELS = re.compile(r"\b(?:" + "|".join(
    label.replace("_", r"[\s_.-]*") for label in (
        *SCENARIO_ROUTES, *(route.value.replace("-", "_").replace(" ", "_") for route in Route),
        *(variant for variants in SCENARIO_VARIANTS.values() for variant in variants),
        "ground_truth", "scenario_seed", "scenario_variant", "fraud_pattern", "target_route",
    )
) + r")\b", re.IGNORECASE)
CALENDAR_DATE = re.compile(
    r"\b(?:\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}|"
    r"\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2}(?:st|nd|rd|th)?)\b", re.IGNORECASE,
)


class DraftClaim(ClaimForm):
    loss_date: date | None = None
    insured_name: str = Field(min_length=2, max_length=120)
    registration: str = Field(min_length=1, max_length=40)
    policy_id: str = Field(min_length=1, max_length=64)


class ScheduledClaim(DraftClaim):
    motor_benefit: MotorBenefit | None = Field(default=None, exclude_if=lambda value: value is None)
    reported_vehicle_use: str | None = Field(default=None, exclude_if=lambda value: value is None)
    occasional_driver: str | None = Field(default=None, min_length=2, max_length=120, exclude_if=lambda value: value is None)


class ScheduledDraft(GeneratedData):
    scenario: str = Field(min_length=1, max_length=120, description="Brief factual title of the incident.")
    claim: DraftClaim


class RenderedAttachments(GeneratedData):
    documents: list[FiledDocument] = Field(min_length=3, max_length=4)


@dataclass(frozen=True)
class ScheduledScenario:
    schedule: dict
    benefit: dict
    route: ScenarioRoute
    seed: int
    profile: str
    variant: str
    loss_date: date | None
    first_report_day: date
    last_report_day: date
    minimum_amount: Decimal
    maximum_amount: Decimal
    document_registration: str
    reused_document: dict | None = None
    wording_clauses: tuple[dict, ...] = ()

    @property
    def motor_benefit(self):
        return MOTOR_BENEFIT_VARIANTS.get(self.variant)

    @property
    def loss_type(self):
        return ("theft" if self.variant == AUTO_EQUIPMENT else "repair") if self.motor_benefit else self.benefit["loss_type"]

    @property
    def tow_location(self):
        return TOW_LOCATIONS[self.seed % len(TOW_LOCATIONS)] if self.motor_benefit == "roadside_towing" else None

    @property
    def occasional_driver(self):
        if self.variant != CONDITIONAL_DRIVER:
            return None
        insured = self.schedule["insured_name"].casefold()
        named = [name for name in self.schedule.get("named_drivers", []) if name.casefold() != insured]
        names = named or [name for name in OCCASIONAL_DRIVERS if name.casefold() != insured]
        return names[self.seed % len(names)]


def amount_range(schedule, benefit, variant):
    basis = benefit["limit_basis"]
    limit = benefit.get("limit_lkr")
    loss_type = benefit["loss_type"]
    if basis in {"excluded", "unconfirmed", "unspecified"}:
        return None
    if variant == AUTO_REPAIR and fixed_repair_terms(benefit) is None:
        return None
    if variant == UNLISTED and loss_type not in OWN_VEHICLE_LOSSES:
        return None
    if variant in {SUBLIMIT, DAMAGE_BOUNDARY, NEAR_LIMIT, CONDITIONAL_DRIVER, INFLATED, DISGUISED, CLERICAL} and loss_type != "repair":
        return None
    if variant == SHARED_LIABILITY and loss_type != "third_party_property_damage":
        return None
    excess_fields = (benefit.get("compulsory_excess_lkr"), benefit.get("voluntary_excess_lkr"))
    excess = sum(Decimal(amount or 0) for amount in excess_fields)
    limit = Decimal(limit) if limit is not None else None
    if variant == SUBLIMIT and not CONDITIONAL_COST.search(quoted_terms(benefit)):
        return None
    if variant == CONDITIONAL_DRIVER:
        drivers = schedule.get("authorised_drivers", "")
        if not COVERED_DRIVER.search(drivers) or (NAMED_DRIVER_ONLY.search(drivers) and
                all(name.casefold() == schedule["insured_name"].casefold() for name in schedule.get("named_drivers", []))):
            return None
    if variant == DISGUISED and not commercial_use_exclusion(schedule):
        return None
    if variant == NEAR_LIMIT:
        if limit is None or limit <= AUTHORITY:
            return None
        minimum = max(excess + CENT, AUTHORITY + CENT, (limit * Decimal("0.85")).quantize(CENT, rounding=ROUND_UP))
        maximum = (limit * Decimal("0.98")).quantize(CENT, rounding=ROUND_DOWN)
        return (minimum, maximum) if minimum <= maximum else None
    minimum = excess + CENT
    maximum = AUTHORITY / 2
    if variant == INFLATED:
        minimum = max(minimum, AUTHORITY * Decimal("0.85"))
        maximum = AUTHORITY * Decimal("0.95")
    elif loss_type == "theft":
        minimum = max(minimum, AUTHORITY)
        maximum = Decimal("10000000")
    if variant == CLERICAL:
        minimum = max(minimum, Decimal("1000.00"))
    if variant == POLICE_MISSING:
        threshold = mandatory_police_threshold(schedule, benefit)
        if threshold is None:
            return None
        minimum = max(minimum, threshold + CENT)
        maximum = max(maximum, minimum * Decimal("1.15"))
    if limit is not None:
        maximum = min(maximum, limit)
    maximum = maximum.quantize(CENT, rounding=ROUND_DOWN)
    return (minimum, maximum) if minimum <= maximum else None


def motor_benefit_range(schedule, benefit, variant):
    limit = Decimal(benefit["limit_lkr"])
    excess = motor_benefit_excess(schedule, benefit, MOTOR_BENEFIT_VARIANTS[variant])
    if excess is None:
        return None
    if variant in {AUTO_TOWING, PAYMENT_MISSING}:
        return (limit, limit) if excess < limit <= AUTHORITY else None
    maximum = min(AUTHORITY / 2, limit).quantize(CENT, rounding=ROUND_DOWN)
    minimum = max(excess + CENT, (maximum / 2).quantize(CENT, rounding=ROUND_DOWN))
    return (minimum, maximum) if minimum <= maximum else None


def scenario_wording(wording, schedule, benefit, variant):
    if not (cover := wording.benefit_clause(benefit)):
        return ()
    quote = normalized(cover["quote"])
    service = MOTOR_BENEFIT_VARIANTS.get(variant)
    if service and not re.search(BENEFIT_NAMES[service], quote, re.I):
        return ()
    if variant == AUTO_GLASS and not GLASS_WAIVER.search(quote):
        return ()
    if variant == SUBLIMIT and not CONDITIONAL_COST.search(quote):
        return ()
    condition = None
    if variant == DISGUISED:
        condition = COMMERCIAL_EXCLUSION.search
    elif variant == CONDITIONAL_DRIVER:
        def condition(text):
            return COVERED_DRIVER.search(text) and re.search(r"\bdriv(?:e|er|ers|en|ing)\b", text, re.I)
    elif variant == POLICE_MISSING:
        threshold = mandatory_police_threshold(schedule, benefit)

        def condition(text):
            return threshold in police_thresholds(text)

    clauses = [cover]
    if condition:
        required = wording.find_clause(condition)
        if not required:
            return ()
        clauses.append(required)
    for saved in schedule.get("wording_references", []):
        if len(clauses) == 4:
            break
        if (reference := wording.citation(saved)) and reference not in clauses:
            clauses.append(reference)
    return tuple(clauses)


def plan_scheduled_claims(store, count):
    with store.session() as session:
        rows = session.scalars(select(PolicySchedule).order_by(PolicySchedule.id)).all()
        if not rows:
            raise ValueError("Generate a policy schedule before generating claims.")
        documents = {}
        schedules = []
        for row in rows:
            if row.document_id not in documents:
                documents[row.document_id] = linked_wording(session, row.document_id)
            wording = documents[row.document_id]
            if wording and wording.matches(row.payload):
                schedules.append((row.payload, wording))
        registered = {row.payload["vehicle"]["registration"] for row in rows}
        history = [(claim.payload, document) for claim, document in prior_cost_documents(session, sorted(MOTOR_LOSSES))]
    if not schedules:
        raise ValueError("No saved policy schedule matches an available indexed wording version.")
    today = local_date()
    candidates = {}
    for schedule, wording in schedules:
        start = date.fromisoformat(schedule["starts_on"])
        end = min(date.fromisoformat(schedule["ends_on"]), today)
        if start > end:
            continue
        provisional = wording.metadata["document_kind"] == "product_summary"
        for route, variants in SCENARIO_VARIANTS.items():
            if route == "auto_approval" and (provisional or selected_pdf_workflow() != V3):
                continue
            if route == "need_information" and provisional:
                continue
            for variant in variants:
                if service := MOTOR_BENEFIT_VARIANTS.get(variant):
                    benefit = fixed_motor_benefit(schedule, service)
                    if benefit and service == "roadside_towing" and towing_excess_unresolved(schedule, benefit):
                        continue
                    if (benefit and (bounds := motor_benefit_range(schedule, benefit, variant))
                            and (clauses := scenario_wording(wording, schedule, benefit, variant))):
                        candidates.setdefault(route, {}).setdefault(variant, []).append((schedule, benefit, bounds, clauses, None))
                    continue
                for benefit in schedule["cover"]:
                    if benefit["loss_type"] not in MOTOR_LOSSES or not (bounds := amount_range(schedule, benefit, variant)):
                        continue
                    clauses = scenario_wording(wording, schedule, benefit, variant)
                    if not clauses:
                        continue
                    pool = candidates.setdefault(route, {})
                    if variant != REUSED:
                        pool.setdefault(variant, []).append((schedule, benefit, bounds, clauses, None))
                        continue
                    for prior, document in history:
                        if (prior["loss_type"] != benefit["loss_type"] or prior.get("motor_benefit") or not prior.get("loss_date")
                                or not prior.get("incident_id") or prior.get("policy_id") != schedule["policy_number"]
                                or prior["registration"] != schedule["vehicle"]["registration"]
                                or not document.get("date")):
                            continue
                        incident_day = date.fromisoformat(max(prior["loss_date"], document["date"])) + timedelta(days=2)
                        amount = Decimal(document["total"])
                        if (start <= incident_day <= end and start <= date.fromisoformat(document["date"]) <= today
                                and bounds[0] <= amount <= bounds[1] and cost_line_total(document["line_items"]) == amount):
                            reused = {**document, "document_reference": document["id"]}
                            pool.setdefault(REUSED, []).append((schedule, benefit, (amount, amount), clauses, (incident_day, reused)))
    candidates = {route: variants for route, variants in candidates.items() if variants}
    if not candidates:
        raise ValueError("No started policy schedule has a covered motor benefit supported by its linked wording and suitable limits.")

    batch_rng = random.Random(secrets.randbits(53))
    routes = list(candidates)
    batch_rng.shuffle(routes)
    assignments = [routes[index % len(routes)] for index in range(count)]
    batch_rng.shuffle(assignments)
    remaining = {route: [] for route in routes}
    plans = []
    for route in assignments:
        case_seed = batch_rng.getrandbits(53)
        rng = random.Random(case_seed)
        if not remaining[route]:
            remaining[route].extend(candidates[route])
        variant = remaining[route].pop(rng.randrange(len(remaining[route])))
        schedule, benefit, (minimum, maximum), clauses, reused = rng.choice(candidates[route][variant])
        if variant in SPECIFIC_PROFILES:
            profile = SPECIFIC_PROFILES[variant][0]
        else:
            profile = rng.choice([profile for profile, (loss, _) in INCIDENT_PROFILES.items() if loss == benefit["loss_type"]])
        start, end = (date.fromisoformat(schedule[field]) for field in ("starts_on", "ends_on"))
        last_covered_day = min(end, today)
        first_covered_day = max(start, last_covered_day - timedelta(days=365))
        incident_day = first_covered_day + timedelta(days=rng.randint(0, (last_covered_day - first_covered_day).days))
        registration = schedule["vehicle"]["registration"]
        if variant == BACKDATED:
            incident_day = start + timedelta(days=rng.randint(0, min(12, (last_covered_day - start).days)))
        elif variant == UNLISTED:
            prefix, _, _ = registration.rpartition("-")
            available = [candidate for suffix in range(10_000) if (candidate := f"{prefix}-{suffix:04d}") not in registered]
            if not available:
                raise ValueError("No unused synthetic registration is available for the assigned vehicle prefix.")
            registration = rng.choice(available)
        if reused:
            incident_day, document = reused
        plans.append(ScheduledScenario(schedule, benefit, route, case_seed, profile, variant,
                                       incident_day, incident_day, min(today, incident_day + timedelta(days=7)),
                                       minimum, maximum, registration, document if reused else None, clauses))
    return plans


def request_validated(client, model, prompt, tool, schema, validate):
    correction = ""
    for _ in range(2):
        try:
            result = request_tool(client, model, prompt + correction, tool, schema)
        except GenerationValidationError as error:
            issue = str(error)
        else:
            issue = validate(result)
            if not issue:
                return result
        correction = "\n\nRegenerate the complete response. " + issue
    raise GenerationError(f"The generated record did not satisfy its assignment. {issue}")


def draft_claim(client, model, plan, recent):
    policy = plan.schedule
    route_instruction = {
        "auto_approval": "Use a straightforward covered incident, modest damage and clear responsibility.",
        "adjuster_review": "Provide all facts needed from the claimant. Any uncertainty should need an insurer decision, not a missing record. If the claim exceeds LKR 250000, use clear facts and make the damage consistent with that amount.",
        "need_information": "Describe the incident and damage the caller can report.",
        "fraud_escalation": "Describe the assigned event and date as reported by the caller. Keep the narrative factual and avoid adding any further inconsistencies.",
    }[plan.route]
    variant_instruction = {
        AUTO_REPAIR: "Describe a minor repair consistent with the amount. Exclude injury, third-party claims, and separate services.",
        AUTO_MEDICAL: "The policyholder was the unpaid driver. In the narrative, describe a minor acute injury and outpatient treatment on the accident date. Claim only the clinic expense. The expense receipt is a separate record. Leave medical_intake_filed false.",
        AUTO_EQUIPMENT: "Claim only the scheduled part that was stolen, and identify it as fitted to the insured vehicle. Leave the vehicle at the scene.",
        AUTO_TOWING: "Claim only the paid tow from the collision site to the nearest insurer-approved repairer. State the pickup location and destination.",
        PAYMENT_MISSING: "Claim only the tow from the collision site to the nearest insurer-approved repairer. State the pickup location and destination without making a statement about whether payment was made.",
        AUTO_GLASS: "Claim the glass replacement alone. Identify the broken windscreen and confirm the rest of the car was undamaged.",
        SUBLIMIT: "Describe damage to multiple distinct parts with plausible differences in age or original versus aftermarket specification. Give the facts an adjuster needs to apply the scheduled cap or depreciation.",
        DAMAGE_BOUNDARY: "Clearly disclose one existing defect and describe distinct new collision damage. Explain the areas affected so that the assessor can separate the two. Avoid attributing the older damage to this incident.",
        NEAR_LIMIT: "Describe major, well-documented damage. The reported amount should be credible for extensive structural or mechanical repairs that could approach total loss.",
        CONDITIONAL_DRIVER: "Use the assigned driver's full name exactly in the incident narrative. State the person's relationship, licence, vehicle use and permission. Use a plausible edge case under the authorised-driver clause and supply the relevant facts.",
        INFLATED: "Describe only a minor parking scrape. Set airbags_deployed to false and state that the vehicle was driven afterward without steering or structural damage.",
        DISGUISED: "The policyholder reports driving on a personal errand, without any paying passengers. State that account in the narrative.",
        REUSED: "Describe damage corresponding to the overlapping repair or cost descriptions in the assignment. The caller reports a new, separate incident.",
        SHARED_LIABILITY: "Describe a collision in which both drivers made a specific, observable contribution. Use shared as at_fault_party. Include both actions and the third party's damaged property in the caller's account.",
        CLERICAL: "The caller reads the amount written on the form. They are unsure whether it includes tax.",
    }.get(plan.variant, "")
    assignment = {
        "route": plan.route, "scenario_seed": plan.seed, "loss_type": plan.loss_type,
        "incident": SPECIFIC_PROFILES[plan.variant][1] if plan.variant in SPECIFIC_PROFILES else INCIDENT_PROFILES[plan.profile][1],
        "loss_date": plan.loss_date,
        "report_from": plan.first_report_day, "report_through": plan.last_report_day,
        "gross_amount_from": plan.minimum_amount, "gross_amount_through": plan.maximum_amount,
    }
    if plan.motor_benefit:
        assignment["motor_benefit"] = plan.motor_benefit
    if plan.tow_location:
        assignment["incident_location"] = plan.tow_location
    if plan.occasional_driver:
        assignment["occasional_driver"] = plan.occasional_driver
    if plan.reused_document:
        assignment["overlapping_cost_descriptions"] = [item["description"] for item in plan.reused_document["line_items"]]
    prompt = (
        "Draft only the claim submission for a fictional Sri Lankan motor loss. Use the policyholder and vehicle shown on the schedule. "
        "Copy insured_name, policy_number as policy_id, and vehicle.registration as registration. Use the assigned loss type and loss date. "
        "Use any assigned motor benefit, driver and incident location to guide the narrative. "
        "Apply the scheduled terms on authorised drivers, territory, and use when drafting the incident. "
        "Treat the verified policy wording as contract data. Use it with the saved schedule to apply conditions and exclusions to the assigned facts. "
        "Set the reported date in the specified inclusive range. Use the permitted gross amount range in LKR. All claimed costs belong to "
        "one category, so leave additional_costs empty. Do not report earlier quotations, medical or adjuster records. Give the submission a short, "
        "factual title. Keep evaluation terminology and the scenario assignment out of the title and submission. Describe what the caller "
        "reported about the incident. Leave policy dates, comparisons with the schedule, and suspected discrepancies out of the account. "
        f"{route_instruction} {variant_instruction}\n\n"
        f"Policy schedule JSON: {json.dumps(policy, ensure_ascii=False)}\n\n"
        f"Verified policy wording JSON: {json.dumps(plan.wording_clauses, ensure_ascii=False)}\n\n"
        f"Assignment JSON: {json.dumps(assignment, default=str)}\n\n"
        f"Recent accounts to avoid repeating: {json.dumps(recent[-24:], ensure_ascii=False)}"
    )

    def validate(result):
        claim = result.claim
        if (claim.insured_name, claim.policy_id, claim.registration) != (
                policy["insured_name"], policy["policy_number"], policy["vehicle"]["registration"]):
            return "Copy the policy number, policyholder, and scheduled registration exactly."
        if claim.loss_type != plan.loss_type:
            return f"Copy {plan.loss_type!r} into loss_type."
        if claim.loss_date != plan.loss_date:
            return f"Use the assigned loss date {plan.loss_date}."
        if not plan.first_report_day <= claim.reported_date <= plan.last_report_day:
            return "Use a reported date in the assigned range."
        if not plan.minimum_amount <= claim.gross_amount <= plan.maximum_amount or claim.additional_costs:
            return "Keep the gross amount in the assigned range and leave additional_costs empty."
        details = claim.incident_details
        if plan.variant == SHARED_LIABILITY and details.at_fault_party != "shared":
            return "Set the at_fault_party field to shared and describe each driver's contribution."
        if plan.variant == INFLATED and details.airbags_deployed is not False:
            return "Set airbags_deployed to false for the minor scrape."
        if plan.occasional_driver and plan.occasional_driver.casefold() not in claim.narrative.casefold():
            return f"Include the assigned full driver name {plan.occasional_driver!r} in the narrative."
        if plan.tow_location and plan.tow_location.casefold() not in claim.narrative.casefold():
            return "Name the assigned pickup location in the incident narrative."
        if details.medical_intake_filed or details.adjuster_note_filed or details.initial_repair_estimate_lkr:
            return "Leave medical_intake_filed and adjuster_note_filed false. Omit initial_repair_estimate_lkr."
        if details.medical_intake_date and (plan.variant != AUTO_MEDICAL or details.medical_intake_date != claim.loss_date):
            return "Omit medical_intake_date unless this is the assigned same-day outpatient visit."
        if claim.narrative in recent:
            return "Use a distinct incident narrative."
        if LABELS.search(result.model_dump_json()):
            return "Keep evaluation terminology out of the title and submission."
        if claim.loss_date is None and any(CALENDAR_DATE.search(value) for value in (
                result.scenario, claim.narrative, details.model_dump_json(exclude_none=True))):
            return "Leave exact calendar dates out of the incident title and narrative."
        return None

    draft = request_validated(client, model, prompt, CLAIM_TOOL, ScheduledDraft, validate)
    claim = ScheduledClaim(**draft.claim.model_dump(), motor_benefit=plan.motor_benefit,
                           reported_vehicle_use="private" if plan.variant == DISGUISED else None,
                           occasional_driver=plan.occasional_driver)
    return draft.scenario, claim


def render_attachments(client, model, claim, plan):
    policy = plan.schedule
    registration = claim.registration
    police_fields = {"incident_date", "incident_location", "commercial_trip_verified"}
    receipt_fields = {"beneficiary", "service_code", "asset_serial", "part_serial"}
    earliest = claim.loss_date or claim.reported_date
    latest = min(local_date(), claim.reported_date + timedelta(days=7))
    cost_deadline = date.fromisoformat(policy["starts_on"]) - timedelta(days=1) if plan.variant == BACKDATED else latest
    witness_only = claim.loss_type == "third_party_injury" or plan.motor_benefit in {"motor_medical", "roadside_towing"}
    cost_type = "motor_benefit_receipt" if plan.motor_benefit else COST_DOCUMENT_TYPES[claim.loss_type][0]
    slots = [
        {"type": cost_type, "registration": plan.document_registration,
         "issued_from": min(earliest, cost_deadline), "issued_through": cost_deadline},
        {"type": "police_report", "registration": registration, "issued_from": earliest, "issued_through": latest},
        {"type": "witness_statement" if witness_only else "independent_assessor_report",
         "registration": registration, "issued_from": earliest, "issued_through": latest},
    ]
    if plan.variant in {AUTO_REPAIR, AUTO_GLASS}:
        slots[2]["assessed_amount_lkr"] = claim.gross_amount
        slots[2]["inspection_basis"] = "physical_inspection"
    if plan.variant == BACKDATED:
        slots[1]["incident_date"] = cost_deadline
    if plan.variant == DISGUISED:
        slots[1]["commercial_trip_verified"] = True
    if plan.variant == DAMAGE_BOUNDARY:
        slots[2]["prior_damage_recorded_on"] = claim.loss_date - timedelta(days=30)
        slots[2]["inspection_basis"] = "physical_inspection"
    if plan.variant == SHARED_LIABILITY:
        slots.append({"type": "witness_statement", "registration": registration, "issued_from": earliest, "issued_through": latest})
    if plan.motor_benefit:
        slots[0].update(service=plan.motor_benefit, payment_status="pro_forma" if plan.variant == PAYMENT_MISSING else "paid")
    if plan.variant == AUTO_MEDICAL:
        slots[0].update(beneficiary=policy["insured_name"], service_code="OUTPATIENT-01", issued_through=earliest)
    if plan.tow_location:
        slots[0].update(service_location=plan.tow_location, issued_through=earliest)
        slots[1].update(incident_location=plan.tow_location, incident_date=earliest)
    if plan.variant == AUTO_EQUIPMENT:
        asset_serial = policy["vehicle"]["chassis_number"]
        part_serial = f"SYN-PART-{plan.seed % 1_000_000:06d}"
        slots[0].update(asset_serial=asset_serial, part_serial=part_serial)
        slots.append({"type": "purchase_receipt", "registration": registration, "asset_serial": asset_serial,
                      "part_serial": part_serial,
                      "issued_from": earliest - timedelta(days=120), "issued_through": earliest - timedelta(days=1)})
    context = {key: policy[key] for key in ("starts_on", "ends_on", "vehicle", "insured_name", "currency", "claims_instructions")}
    context["cover"] = plan.benefit
    submission = claim.model_dump_json(exclude_none=True, exclude={"incident_details"})
    cost_instruction = (
        "Record the vehicle's total pre-loss value as one valuation line. Explain any adjustments in the independent report. "
        if claim.loss_type == "theft" and not plan.motor_benefit else
        "Itemise the charges using positive amounts. Name each actual repair or service. Do not use miscellaneous or bundled charges. The amounts must add up exactly to the printed total. "
    )
    prompt = (
        "Prepare fictional Sri Lankan motor claim evidence as the respective garage, valuation preparer, third party, police or independent assessor. "
        "Produce one document per entry in Document fields, in the same order. Copy each document's assigned registration exactly. "
        "Use each document's assigned inclusive issue-date range, even when the ranges differ. "
        "Put issue dates, vehicle numbers and the itemised costs only in their structured fields for the cost document. "
        "Copy assigned optional fields exactly. Leave unassigned incident_date, incident_location, commercial_trip_verified and prior_damage_recorded_on fields blank. "
        "When a police incident_date is supplied, treat it as the date in the incident register. When a paid passenger booking is marked verified, describe the booking the police checked. "
        "When an earlier condition record is assigned, explain which damage it describes and which damage was caused by the later incident. "
        "Use the submission for incident and damage details. When no loss date is supplied, omit all calendar dates from filenames, summaries, and cost descriptions. Put the issue and inspection dates only in their respective structured fields. "
        "Set the printed total to the gross amount claimed. " + cost_instruction +
        "Keep each line item description under 120 characters. Give the specific work or service as a short phrase. "
        'Return every amount as a plain decimal such as "18500.00", without currency symbols, thousands separators, or calculations. '
        "For an estimate provide a fictional garage name and address. Limit police and witness summaries to one or two short sentences under 300 characters. If an assessor report is assigned, describe its basis, supported amount and limitations in under 650 characters. "
        "Copy the assessed amount and inspection basis exactly when supplied. For an unrecovered vehicle, use document review or photographs to assess its pre-loss value. "
        "For a motor benefit invoice, use the assigned service, status, and identifying fields. The provider and service location must identify the actual clinic or vendor in the fictional record. "
        "For a tow, include the destination repairer and the insurer assistance dispatch reference in the towing cost description. If a witness statement is assigned for the tow, use the dispatcher's account to confirm that dispatch reference and the insurer-approved destination. "
        "For a purchase receipt, describe the single fitted part and identify the vehicle it was fitted to. For a fourth document that is a witness statement, describe both drivers' actions and their positions in a simple scene sketch. "
        "Limit each record to the issuer's observations and assigned fields. Use ordinary filenames appropriate to the issuer and record.\n\n"
        f"Claim submission JSON: {submission}\n\n"
        f"Relevant schedule fields JSON: {json.dumps(context, ensure_ascii=False)}\n\n"
        f"Verified policy wording JSON: {json.dumps(plan.wording_clauses, ensure_ascii=False)}\n\n"
        f"Document fields JSON: {json.dumps(slots, default=str)}"
    )

    assigned_identity = police_fields | receipt_fields

    def validate(result):
        if len(result.documents) != len(slots):
            return "Provide one document for every entry in Document fields, in order."
        cost, _, supporting = result.documents[:3]
        for document, slot in zip(result.documents, slots, strict=True):
            if document.type != slot["type"] or document.registration != slot["registration"]:
                return "Use the assigned document types and registrations in the given order."
            if document.date is None or not slot["issued_from"] <= document.date <= slot["issued_through"]:
                return "Use each document's assigned issue-date range."
            if LABELS.search(document.model_dump_json()):
                return "Use ordinary document filenames and issuer language without process commentary."
            for field in slot.keys() - {"type", "registration", "issued_from", "issued_through"} - assigned_identity:
                if getattr(document, field, None) != slot[field]:
                    return f"Copy the assigned {field} field exactly for the {slot['type']}."
        if (cost.total != claim.gross_amount or not cost.line_items or any(item.amount <= 0 for item in cost.line_items)
                or sum(item.amount for item in cost.line_items) != cost.total):
            return "Itemise positive costs that sum to the printed total, which must equal the gross amount claimed."
        if plan.variant == SUBLIMIT and len({item.description.casefold() for item in cost.line_items}) < 2:
            return "Itemise at least two distinct damaged parts."
        if plan.variant == AUTO_GLASS and any(not re.search(r"\b(?:glass|windscreen|windshield)\b", item.description, re.I)
                                             for item in cost.line_items):
            return "Itemise windscreen glass and services on the windscreen glass only."
        if isinstance(supporting, IndependentAssessorReport):
            if not earliest <= supporting.inspection_date <= supporting.date:
                return "Put the inspection date between the earliest issue date and the date of the assessor report."
            if supporting.prior_damage_recorded_on != slots[2].get("prior_damage_recorded_on"):
                return "Provide an earlier condition record only when it is assigned."
            if claim.loss_type == "theft" and not plan.motor_benefit and supporting.inspection_basis == "physical_inspection":
                return "Assess the unrecovered vehicle using document review or photographs."
        if claim.loss_date is None and any(CALENDAR_DATE.search(value) for document in result.documents
                for value in (document.name, getattr(document, "summary", ""), *(item.description for item in getattr(document, "line_items", ())))):
            return "Leave exact calendar dates out of filenames, report summaries, and item descriptions."
        return None

    documents = request_validated(client, model, prompt, ATTACHMENT_TOOL, RenderedAttachments, validate).documents
    police = documents[1]
    if isinstance(police, PoliceReport):
        documents[1] = police.model_copy(update={field: slots[1].get(field) for field in police_fields})
    if isinstance(documents[0], MotorBenefitReceipt):
        documents[0] = documents[0].model_copy(update={field: slots[0].get(field) for field in receipt_fields})
    if plan.variant == AUTO_EQUIPMENT:
        proof = documents[-1]
        documents[-1] = proof.model_copy(update={field: slots[-1][field] for field in ("asset_serial", "part_serial")})
    cost = documents[0]
    if plan.variant == INFLATED:
        module = (claim.gross_amount * Decimal("0.30")).quantize(CENT, rounding=ROUND_DOWN)
        airbag = LineItem(description="Driver airbag module replacement", amount=module)
        items = [airbag, airbag, LineItem(description="Structural frame straightening", amount=claim.gross_amount - module * 2)]
        documents[0] = cost.model_copy(update={"line_items": items})
    elif plan.variant == CLERICAL:
        items = [item.model_copy(update={"amount": (item.amount * Decimal("0.995")).quantize(CENT, rounding=ROUND_DOWN)
                                         or CENT}) for item in cost.line_items]
        total = sum((item.amount for item in items), Decimal(0))
        documents[0] = cost.model_copy(update={"line_items": items, "total": total})
    elif plan.variant == POLICE_MISSING:
        return [cost, *documents[2:]]
    elif plan.reused_document is not None:
        documents[0] = plan.reused_document
    elif plan.variant == ITEMS_MISSING:
        documents[0] = cost.model_copy(update={"line_items": []})
    elif plan.variant == COST_REGISTRATION_MISSING:
        registration_text = r"[-\s]*".join(re.escape(part) for part in claim.registration.split("-"))
        pattern = re.compile(rf"(?<![A-Za-z0-9]){registration_text}(?![A-Za-z0-9])", re.I)
        items = [item.model_copy(update={"description": pattern.sub("[registration unreadable]", item.description)})
                 for item in cost.line_items]
        documents[0] = cost.model_copy(update={"registration": None, "line_items": items,
                                               "name": pattern.sub("[registration unreadable]", cost.name)})
    return documents


def discrepancies(claim, plan):
    def fact(artifact, field, value, artifact_id):
        return {"artifact": artifact, "artifact_id": artifact_id, "field": field, "value": value}

    def submission(field):
        return fact("claim_submission", f"/{field}", claim[field], claim["claim_id"])

    def schedule(field, value):
        return fact("policy_schedule", f"/{field}", value, plan.schedule["id"])

    def attachment(document, field):
        return fact("attachment", f"/fields/{field}", document.get(field), document["id"])

    cost = claim["documents"][0]
    match plan.variant:
        case "attachment_registration_mismatch":
            evidence = [schedule("vehicle/registration", plan.schedule["vehicle"]["registration"]), attachment(cost, "registration")]
        case "cost_document_before_inception":
            evidence = [schedule("starts_on", plan.schedule["starts_on"]), submission("loss_date"),
                        attachment(cost, "date"), attachment(claim["documents"][1], "incident_date")]
        case "severity_inflation_and_phantom_items":
            evidence = [submission("gross_amount"), attachment(cost, "total"), attachment(cost, "line_items")]
        case "excluded_commercial_use_disguised":
            field = "permitted_use" if commercial_use_exclusion(plan.schedule) == plan.schedule.get("permitted_use") else "special_conditions"
            evidence = [schedule(field, plan.schedule[field]), submission("reported_vehicle_use"),
                        attachment(claim["documents"][1], "commercial_trip_verified")]
        case "duplicated_cost_document":
            evidence = [attachment(cost, "document_reference"), attachment(cost, "line_items"), submission("loss_date")]
        case "missing_mandatory_police_report":
            evidence = [submission("gross_amount"), fact("claim_submission", "/documents", [doc["type"] for doc in claim["documents"]], claim["claim_id"])]
        case "missing_cost_itemisation":
            evidence = [attachment(cost, "total"), attachment(cost, "line_items")]
        case "missing_cost_document_registration":
            evidence = [submission("registration"), attachment(cost, "registration")]
        case "clerical_amount_mismatch":
            evidence = [submission("gross_amount"), attachment(cost, "total")]
        case "missing_proof_of_payment":
            evidence = [submission("motor_benefit"), attachment(cost, "payment_status")]
        case _:
            return []
    return [{"kind": plan.variant, "evidence": evidence}]


INFORMATION_FIELDS = {
    ITEMS_MISSING: "line_items", COST_REGISTRATION_MISSING: "cost_document.registration",
    CLERICAL: "additional_costs", PAYMENT_MISSING: "motor_benefit_receipt.payment_status",
}
FRAUD_INDICATORS = {
    BACKDATED: "police_incident_date_conflict", UNLISTED: "cost_document_registration_conflict",
    INFLATED: "duplicate_airbag_charges_conflict", DISGUISED: "excluded_use_conflicts_with_reported_use",
}


def routing_basis(claim, draft, plan):
    schedule, benefit = plan.schedule, plan.benefit
    if (not plan.wording_clauses or claim["policy_id"] != schedule["policy_number"]
            or claim["registration"] != schedule["vehicle"]["registration"]
            or not schedule["starts_on"] <= claim["loss_date"] <= schedule["ends_on"]):
        raise GenerationError("The generated claim cannot be linked to active coverage in the saved policy wording.")

    gross = Decimal(claim["gross_amount"])
    threshold = mandatory_police_threshold(schedule, benefit)
    report_missing = threshold is not None and gross > threshold and not has_police_report(claim)
    gaps = {item["field"]: item for item in missing_cost_fields(claim)}
    documents = claim["documents"]
    variant = plan.variant
    indicators = scheduled_cost_indicators(claim, schedule)
    reason, evidence = None, []

    if plan.route == "auto_approval":
        financial = scheduled_financials(claim, schedule)
        if financial and gross <= AUTHORITY and not (gaps or report_missing or indicators):
            reason = "The covered expense is documented and within the scheduled limit and automated authority after the applicable excess."
            evidence = ["submission:/gross_amount", "submission:/documents", "policy_schedule:/cover"]
    elif variant in INFORMATION_FIELDS and INFORMATION_FIELDS[variant] in gaps and not indicators:
        gap = gaps[INFORMATION_FIELDS[variant]]
        reason = gap["reason"]
        evidence = ["submission:/documents/0", "submission:/gross_amount"]
        if variant == CLERICAL:
            difference = gross - Decimal(documents[0]["total"])
            reason = f"The claim form exceeds the itemised invoice by LKR {difference:,.2f}. Confirm the tax and reconcile the totals."
    elif variant == POLICE_MISSING and report_missing and not indicators:
        reason = f"The policy requires a police report for claims above LKR {threshold:,.2f}. The submission is above that amount and contains no report."
        evidence = ["submission:/gross_amount", "submission:/documents"]
    elif variant in FRAUD_INDICATORS:
        indicator = next((item for item in indicators if item["kind"] == FRAUD_INDICATORS[variant]), None)
        if indicator:
            reason, evidence = indicator["reason"], indicator["evidence"]
    elif variant == REUSED and plan.reused_document:
        original = plan.reused_document
        if {**documents[0], "id": original["id"]} == original and original["date"] < claim["loss_date"]:
            reason = "The cost document is an exact copy from an earlier incident on the same policy."
            evidence = ["submission:/documents/0/document_reference", "submission:/loss_date", "policy_schedule:/policy_number"]
    elif plan.route == "adjuster_review" and not (gaps or report_missing or indicators):
        if variant == SHARED_LIABILITY and draft.incident_details.at_fault_party == "shared" and any(
                document["type"] == "witness_statement" for document in documents):
            reason = "The policy covers third-party property damage. The documented shared fault requires the insurer to apportion liability."
            evidence = ["submission:/narrative", "submission:/documents"]
        elif variant == SUBLIMIT and len(documents[0]["line_items"]) >= 2:
            reason = "The cited cost conditions require interpretation of the applicable sublimit or depreciation across the itemised repairs."
            evidence = ["submission:/documents/0/line_items", "policy_schedule:/cover"]
        elif variant == DAMAGE_BOUNDARY and any(
                (previous := document.get("prior_damage_recorded_on")) and previous < claim["loss_date"] for document in documents):
            reason = "The independent assessment distinguishes prior and new damage. An adjuster must identify the repair costs attributable to the covered incident."
            evidence = ["submission:/narrative", "submission:/documents"]
        elif variant == NEAR_LIMIT and (limit := Decimal(benefit["limit_lkr"])) and AUTHORITY < gross <= limit and gross >= limit * Decimal("0.85"):
            reason = f"The documented claim approaches the scheduled LKR {limit:,.2f} limit and exceeds automated authority."
            evidence = ["submission:/gross_amount", "submission:/documents", "policy_schedule:/cover"]
        elif variant == CONDITIONAL_DRIVER and claim.get("occasional_driver") == plan.occasional_driver:
            reason = "The insurer must verify how the cited driver authorisation conditions apply to the documented occasional driver."
            evidence = ["submission:/occasional_driver", "submission:/narrative", "policy_schedule:/authorised_drivers"]
    if not reason:
        raise GenerationError("The generated evidence does not support the assigned route under the verified policy wording.")
    return {"route": plan.route, "reason": reason, "document_id": schedule["wording"]["id"],
            "document_version": schedule["wording"]["version"], "document_hash": schedule["wording"]["file_hash"],
            "coverage": benefit.get("benefit_name") or plan.loss_type,
            "policy_refs": list(dict.fromkeys(clause["reference"] for clause in plan.wording_clauses)), "evidence": evidence}


def preview_scheduled_batch(plans, batch_id, *, discarded_previews=(), noise_level=0.2, actor_id=None):
    with operation_trace("generator.preview", batch_id, metadata={"batch_size": len(plans)}, actor_id=actor_id) as span:
        recent = list(discarded_previews)
        fixtures = []
        with generation_client() as (client, model):
            for plan in plans:
                scenario, draft = draft_claim(client, model, plan, recent)
                documents = render_attachments(client, model, draft, plan)
                vehicle = {key: plan.schedule["vehicle"][key] for key in ("make", "model", "year", "chassis_number")}
                claim, labels = render_submission(draft, documents, {"vehicle": vehicle}, noise_level)
                basis = routing_basis(claim, draft, plan)
                differences = discrepancies(claim, plan)
                fraud_pattern = plan.variant if plan.route == "fraud_escalation" else None
                flags = [*labels["fraud_siu_flags"], *([fraud_pattern] if fraud_pattern else [])]
                labels.update(route=basis["route"], scenario_seed=plan.seed, scenario_variant=plan.variant, fraud_pattern=fraud_pattern,
                              routing_basis=basis, policy_wording_clauses=list(plan.wording_clauses), discrepancies=differences,
                              fraud_siu_flags=flags, siu_flag_count=len(flags))
                fixtures.append({"claim": claim, "scenario": scenario, "scenario_type": plan.profile,
                                 "policy_schedule": plan.schedule, "ground_truth_labels": labels})
                recent.append(draft.narrative)
        span.summary = {"status": "complete", "output_count": len(fixtures)}
        return {"batch_id": batch_id, "model": model, "fixtures": fixtures}
