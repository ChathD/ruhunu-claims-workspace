"""Generate and save fictional people, vehicle policies and their linked wording."""
import base64
import uuid
from collections import Counter
from datetime import date
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator
from pydantic_core import PydanticCustomError
from sqlalchemy import select

from backend.storage.sqlite_store import Person, Policy, PolicyBinding, PolicyDocument, SyntheticPolicyHolder, digest
from backend.generation.claude_generation_client import GeneratedData, GenerationError, GenerationValidationError, generation_client, local_date, request_batch
from backend.claim_schemas import ChargeCategory, InsuranceType, Money, MotorLossType, THIRD_PARTY_LOSSES, VehicleEvidence
from backend.evidence.required_claim_evidence import DRUNK_DRIVING_EXCLUSION, coverage_document_checklist
from backend.generation.policy_copy_checks import words
from backend.evidence.policy_wording_index import GENERATED_WORDING_PRODUCT, document_record, generated_wording_matches


SupportingDocumentType = Literal["incident_report", "witness_statement"]
GENERATION_TOOL = "submit_people_and_policies"
POLICY_BATCH_SIZE = 3
TermItem = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=160)]
COMMON_ADDITIONAL_CHARGES = {"towing", "storage", "diagnostics", "temporary_transport"}
VehicleType = Literal["hatchback", "sedan", "suv", "pickup", "van", "motorcycle", "light_truck"]
PermittedUse = Literal["private", "business_travel", "goods_delivery", "passenger_hire", "mixed_private_business"]
RepairCause = Literal["collision", "flood", "storm", "fire", "vandalism", "falling_objects", "animal_impact"]
DISTRICTS = (
    "Colombo", "Gampaha", "Kalutara", "Kandy", "Matale", "Nuwara Eliya", "Galle", "Matara", "Hambantota",
    "Jaffna", "Kilinochchi", "Mannar", "Vavuniya", "Mullaitivu", "Batticaloa", "Ampara", "Trincomalee",
    "Kurunegala", "Puttalam", "Anuradhapura", "Polonnaruwa", "Badulla", "Monaragala", "Ratnapura", "Kegalle",
)
VEHICLE_PROFILES = (
    ("hatchback", "private", "flood", "Consider floodwater damage to cabin trim, wiring and electronic modules."),
    ("van", "goods_delivery", "fire", "Consider fire damage to the load area. Address carriage of goods separately."),
    ("suv", "mixed_private_business", "storm", "Consider storm damage to exterior fittings. Define any off-road restrictions."),
    ("motorcycle", "goods_delivery", "vandalism", "Consider damage to controls. Distinguish removable delivery gear."),
    ("sedan", "business_travel", "falling_objects", "Consider objects damaging the roof or windscreen. Address passenger hire."),
    ("pickup", "mixed_private_business", "animal_impact", "Consider damage to the grille or underride protection. Address cargo and commercial-use restrictions."),
    ("light_truck", "goods_delivery", "fire", "Consider accidental fire damage to cab wiring. Distinguish insured bodywork from cargo."),
    ("van", "passenger_hire", "storm", "Consider hail damage on passenger service routes. Define limits on the permitted passenger use."),
    ("sedan", "passenger_hire", "vandalism", "Consider damage to door locks. Address liability for carrying paying passengers."),
)


class PersonDraft(GeneratedData):
    full_name: str = Field(min_length=2, max_length=120)
    district: str = Field(min_length=2, max_length=80)
    locality: str = Field(min_length=2, max_length=120, description="Town or neighbourhood within the assigned district.")


class InsuredVehicle(VehicleEvidence):
    make: str = Field(min_length=1, max_length=60)
    model: str = Field(min_length=1, max_length=80)
    year: int = Field(ge=1886, le=2100)


class LabourTerms(GeneratedData):
    basis: Literal["no_separate_limit", "per_panel", "per_claim"]
    limit: Money | None

    @model_validator(mode="after")
    def validate_limit(self):
        if self.basis == "no_separate_limit" and self.limit is not None:
            raise ValueError("Labour without a separate limit must use null")
        if self.basis != "no_separate_limit" and (self.limit is None or self.limit <= 0):
            raise ValueError("A labour limit requires a positive amount")
        return self


class PaintingTerms(GeneratedData):
    scope: Literal["excluded", "affected_panels_only", "full_respray_if_required"]
    limit_per_panel: Money | None
    max_panels: int | None = Field(ge=1, le=30)

    @model_validator(mode="after")
    def validate_limits(self):
        if self.scope == "excluded" and (self.limit_per_panel is not None or self.max_panels is not None):
            raise ValueError("Excluded painting cannot have limits")
        if self.limit_per_panel is not None and self.limit_per_panel <= 0:
            raise ValueError("A painting limit must be positive")
        return self


class NamedAdditionalCharge(GeneratedData):
    category: ChargeCategory
    name: TermItem


class AdditionalChargeBenefit(NamedAdditionalCharge):
    conditions: str = Field(min_length=1, description="Conditions that determine whether the additional charge is eligible.")
    limit_per_incident: Money | None = Field(gt=0)
    receipt_required: bool


class AdditionalChargeExclusion(NamedAdditionalCharge):
    reason: str = Field(min_length=1, description="Explanation of why the additional charge is excluded.")


class AdditionalChargeTerms(GeneratedData):
    covered: bool
    eligible_charges: list[AdditionalChargeBenefit] = Field(max_length=8)
    exclusions: list[AdditionalChargeExclusion] = Field(max_length=12)

    @model_validator(mode="after")
    def validate_cover(self):
        if self.covered != bool(self.eligible_charges):
            raise ValueError("List eligible charges when additional charges are covered, or use an empty list when excluded")
        categories = [charge.category for charge in [*self.eligible_charges, *self.exclusions]]
        if len(set(categories)) != len(categories):
            raise ValueError("A charge category may appear only once across covered charges and exclusions")
        unstated = COMMON_ADDITIONAL_CHARGES.difference(categories)
        if unstated:
            raise ValueError(f"Explicitly cover or exclude these additional charges: {', '.join(sorted(unstated))}")
        return self


class RepairBenefit(GeneratedData):
    cause: RepairCause
    description: str = Field(min_length=1, max_length=500)
    covered_repairs: list[TermItem] = Field(min_length=1, max_length=8)
    conditions: list[TermItem] = Field(max_length=6)
    exclusions: list[TermItem] = Field(max_length=6)


class RepairTerms(GeneratedData):
    description: str = Field(min_length=10, max_length=1000)
    flood_damage: Literal["covered", "excluded"] = Field(description="Whether flooding is an insured cause of direct repair damage.")
    repair_benefits: list[RepairBenefit] = Field(min_length=1, max_length=7)
    exclusions: list[TermItem] = Field(min_length=1, max_length=12)
    labour: LabourTerms
    painting: PaintingTerms
    additional_charges: AdditionalChargeTerms

    @model_validator(mode="after")
    def validate_benefits(self):
        causes = {benefit.cause for benefit in self.repair_benefits}
        if len(causes) != len(self.repair_benefits) or not (causes - {"collision"}):
            raise PydanticCustomError("invalid_repair_benefits", "List distinct repair benefits including at least one cause beyond collision")
        if ("flood" in causes) != (self.flood_damage == "covered"):
            raise PydanticCustomError("invalid_repair_benefits", "Include an itemised flood repair benefit exactly when flood damage is covered")
        return self


class TheftTerms(GeneratedData):
    description: str = Field(min_length=10, max_length=1000)
    basis: Literal["vehicle_valuation_only", "vehicle_and_listed_fittings"]
    covered_fittings: list[TermItem] = Field(max_length=12)

    @model_validator(mode="after")
    def validate_fittings(self):
        if bool(self.covered_fittings) != (self.basis == "vehicle_and_listed_fittings"):
            raise ValueError("Provide a non-empty list only when theft cover includes listed fittings")
        return self


class SupportingDocumentRule(GeneratedData):
    loss_type: MotorLossType
    document_type: SupportingDocumentType
    requirement: Literal["required", "optional"]
    purpose: str = Field(min_length=1)


class LiabilityTerms(GeneratedData):
    description: str = Field(min_length=10, max_length=1000)
    conditions: list[TermItem] = Field(min_length=1, max_length=8)
    exclusions: list[TermItem] = Field(min_length=1, max_length=8)


class PolicyDraft(GeneratedData):
    insurance_type: InsuranceType
    vehicle_type: VehicleType
    permitted_use: PermittedUse
    use_conditions: list[TermItem] = Field(min_length=1, max_length=6)
    version: str = Field(min_length=1, max_length=40)
    registration: str = Field(pattern=r"^SYN-[A-Z0-9-]{1,36}$")
    vehicle: InsuredVehicle
    starts_on: date
    ends_before: date
    covered_losses: list[MotorLossType] = Field(min_length=1, max_length=4)
    repair_terms: RepairTerms | None
    theft_terms: TheftTerms | None
    liability_terms: LiabilityTerms
    police_report_required_for: list[MotorLossType] = Field(default_factory=list, max_length=4)
    supporting_documents: list[SupportingDocumentRule] = Field(default_factory=list, max_length=8)
    limit: Money
    deductible: Money
    clause: str = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_terms(self):
        losses = set(self.covered_losses)
        if self.insurance_type == "third_party" and not losses.issubset(THIRD_PARTY_LOSSES):
            raise PydanticCustomError("invalid_insurance_scope", "Third-party policies cover only the listed third-party liability losses")
        if self.insurance_type == "comprehensive" and not (losses & THIRD_PARTY_LOSSES and losses - THIRD_PARTY_LOSSES):
            raise PydanticCustomError("invalid_insurance_scope", "Comprehensive policies must include liability and an own-vehicle repair or theft benefit")
        if self.starts_on > local_date() or self.starts_on >= self.ends_before:
            raise PydanticCustomError("invalid_policy_period",
                                      "Use a start date on or before today and an end date after the start date")
        if len(losses) != len(self.covered_losses):
            raise ValueError("List each covered loss type only once")
        if not 0 <= self.deductible < self.limit:
            raise PydanticCustomError("invalid_policy_amounts", "Use a nonnegative deductible smaller than the policy limit")
        if ("repair" in self.covered_losses) != bool(self.repair_terms) or ("theft" in self.covered_losses) != bool(self.theft_terms):
            raise ValueError("Supply terms for every selected loss type, and null terms for any other loss type")
        required = set(self.police_report_required_for)
        if len(required) != len(self.police_report_required_for) or not required.issubset(self.covered_losses):
            raise ValueError("Police report requirements must list distinct covered loss types")
        document_pairs = {(rule.loss_type, rule.document_type) for rule in self.supporting_documents}
        if len(document_pairs) != len(self.supporting_documents) or any(loss not in self.covered_losses for loss, _ in document_pairs):
            raise ValueError("List each supporting document at most once for each covered loss type")
        if self.repair_terms:
            terms = self.repair_terms
            limits = [terms.labour.limit, terms.painting.limit_per_panel]
            limits.extend(item.limit_per_incident for item in terms.additional_charges.eligible_charges)
            if any(limit is not None and limit > self.limit for limit in limits):
                raise ValueError("Repair sublimits cannot exceed the overall policy limit")
        return self


class PolicyCase(GeneratedData):
    person: PersonDraft
    policy: PolicyDraft
    wording: str = Field(min_length=100, description="Full synthetic policy wording in plain text with named, numbered clauses.")


class PolicyCases(GeneratedData):
    cases: list[PolicyCase] = Field(min_length=1, max_length=POLICY_BATCH_SIZE)


def list_people_policies(store):
    with store.session() as session:
        rows = session.execute(select(Person, Policy, PolicyDocument.id, PolicyDocument.title,
                                      PolicyDocument.filename, PolicyDocument.version, PolicyDocument.status, PolicyDocument.error)
            .select_from(SyntheticPolicyHolder)
            .join(Person, Person.id == SyntheticPolicyHolder.person_id)
            .join(Policy, Policy.id == SyntheticPolicyHolder.policy_id)
            .outerjoin(PolicyBinding, PolicyBinding.policy_id == Policy.id)
            .outerjoin(PolicyDocument, (PolicyDocument.id == PolicyBinding.document_id) & generated_wording_matches(Policy.id))
            .order_by(Person.full_name, Policy.id)).all()
    return [{"person": {"person_id": person.id, "full_name": person.full_name, "district": person.district,
                        "locality": policy.payload.get("policyholder_locality")},
             "policy": policy.payload,
             "wording": {"id": document_id, "title": title, "filename": filename, "version": version,
                         "status": status, "error": error} if document_id else None}
            for person, policy, document_id, title, filename, version, status, error in rows]


def policy_design(policy, wording):
    """Compare the vehicle, substantive terms and wording independently of record IDs."""
    vehicle = policy.get("vehicle") or {}
    repair = dict(policy.get("repair_terms") or {})
    if repair:
        repair["exclusions"] = [item for item in repair.get("exclusions", []) if item != DRUNK_DRIVING_EXCLUSION]
    terms = {key: policy.get(key) for key in ("insurance_type", "vehicle_type", "permitted_use", "use_conditions",
             "theft_terms", "liability_terms", "police_report_required_for", "supporting_documents")}
    terms.update(covered_losses=sorted(policy["covered_losses"]), repair_terms=repair)
    vehicle_name = f"{vehicle.get('make', '')} {vehicle.get('model', '')}".strip()
    return vehicle_name, "".join(words(vehicle_name)), digest(terms), digest(words(wording)) if wording else None


def requested_insurance_types(policies, count):
    comprehensive = sum(policy.get("insurance_type") == "comprehensive" for policy in policies)
    third_party = sum(policy.get("insurance_type") == "third_party" for policy in policies)
    order = ("comprehensive", "third_party") if comprehensive <= third_party else ("third_party", "comprehensive")
    return [order[index % 2] for index in range(count)]


def requested_vehicle_profiles(policies, insurance_types):
    usage = [(item.get("vehicle_type"), item.get("permitted_use"), item.get("insurance_type")) for item in policies]
    selected = []
    for insurance_type in insurance_types:
        profile = min(VEHICLE_PROFILES, key=lambda item: (
            sum(prior[0] == item[0] for prior in selected),
            usage.count((*item[:2], insurance_type)),
            sum(prior[0] == item[0] for prior in usage),
            sum(prior[:2] == item[:2] for prior in usage),
        ))
        selected.append(profile)
        usage.append((*profile[:2], insurance_type))
    return selected


def requested_districts(districts, count):
    usage = Counter(words(district) for district in districts)
    selected = []
    for _ in range(count):
        district = min(DISTRICTS, key=lambda name: usage[words(name)])
        selected.append(district)
        usage[words(district)] += 1
    return selected


def generate_people_policies(store, count):
    cases = []
    today = local_date()
    with store.session() as session:
        saved = session.execute(select(Policy.payload, PolicyDocument.content, Person.full_name, Person.district).select_from(Policy)
            .join(SyntheticPolicyHolder, SyntheticPolicyHolder.policy_id == Policy.id)
            .join(Person, Person.id == SyntheticPolicyHolder.person_id)
            .outerjoin(PolicyBinding, PolicyBinding.policy_id == Policy.id)
            .outerjoin(PolicyDocument, (PolicyDocument.id == PolicyBinding.document_id) & generated_wording_matches(Policy.id))
            .order_by(PolicyDocument.created_at, Policy.id)).all()
    used = []
    for policy, content, _, _ in saved:
        wording = base64.b64decode(content).decode().partition("\n\n")[2] if content else ""
        wording = wording.partition("\n\nScope by insurance type")[0].partition("\n\nFlood damage under repair cover")[0].partition("\n\nDocuments for coverage assessment")[0]
        used.append(policy_design(policy, wording))
    saved_policies = [policy for policy, _, _, _ in saved]
    insurance_types = requested_insurance_types(saved_policies, count)
    profiles = requested_vehicle_profiles(saved_policies, insurance_types)
    districts = requested_districts([district for _, _, _, district in saved], count)
    names = [name for _, _, name, _ in saved]
    locations = [(district, policy["policyholder_locality"]) for policy, _, _, district in saved if policy.get("policyholder_locality")]
    with generation_client() as (client, model):
        for offset in range(0, count, POLICY_BATCH_SIZE):
            size = min(POLICY_BATCH_SIZE, count - offset)
            expected_types = insurance_types[offset:offset + size]
            expected_profiles = profiles[offset:offset + size]
            expected_districts = districts[offset:offset + size]
            profile_instructions = " ".join(
                f"Case {index}: vehicle_type={vehicle}, permitted_use={use}. Use district {district}."
                + (f" If offering repair cover, include a {cause} repair benefit and tailor it to this profile. {hint}"
                   if product == "comprehensive" else "")
                for index, ((vehicle, use, cause, hint), product, district) in enumerate(
                    zip(expected_profiles, expected_types, expected_districts, strict=True), start=1)
            )
            prompt = (
                f"Create exactly {size} fictional Sri Lankan people and one associated motor policy for each. "
                f"Today in Sri Lanka is {today.isoformat()}. Use fictional names, Sri Lankan districts and SYN- vehicle registrations. "
                f"Set starts_on to {today.isoformat()} or earlier, preferably several months earlier so a claim can be generated. "
                "Set ends_before after starts_on. Set a positive overall limit and a nonnegative deductible strictly below it. "
                "Use distinct full names, and vary given names and surnames. Choose a specific town or neighbourhood in each assigned district. "
                "Use the locality field for the town or neighbourhood and the district field for the assigned district. "
                f"Avoid reusing these recent names: {', '.join(names[-24:]) or 'none'}. "
                f"Avoid reusing these recent locations: {', '.join(f'{locality}, {district}' for district, locality in locations[-24:]) or 'none'}. "
                f"Use these insurance_type values in order: {', '.join(expected_types)}. Vary the term dates, limits and deductibles. "
                f"Use these exact vehicle and use assignments. {profile_instructions} "
                "Choose a make and model that actually belong to the assigned vehicle class. In use_conditions, state which trips and activities are allowed "
                "and which are outside cover. Tailor liability, theft and, where applicable, repair conditions to the vehicle's work or private use. "
                "For example, a private hatchback can exclude paid delivery, while a delivery van may cover its own listed damage on delivery trips "
                "but exclude loss of goods. A passenger-hire policy can address passengers and authorised hire journeys. "
                "Comprehensive cover includes third-party liability and listed own-vehicle repair or theft. Usually cover both own-vehicle loss types, with tailored exclusions. "
                "Third-party cover only covers liability for the listed third_party_property_damage and/or third_party_injury losses. "
                "Set repair_terms and theft_terms to null on third-party cover. Exclude damage to or theft of the insured vehicle. "
                "For both products, explain in liability_terms whose property or injury is covered, how the insured party's liability is established, "
                "and conditions and exclusions. Require an itemised third-party demand and specify any necessary police or supporting records. "
                "Use plausible fictional terms. The insured vehicle's registration identifies the incident, not the third party's damaged property. "
                "Use a different vehicle make and model combination for every policy. Vary makes, model years and vehicle types, "
                "such as a compact hatchback, van, pickup, sedan or crossover from different manufacturers. The server assigns a fictional chassis number. "
                "Give each policy a distinct combination of covered events, repair scope, exclusions and applicable evidence or benefit conditions. "
                "Write new clauses and headings that describe these differences. Changing dates, names, amounts or identifiers alone is insufficient. "
                f"Avoid these recently used vehicle models: {', '.join(record[0] for record in used[-24:] if record[0]) or 'none'}. "
                "When repair is offered, list separate repair_benefits by insured cause. Include at least one non-collision cause suited to the vehicle and permitted use. "
                "Give each cause a description, specific covered parts and work, conditions, and exclusions. Use one to four benefits. "
                "Vary the mix of flood, storm, fire, vandalism, falling objects and animal impact. "
                "Collision is an option. Do not make every policy collision-focused or use the same part list for unrelated causes. "
                "Require an itemised repair estimate showing registration, garage name and full address. A VIN or chassis number is not required on the estimate. "
                "The server adds the drunk-driving exclusion under repair exclusions. The police report is the source for checking this exclusion. "
                "Leave this exclusion and the repair estimate clause to the server. "
                "For every repair policy, set flood_damage to covered or excluded. Vary this choice across policies and use both choices "
                "when generating multiple repair policies. When covered, include repair work that can apply to direct floodwater damage, "
                "such as drying the cabin or replacing flood-damaged wiring. When excluded, floodwater damage is excluded even for otherwise listed repairs. "
                "Include a dedicated flood repair benefit only when flood_damage is covered. Keep the draft wording consistent with the scheduled causes and scope. "
                "The server writes the flood clause, permitted use and itemised repair benefits from the schedule. "
                "Define labour and painting limits in their structured fields. Use null where there is no separate cap. Use painting.scope to distinguish affected panels from a necessary full respray. "
                "Always set additional_charges.covered and list the eligible charges if covered. Explicitly list each excluded charge in additional_charges.exclusions, with its category, name and reason. "
                "Address towing, storage, diagnostics and temporary_transport explicitly as eligible or excluded. Unlisted extras are excluded. "
                "Vary the benefits across policies. Eligible additional charges can include towing, short-term storage, diagnostic inspections, "
                "temporary transport, or other named services. A policy can cover additional charges without covering towing. "
                "For each eligible charge give a distinct lowercase snake_case category, a readable name, exact eligibility conditions, "
                "whether a receipt is required, and a per-incident sublimit or null for no separate cap. Use [] if all additional charges are excluded. "
                "Keep the description and other entries consistent with these fields. All sublimits sit within the overall limit. "
                "Use null repair_terms if repair is not covered. "
                "For theft cover, specify how vehicle value is assessed and vary between vehicle_valuation_only and vehicle_and_listed_fittings. "
                "For the latter, name the exact separately covered fittings and describe the evidence required. "
                "Vehicle valuation only pays for the vehicle and excludes separately claimed fittings and other costs. "
                "Use an empty covered_fittings list for valuation-only policies and null theft_terms if theft is not covered. "
                "Vary the police-report requirement across fictional policies. Set police_report_required_for to the covered loss types "
                "that require a police report as a condition of assessment, or [] when it is optional for every covered loss type. "
                "Use this field for police-report requirements so the terms are consistent. "
                "Use supporting_documents for extra incident evidence, such as a witness_statement describing what an observer saw "
                "or an incident_report recording the circumstances. For each rule specify loss_type, document_type, required or optional "
                "as requirement, and the coverage fact it helps verify as purpose. Only require evidence critical for every incident of that loss type. "
                "Include optional rules where useful and vary the mix by policy. "
                "Use YYYY-MM-DD dates and LKR decimal strings. The policy end date is exclusive. "
                "Alongside each schedule, write the full fictional policy wording in wording. Use short numbered clauses "
                "with clear headings for scope, repair and theft benefits, exclusions, limits, deductibles, additional charges "
                "and required evidence as applicable. Spell out the benefit conditions in prose, including when listed fittings "
                "or extra repair costs qualify. Match the schedule's terms, including which evidence is required or optional. Where identity, dates or amounts are "
                "schedule-specific, refer to that schedule. Do not write IDs, registrations or chassis numbers in the wording; "
                "the server supplies the final policy identity. Aim for 250–400 words in the wording. "
                "For example, the tool input shape is "
                '{"cases":[{"person":{"full_name":"Sanduni Jayalath","district":"Galle","locality":"Hikkaduwa"},'
                '"policy":{"insurance_type":"comprehensive","vehicle_type":"hatchback","permitted_use":"private",'
                '"use_conditions":["Personal trips and commuting are included. Paid delivery and carrying passengers for hire are excluded."],'
                '"version":"2025.2","registration":"SYN-EX-381","starts_on":"2025-06-01",'
                '"ends_before":"2026-06-01","vehicle":{"make":"Toyota","model":"Vitz","year":2018},'
                '"covered_losses":["repair","theft","third_party_property_damage"],'
                '"liability_terms":{"description":"Covers liability for damage to another person’s property in a collision involving the insured vehicle.",'
                '"conditions":["Evidence must link the insured driver to the collision and establish responsibility."],'
                '"exclusions":["Damage to property owned by the insured is outside this liability benefit."]},'
                '"police_report_required_for":["theft"],'
                '"supporting_documents":[{"loss_type":"repair","document_type":"witness_statement","requirement":"optional",'
                '"purpose":"Corroborate the reported collision when a witness was present."}],'
                '"repair_terms":{"description":"Insures specified direct collision and flood damage.",'
                '"flood_damage":"covered","repair_benefits":['
                '{"cause":"collision","description":"Sudden impacts on public roads or in parking areas.",'
                '"covered_repairs":["Bumper and affected panel repair","Fitting labour on impact-damaged panels"],"conditions":[],"exclusions":[]},'
                '{"cause":"flood","description":"Sudden external floodwater entering the cabin.",'
                '"covered_repairs":["Dry and clean submerged cabin trim","Replace flood-damaged electrical connectors"],'
                '"conditions":["Water exposure during a permitted journey or while parked"],"exclusions":["Mould from gradual leaks"]}],'
                '"exclusions":["Full-vehicle respray","Mechanical wear and maintenance"],'
                '"labour":{"basis":"per_panel","limit":"15000.00"},'
                '"painting":{"scope":"affected_panels_only","limit_per_panel":"8000.00","max_panels":4},'
                '"additional_charges":{"covered":true,"eligible_charges":[{"category":"storage","name":"Vehicle storage",'
                '"limit_per_incident":"10000.00","receipt_required":true,'
                '"conditions":"Up to two days of storage at the approved repairer while awaiting inspection following a covered collision."}],'
                '"exclusions":[{"category":"towing","name":"Towing","reason":"Recovery and transport are outside repair cover."},'
                '{"category":"diagnostics","name":"Diagnostic inspections","reason":"Stand-alone inspections are outside this policy."},'
                '{"category":"temporary_transport","name":"Temporary transport","reason":"Replacement vehicle hire and travel costs are excluded."}]}},'
                '"theft_terms":{"basis":"vehicle_and_listed_fittings","covered_fittings":["Fixed dash camera"],'
                '"description":"Covers pre-loss market valuation of the vehicle plus its fixed dash camera if itemised separately with proof of installation."},'
                '"limit":"400000.00","deductible":"12500.00","clause":"SYN-MOTOR-02"},'
                '"wording":"1. Collision repairs. This cover pays for collision damage to the scheduled vehicle’s bumpers '
                'and body panels, subject to the schedule and the following clauses. ..."}]}. '
                "The example abbreviates the document; write the full wording. "
                "A third-party schedule instead lists only its liability loss types, supplies liability_terms, and uses null for both own-vehicle term sections. "
                "Invent fresh values and submit the requested number of records."
            )
            for attempt in range(2):
                try:
                    batch = request_batch(client, model, prompt, GENERATION_TOOL, PolicyCases, size)
                except GenerationValidationError as error:
                    if attempt or (error.issue_type not in {"invalid_insurance_scope", "invalid_repair_benefits",
                                                            "invalid_policy_period", "invalid_policy_amounts"}
                                   and not {"insurance_type", "liability_terms", "repair_benefits", "vehicle_type",
                                            "permitted_use", "use_conditions", "flood_damage", "person"}.intersection(error.location)):
                        raise
                    prompt += f" Correct the policy schedule. {error} Return the full batch in the same order."
                    continue
                designs = [policy_design(case.policy.model_dump(mode="json"), case.wording) for case in batch]
                wrong_types = [case.policy.insurance_type for case in batch] != expected_types
                wrong_profiles = any(
                    (case.policy.vehicle_type, case.policy.permitted_use) != profile[:2]
                    or (case.policy.repair_terms and profile[2] not in {benefit.cause for benefit in case.policy.repair_terms.repair_benefits})
                    for case, profile in zip(batch, expected_profiles, strict=True)
                )
                wrong_districts = [words(case.person.district) for case in batch] != [words(district) for district in expected_districts]
                fields = ("vehicle make/model", "policy terms", "policy wording")
                repeated = []
                for index, label in enumerate(fields, start=1):
                    current = [design[index] for design in designs]
                    previous = {design[index] for design in used}
                    if len(set(current)) != len(current) or previous.intersection(current):
                        repeated.append(label)
                new_names = [case.person.full_name for case in batch]
                new_locations = [(district, case.person.locality) for district, case in zip(expected_districts, batch, strict=True)]
                name_keys = ["".join(words(name)) for name in new_names]
                location_keys = {(words(district), words(locality)) for district, locality in new_locations}
                if len(set(name_keys)) != len(batch) or set(name_keys).intersection("".join(words(name)) for name in names):
                    repeated.append("policyholder name")
                if len(location_keys) != len(batch) or location_keys.intersection((words(district), words(locality)) for district, locality in locations):
                    repeated.append("policyholder location")
                if not repeated and not wrong_types and not wrong_profiles and not wrong_districts:
                    cases.extend(batch)
                    used.extend(designs)
                    names.extend(new_names)
                    locations.extend(new_locations)
                    break
                if attempt and wrong_types:
                    raise GenerationError("Claude did not return the requested insurance types after a retry.")
                if attempt and wrong_profiles:
                    raise GenerationError("Claude did not follow the requested vehicle, permitted use and repair coverage assignments after a retry.")
                if attempt and wrong_districts:
                    raise GenerationError("Claude did not use the requested policyholder districts after a retry.")
                if attempt:
                    raise GenerationError(f"Claude repeated {', '.join(repeated)} after a retry. Generate a new batch.")
                if repeated:
                    prompt += f" The previous set repeated {', '.join(repeated)}. Generate entirely different policies and wording."
                if wrong_types:
                    prompt += f" The previous types were incorrect. Use exactly these types in order: {', '.join(expected_types)}."
                if wrong_profiles:
                    prompt += f" The vehicle, permitted use or repair coverage assignments were incorrect. {profile_instructions}"
                if wrong_districts:
                    prompt += f" Use these policyholder districts in order: {', '.join(expected_districts)}."
    with store.session.begin() as session:
        for case, district in zip(cases, districts, strict=True):
            person_id = f"PER-{uuid.uuid4().hex.upper()}"
            policy_id = f"POL-{uuid.uuid4().hex.upper()}"
            policy = {"policy_id": policy_id, "policyholder_locality": case.person.locality, **case.policy.model_dump(mode="json")}
            # Registrations are assigned uniquely to new vehicles in this catalogue.
            policy["registration"] = f"{policy['registration'][:27]}-{policy_id[-12:]}"
            policy["vehicle"]["chassis_number"] = f"SYN-CH-{policy_id[4:]}"
            header = (f"Synthetic motor policy wording\nPolicy {policy_id} · Version {policy['version']} · "
                      f"Schedule clause {policy['clause']}\n\n")
            wording_text = header + case.wording
            scope = ("This third-party policy covers only the listed liability for harm to others. Damage to or theft of the insured vehicle is excluded."
                     if case.policy.insurance_type == "third_party" else
                     "This comprehensive policy covers the listed own-vehicle benefits and the listed third-party liability subject to their separate terms.")
            wording_text += "\n\nScope by insurance type\n" + scope
            wording_text += (f"\n\nPermitted vehicle use\nThe insured vehicle is a {policy['vehicle_type'].replace('_', ' ')} "
                             f"scheduled for {policy['permitted_use'].replace('_', ' ')}. "
                             + " ".join(case.policy.use_conditions))
            liability = case.policy.liability_terms
            wording_text += "\n\nThird-party liability\n" + liability.description
            listed_harm = [loss.removeprefix("third_party_").replace("_", " ") for loss in policy["covered_losses"] if loss in THIRD_PARTY_LOSSES]
            wording_text += "\nCovered liability losses: " + ", ".join(listed_harm) + "."
            wording_text += "\nConditions\n" + "\n".join(f"- {term}" for term in liability.conditions)
            wording_text += "\nExclusions\n" + "\n".join(f"- {term}" for term in liability.exclusions)
            if case.policy.repair_terms:
                if DRUNK_DRIVING_EXCLUSION not in policy["repair_terms"]["exclusions"]:
                    policy["repair_terms"]["exclusions"].append(DRUNK_DRIVING_EXCLUSION)
                flood_clause = {
                    "covered": "Direct physical damage to the insured vehicle caused by flooding is covered. Repair work must be listed "
                               "in the schedule and is subject to the applicable exclusions, limits and deductible.",
                    "excluded": "Damage caused by flooding or floodwater is excluded from repair cover. This includes damage from "
                                "submersion or driving through floodwater, even when the affected parts are normally covered.",
                }
                wording_text += "\n\nFlood damage under repair cover\n" + flood_clause[case.policy.repair_terms.flood_damage]
                for benefit in case.policy.repair_terms.repair_benefits:
                    wording_text += f"\n\nRepair benefit for {benefit.cause.replace('_', ' ')}\n{benefit.description}"
                    wording_text += "\nCovered repairs: " + "; ".join(benefit.covered_repairs) + "."
                    if benefit.conditions:
                        wording_text += "\nConditions: " + "; ".join(benefit.conditions) + "."
                    if benefit.exclusions:
                        wording_text += "\nExclusions: " + "; ".join(benefit.exclusions) + "."
                wording_text += ("\n\nRepair estimate requirements\nSubmit a dated, itemised repair estimate showing "
                                 "the vehicle registration, garage name and full garage address. A VIN or chassis number "
                                 "is not required on the repair estimate.")
                wording_text += "\n\nRepair exclusion for drunk driving\n" + DRUNK_DRIVING_EXCLUSION
            checklist = ["\n\nDocuments for coverage assessment"]
            for loss_type in policy["covered_losses"]:
                checklist.append(f"{loss_type.replace('_', ' ').capitalize()} claims")
                checklist.extend(f"- {rule['name']} ({rule['requirement']}). {rule['purpose']}"
                                 for rule in coverage_document_checklist(policy, loss_type))
            wording = document_record(title=f"Synthetic wording · {policy_id}", product=GENERATED_WORDING_PRODUCT,
                                      version=policy["version"], filename=f"{policy_id}.txt", media_type="text/plain",
                                      content=base64.b64encode((wording_text + "\n".join(checklist)).encode()).decode())
            session.add(Person(id=person_id, full_name=case.person.full_name, district=district))
            session.add(Policy(id=policy_id, payload=policy))
            session.add(wording)
            session.flush()
            session.add(SyntheticPolicyHolder(policy_id=policy_id, person_id=person_id))
            session.add(PolicyBinding(policy_id=policy_id, document_id=wording.id))
    return {"count": len(cases), "model": model}
