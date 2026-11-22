"""Classify synthetic claims from their evidence and validate intended omissions or coverage differences."""
from datetime import date
from decimal import Decimal

from backend.agents.coverage_specialist import additional_charge_review_reasons, document_vehicle_matches, normalize_chassis
from backend.agents.intake_specialist import cost_breakdown, cost_evidence
from backend.claim_schemas import AUTHORITY, MOTOR_LOSSES
from backend.evidence.required_claim_evidence import COST_DOCUMENT_KINDS, additional_charge_totals, has_police_report, missing_supporting_documents, policy_lists_loss, requires_police_report


COMPLETE_SCENARIOS = {"ordinary", "optional_police_report_attached", "required_police_report_attached",
                      "covered_additional_cost"}
ESTIMATE_NARRATIVE_MISMATCH = "estimate_narrative_mismatch"
REPAIR_SUBLIMIT_EXCEEDED = "repair_sublimit_exceeded"
INCOMPLETE_NARRATIVE = "incomplete_narrative"
CONTRADICTORY_NARRATIVE = "contradictory_narrative"
NARRATIVE_SCENARIOS = {INCOMPLETE_NARRATIVE, CONTRADICTORY_NARRATIVE}
MISSING_SUPPORTING_DOCUMENT = "missing_required_supporting_document"
OMISSIONS = {
    "missing_loss_date": "Date of loss on the claim form",
    "missing_chassis": "Chassis number on the claim form",
    "missing_cost_document": "Itemised cost document",
    "missing_itemisation": "Individual charges on the cost document",
    "missing_required_police_report": "Policy-required police report",
    MISSING_SUPPORTING_DOCUMENT: "Policy-required witness statement or incident report",
}
SCENARIOS = {
    "ordinary": "An in-period covered loss with complete cost evidence. Choose a loss for which the police report is optional and deliberately omit it.",
    "optional_police_report_attached": "An otherwise ordinary covered loss for which a police report is optional. Include it as supporting incident evidence.",
    "required_police_report_attached": "An otherwise ordinary covered loss for which the selected policy requires a police report. Include the report.",
    "missing_required_police_report": "An otherwise ordinary covered loss for which the selected policy requires a police report. Omit the report and keep all cost documents.",
    MISSING_SUPPORTING_DOCUMENT: "An otherwise complete covered loss. Omit one required witness statement or incident report.",
    "before_policy_start": "A readable loss date before the policy start date.",
    "after_policy_end": "A readable loss date on or after the policy end date and no later than today.",
    "uncovered_loss": "An in-period claim for a loss outside the policy's listed cover, such as own-vehicle repairs under third-party cover.",
    "missing_loss_date": "The date of loss is unreadable; use a null date.",
    "wrong_vehicle": "The claim or cost document identifies a different vehicle through its registration or chassis number.",
    "missing_chassis": "The policy records a chassis number but the claimant cannot supply it.",
    "missing_cost_document": "The cost document required for the loss is omitted.",
    "missing_itemisation": "The cost document has a total but no individual charges.",
    "cost_document_before_loss": "An otherwise consistent cost document is dated before the incident.",
    "cost_document_arithmetic_mismatch": "The itemised charges do not add up to the document total.",
    ESTIMATE_NARRATIVE_MISMATCH: "An itemised repair estimate includes work on a part the claimant explicitly says was undamaged.",
    INCOMPLETE_NARRATIVE: "A claimant account omits one important incident fact such as the location, cause or damage. Keep the submitted documents.",
    CONTRADICTORY_NARRATIVE: "Two statements in the claimant's account give incompatible facts about the same incident. Attached evidence supports one account.",
    "above_policy_limit": "The gross amount exceeds the policy limit and matches the itemised cost evidence.",
    REPAIR_SUBLIMIT_EXCEEDED: "An otherwise covered repair has one itemised labour or painting charge above its own policy cap. The reconciled gross stays within the overall limit and LKR 250,000.",
    "unexplained_additional_cost": "Gross is above the cost document total with no additional charges itemised.",
    "partial_amount": "The claimant requests less than the cost document total and explains the difference.",
    "itemised_additional_cost": "Gross includes a separately itemised and explained charge that needs coverage or evidence review.",
    "covered_additional_cost": "An otherwise ordinary repair with a separately claimed charge expressly covered by the policy and its required receipt.",
}
REUSED_COST_DOCUMENT = "reused_cost_document"


def available_scenarios(policy, today):
    scenarios = dict(SCENARIOS)
    covered = set(policy["covered_losses"])
    if "repair" not in covered:
        scenarios.pop(ESTIMATE_NARRATIVE_MISMATCH)
    if not repair_estimate_sublimits(policy):
        scenarios.pop(REPAIR_SUBLIMIT_EXCEEDED)
    if not any(rule["requirement"] == "required" for rule in policy.get("supporting_documents", [])):
        scenarios.pop(MISSING_SUPPORTING_DOCUMENT)
    report_required = covered.intersection(policy.get("police_report_required_for") or ())
    if not report_required:
        scenarios.pop("required_police_report_attached")
        scenarios.pop("missing_required_police_report")
    if covered == report_required:
        scenarios.pop("ordinary")
        scenarios.pop("optional_police_report_attached")
    if policy["ends_before"] > today.isoformat():
        scenarios.pop("after_policy_end")
    if policy["starts_on"] == date.min.isoformat():
        scenarios.pop("before_policy_start")
    if MOTOR_LOSSES.issubset(covered):
        scenarios.pop("uncovered_loss")
    if not (policy.get("vehicle") or {}).get("chassis_number"):
        scenarios.pop("missing_chassis")
    if not ((policy.get("repair_terms") or {}).get("additional_charges") or {}).get("eligible_charges"):
        scenarios.pop("covered_additional_cost")
    if Decimal(policy["limit"]) >= Decimal("999999999999.99"):
        scenarios.pop("above_policy_limit")
    return scenarios


def repair_estimate_sublimits(policy):
    if not policy_lists_loss(policy, "repair"):
        return {}
    maximum = min(Decimal(policy["limit"]), AUTHORITY)
    if Decimal(policy["deductible"]) >= maximum:
        return {}
    terms = policy.get("repair_terms") or {}
    labour, painting = terms.get("labour") or {}, terms.get("painting") or {}
    limits = {}
    if labour.get("basis") in {"per_panel", "per_claim"} and labour.get("limit") is not None:
        limits["labour"] = (labour["basis"], Decimal(labour["limit"]))
    if painting.get("scope") in {"affected_panels_only", "full_respray_if_required"} and painting.get("limit_per_panel") is not None:
        limits["painting"] = ("per_panel", Decimal(painting["limit_per_panel"]))
    # Leave room for a strictly higher line and at least one other positive line.
    return {key: value for key, value in limits.items() if 0 < value[1] <= maximum - Decimal("0.02")}


def classify_scenario(claim, policy):
    loss_date = claim.get("loss_date")
    if not loss_date:
        return "missing_loss_date"
    if loss_date < policy["starts_on"]:
        return "before_policy_start"
    if loss_date >= policy["ends_before"]:
        return "after_policy_end"
    if not policy_lists_loss(policy, claim["loss_type"]):
        return "uncovered_loss"
    documents, _, _ = cost_evidence(claim)
    if not documents:
        return "missing_cost_document"
    if any(document.get("document_reference") for document in documents):
        return REUSED_COST_DOCUMENT
    if any(not document_vehicle_matches(document, policy) for document in documents):
        return "wrong_vehicle"
    insured_vehicle = policy.get("vehicle") or {}
    reported_vehicle = claim.get("vehicle") or {}
    insured_chassis = normalize_chassis(insured_vehicle.get("chassis_number") or "")
    reported_chassis = normalize_chassis(reported_vehicle.get("chassis_number") or "")
    document_chassis = [normalize_chassis(document.get("chassis_number") or "") for document in documents
                        if document["type"] != "repair_estimate"]
    if insured_chassis and any(number and number != insured_chassis for number in [reported_chassis, *document_chassis]):
        return "wrong_vehicle"
    if any(insured_vehicle.get(field) and reported_vehicle.get(field)
           and str(insured_vehicle[field]).casefold() != str(reported_vehicle[field]).casefold()
           for field in ("make", "model", "year")):
        return "wrong_vehicle"
    if insured_chassis and not reported_chassis:
        return "missing_chassis"
    if any(not document.get("line_items") or document.get("total") is None or
           any(line.get("amount") is None for line in document["line_items"]) for document in documents):
        return "missing_itemisation"
    if any(document.get("date") and document["date"] < loss_date for document in documents):
        return "cost_document_before_loss"
    if any(sum(Decimal(line["amount"]) for line in document["line_items"]) != Decimal(document["total"]) for document in documents):
        return "cost_document_arithmetic_mismatch"
    gross = Decimal(claim["gross_amount"])
    if gross > Decimal(policy["limit"]):
        return "above_policy_limit"
    document_total, extras = cost_breakdown(claim, documents)
    if gross > document_total + extras:
        return "unexplained_additional_cost"
    if gross < document_total + extras:
        return "partial_amount"
    if missing_supporting_documents(policy, claim):
        return MISSING_SUPPORTING_DOCUMENT
    report_required = requires_police_report(policy, claim)
    report_attached = has_police_report(claim)
    if report_required and not report_attached:
        return "missing_required_police_report"
    if extras:
        terms = (policy.get("repair_terms") or {}).get("additional_charges")
        if claim["loss_type"] == "repair" and terms and not additional_charge_review_reasons(claim, terms):
            return "covered_additional_cost"
        return "itemised_additional_cost"
    if report_required:
        return "required_police_report_attached"
    return "optional_police_report_attached" if report_attached else "ordinary"


def baseline_limit(policy):
    limit = Decimal(policy["limit"])
    return min(limit, AUTHORITY) if Decimal(policy["deductible"]) < AUTHORITY else limit


def is_complete_reference_claim(claim, policy):
    """Validate the structured evidence needed before a planned omission is applied."""
    if classify_scenario(claim, policy) not in COMPLETE_SCENARIOS:
        return False
    documents, _, _ = cost_evidence(claim)
    if len(documents) != 1 or not (Decimal(policy["deductible"]) < Decimal(claim["gross_amount"]) <= baseline_limit(policy)):
        return False
    if documents[0]["type"] == "repair_estimate" and any(
            not (documents[0].get(key) or "").strip() for key in ("garage_name", "garage_address")):
        return False
    if any(not document_vehicle_matches(doc, policy)
           or not claim["loss_date"] <= doc["date"] <= claim["reported_date"] for doc in claim["documents"]):
        return False
    charge_categories = additional_charge_totals(claim)
    if (sum(doc["type"] == "police_report" for doc in claim["documents"]) > 1
            or any(doc["type"] in COST_DOCUMENT_KINDS and doc != documents[0]
                   or doc["type"] == "additional_cost_receipt" and doc["category"] not in charge_categories
                   for doc in claim["documents"])):
        return False
    if any(Decimal(line["amount"]) <= 0 for line in documents[0]["line_items"]):
        return False
    return True


def omit_required_evidence(claim, scenario):
    """Apply a single deliberate defect to the verified complete submission."""
    documents, cost_type, _ = cost_evidence(claim)
    if scenario == "missing_loss_date":
        claim["loss_date"] = None
    elif scenario == "missing_chassis" and claim.get("vehicle"):
        claim["vehicle"]["chassis_number"] = None
    elif scenario == "missing_cost_document":
        claim["documents"] = [doc for doc in claim["documents"] if doc["type"] != cost_type]
    elif scenario == "missing_itemisation":
        documents[0]["line_items"] = []
    elif scenario == "missing_required_police_report":
        claim["documents"] = [doc for doc in claim["documents"] if doc["type"] != "police_report"]
