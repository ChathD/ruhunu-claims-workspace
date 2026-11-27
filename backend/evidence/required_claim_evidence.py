"""Resolve claim evidence requirements and identify gaps that block assessment."""
import re
from decimal import Decimal

from backend.claim_schemas import THIRD_PARTY_LOSSES

SUPPORTING_DOCUMENT_LABELS = {"incident_report": "Incident report", "witness_statement": "Witness statement"}
INCOME_LOSS_EVIDENCE = {
    "work_incapacity_evidence": "Evidence of time unable to work",
    "earnings_evidence": "Evidence of earnings",
}
INCOME_LOSS = re.compile(
    r"\b(?:loss[ -]+of|lost|foregone|forgone)[ -]+(?:net |daily )?(?:income|earnings|wages|salary|pay)\b"
    r"|\b(?:income|earnings|wage|salary)[ -]+loss\b|\b(?:unpaid|missed) (?:work ?days?|shifts?)\b", re.I,
)
DRUNK_DRIVING_EXCLUSION = "Repair damage while the insured vehicle was driven under the influence of alcohol, as substantiated by the police report."
COST_DOCUMENT_TYPES = {
    "theft": ("vehicle_valuation", "Itemised vehicle valuation"),
    "repair": ("repair_estimate", "Itemised repair estimate"),
    "glass": ("repair_estimate", "Itemised repair estimate"),
    "third_party_property_damage": ("third_party_demand", "Itemised third-party property damage demand"),
    "third_party_injury": ("third_party_demand", "Itemised third-party injury demand"),
}
COST_DOCUMENT_KINDS = {kind for kind, _ in COST_DOCUMENT_TYPES.values()}

REQUIRED_CHECKS = {
    "policy_id": "policy_lookup",
    "loss_date": "policy_period",
    "registration": "vehicle_match",
    "loss_type": "loss_and_costs",
    "narrative": "loss_and_costs",
    "gross_amount": "loss_and_costs",
    "currency": "loss_and_costs",
    "repair_estimate": "loss_and_costs",
    "repair_estimate.garage_name": "loss_and_costs",
    "repair_estimate.garage_address": "loss_and_costs",
    "vehicle_valuation": "loss_and_costs",
    "third_party_demand": "loss_and_costs",
    "motor_benefit_receipt": "loss_and_costs",
    "motor_benefit_receipt.payment_status": "loss_and_costs",
    "police_report": "loss_and_costs",
    "incident_report": "loss_and_costs",
    "witness_statement": "loss_and_costs",
    "cost_document.registration": "loss_and_costs",
    "additional_cost_receipt": "loss_and_costs",
    "date": "loss_and_costs",
    "total": "loss_and_costs",
    "line_items": "loss_and_costs",
    "additional_costs": "loss_and_costs",
    **dict.fromkeys(INCOME_LOSS_EVIDENCE, "loss_and_costs"),
}


def policy_lists_loss(policy, loss_type):
    if policy.get("insurance_type") == "third_party" and loss_type not in THIRD_PARTY_LOSSES:
        return False
    return loss_type in policy.get("covered_losses", ())


def policy_applies_to_loss(policy, claim):
    loss_date = claim.get("loss_date")
    start, end = policy.get("starts_on"), policy.get("ends_before")
    loss_type = claim.get("loss_type")
    return bool(loss_date and start and end and start <= loss_date < end and policy_lists_loss(policy, loss_type))


def requires_police_report(policy, claim):
    return policy_applies_to_loss(policy, claim) and claim.get("loss_type") in (policy.get("police_report_required_for") or ())


def supporting_document_rules(policy, loss_type):
    return [rule for rule in policy.get("supporting_documents", []) if rule["loss_type"] == loss_type]


def missing_supporting_documents(policy, claim):
    if not policy_applies_to_loss(policy, claim):
        return []
    attached = {document.get("type") for document in claim.get("documents", [])}
    return [rule for rule in supporting_document_rules(policy, claim["loss_type"])
            if rule["requirement"] == "required" and rule["document_type"] not in attached]


def coverage_document_checklist(policy, loss_type):
    if loss_type not in COST_DOCUMENT_TYPES:
        return []
    cost_type, cost_name = COST_DOCUMENT_TYPES[loss_type]
    purpose = ("Identify the third party and substantiate their claimed costs for the reported incident."
               if loss_type in THIRD_PARTY_LOSSES else "Link the costs to the insured vehicle and the reported loss.")
    if cost_type == "repair_estimate":
        purpose = "Link the proposed repairs and costs to the reported loss and identify the garage by name and address."
    documents = [{"document_type": cost_type, "name": cost_name, "requirement": "required",
                  "purpose": purpose}]
    if "police_report_required_for" in policy:
        documents.append({"document_type": "police_report", "name": "Police report",
                          "requirement": "required" if loss_type in policy["police_report_required_for"] else "optional",
                          "purpose": "Corroborate when and how the incident occurred."})
    documents.extend({**rule, "name": SUPPORTING_DOCUMENT_LABELS[rule["document_type"]]}
                     for rule in supporting_document_rules(policy, loss_type))
    if loss_type == "repair":
        terms = (policy.get("repair_terms") or {}).get("additional_charges") or {}
        documents.extend({"document_type": "additional_cost_receipt", "name": f"{benefit['name']} receipt",
                          "category": benefit["category"], "requirement": "required" if benefit["receipt_required"] else "optional",
                          "purpose": f"Substantiate {benefit['name'].lower()} when that separate charge is claimed."}
                         for benefit in terms.get("eligible_charges", []))
    return documents


def additional_charge_totals(claim):
    totals = {}
    for charge in claim.get("additional_costs", ()):
        if category := charge.get("category"):
            totals[category] = totals.get(category, Decimal(0)) + Decimal(charge["amount"])
    return totals


def cost_line_total(line_items, *, initial=Decimal(0)):
    return sum((Decimal(item["amount"]) for item in line_items), initial)


def additional_charge_receipts(claim, category):
    return [doc for doc in claim.get("documents", ())
            if doc.get("type") == "additional_cost_receipt" and doc.get("category") == category]


def has_police_report(claim):
    return any(doc.get("type") == "police_report" for doc in claim.get("documents", ())) or any(
        doc.get("status") == "extracted" and doc.get("document_type") == "police_report"
        for doc in claim.get("uploaded_evidence", ())
    )


def required_gap(field, document, location, reason):
    return {"field": field, "document": document, "location": location, "reason": reason,
            "required_for": REQUIRED_CHECKS[field]}


def missing_income_loss_evidence(claim, coverage_items):
    if claim.get("loss_type") != "third_party_injury" or any(
            item["outcome"] == "not_covered" and INCOME_LOSS.search(item["item"]) for item in coverage_items):
        return []
    documents = claim.get("documents") or []
    amount = sum((Decimal(item["amount"]) for doc in documents if doc.get("type") == "third_party_demand"
                  for item in doc.get("line_items") or [] if item.get("amount") not in (None, "")
                  and INCOME_LOSS.search(item.get("description") or "")), Decimal(0))
    if amount <= 0:
        return []
    attached = {doc.get("type") for doc in documents} | {
        doc.get("document_type") for doc in claim.get("uploaded_evidence") or [] if doc.get("status") == "extracted"
    }
    demand = f"The third-party demand claims {claim['currency']} {amount:,.2f} for lost income."
    explanations = {
        "work_incapacity_evidence": "Provide a medical certificate or other evidence showing how long the injury prevented work.",
        "earnings_evidence": "Provide pay records, an employer or client statement, or other records showing the earnings rate and income lost.",
    }
    return [required_gap(field, label, "Third-party demand", f"{demand} {explanations[field]}")
            for field, label in INCOME_LOSS_EVIDENCE.items() if field not in attached]


def required_claimant_gaps(details):
    gaps = []
    for gap in details.get("missing", []):
        check = REQUIRED_CHECKS.get(gap.get("field"))
        # Before required_for was persisted, these fields were recorded by the same required checks.
        if check and gap.get("required_for", check) == check and gap.get("document") and gap.get("reason"):
            gaps.append(gap)
    if details.get("coverage_status") not in {"INACTIVE", "NOT_COVERED"} and (assessment := details.get("third_party_assessment")):
        evidence_gaps = [*assessment["liability"]["evidence_requests"],
                         *(request for cost in assessment["costs"] if cost["policy_applicability"] in {"applies", "unresolved"}
                           for request in cost["evidence_requests"])]
        if total_gap := assessment.get("total_evidence_request"):
            evidence_gaps.append(total_gap)
        gaps.extend(gap for gap in evidence_gaps if gap["owner"] == "claimant")
    return gaps


def committed_required_gaps(findings):
    for stage in ("intake", "coverage"):
        details = findings.get(stage, {}).get("details", {})
        if details.get("checks_executed"):
            yield from required_claimant_gaps(details)
