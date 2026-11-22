"""Check whether the claim has enough evidence for Coverage and record blocking gaps."""
from decimal import Decimal

from backend.claim_schemas import Finding
from backend.evidence.required_claim_evidence import (
    COST_DOCUMENT_TYPES, SUPPORTING_DOCUMENT_LABELS, additional_charge_receipts, additional_charge_totals, has_police_report, required_gap,
)
def cost_evidence(claim):
    documents = claim.get("documents") or []
    document_type, name = COST_DOCUMENT_TYPES.get(claim.get("loss_type"), (None, None))
    matching = [doc for doc in documents if document_type and doc.get("type") == document_type]
    return matching, document_type, name


def cost_breakdown(claim, documents):
    document_total = sum((Decimal(doc["total"]) for doc in documents), Decimal(0))
    additional_total = sum((Decimal(cost["amount"]) for cost in claim.get("additional_costs", [])), Decimal(0))
    return document_total, additional_total


def missing_cost_fields(claim):
    documents, field, document_name = cost_evidence(claim)
    if field is None:
        return []
    if not documents:
        reasons = {
            "vehicle_valuation": "The valuation is needed to establish the vehicle's value and assess the amount claimed under theft cover.",
            "repair_estimate": "An itemised estimate from the garage is needed to link the proposed repairs to the incident and assess the claimed repair cost.",
            "third_party_demand": "The itemised demand is needed to identify the third party and substantiate their claimed costs for the incident.",
        }
        return [required_gap(field, document_name, None, reasons[field])]
    required_fields = {
        "date": ("Document date", "The date is needed to check that the cost document relates to this incident."),
        "total": ("Document total", "The total is needed to reconcile the charges with the amount claimed."),
    }
    missing = []
    for document in documents:
        name = document.get("name") or document.get("id") or document_name
        if field == "repair_estimate":
            for key, label, reason in (
                ("garage_name", "Garage name", "The garage name identifies who prepared the repair estimate."),
                ("garage_address", "Garage address", "The garage address identifies the premises that issued the estimate so staff can verify its source."),
            ):
                if not (document.get(key) or "").strip():
                    missing.append(required_gap(f"repair_estimate.{key}", name, label, reason))
        if not (document.get("registration") or "").strip():
            missing.append(required_gap("cost_document.registration", name, "Vehicle registration",
                                        "The registration links the cost document to the reported vehicle."))
        for key, (label, reason) in required_fields.items():
            if document.get(key) in (None, ""):
                missing.append(required_gap(key, name, label, reason))
        items = document.get("line_items") or []
        if not items or any(item.get("amount") in (None, "") for item in items):
            missing.append(required_gap("line_items", name, "Itemised charges",
                                        "Itemised amounts are needed to reconcile the document total."))
    if all(doc.get("total") not in (None, "") for doc in documents) and claim.get("gross_amount") not in (None, ""):
        document_total, additional_total = cost_breakdown(claim, documents)
        difference = Decimal(claim["gross_amount"]) - document_total
        if difference > 0 and not additional_total:
            missing.append(required_gap("additional_costs", "Claim form", "Additional costs",
                                        f"List each charge and its reason for the LKR {difference:.2f} above the cost documents so we can check whether those charges relate to the incident and are covered."))
    return missing


def investigate(tools):
    claim = tools["read_submission"].invoke({})
    requirements = tools["read_coverage_requirements"].invoke({}) if claim.get("policy_id") else None
    missing = []
    if not claim.get("loss_date"):
        missing.append(required_gap("loss_date", "Claim form", "Loss details / incident date",
                                    "A readable loss date is needed to determine the effective policy period."))
    typed_exclusion = requirements and requirements.get("typed_loss_excluded")
    cost_gaps = [] if typed_exclusion else missing_cost_fields(claim)
    if requirements:
        charge_totals = additional_charge_totals(claim)
        for benefit in requirements.get("charge_receipt_requirements", []):
            category = benefit["category"]
            if category in charge_totals and not additional_charge_receipts(claim, category):
                cost_gaps.append(required_gap("additional_cost_receipt", f'{benefit["name"]} receipt', None,
                                              f'The policy requires a receipt to substantiate the separate {benefit["name"].lower()} charge.'))
    missing.extend(cost_gaps)
    police_report_missing = bool(requirements and requirements.get("police_report_required") and not has_police_report(claim))
    if police_report_missing:
        missing.append(required_gap("police_report", "Police report", None,
                                    "The policy requires a police report for this loss type to corroborate the reported incident and complete the coverage assessment."))
    supporting_gaps = requirements.get("missing_supporting_documents", []) if requirements else []
    for rule in supporting_gaps:
        field = rule["document_type"]
        missing.append(required_gap(field, SUPPORTING_DOCUMENT_LABELS[field], None,
                                    f"The policy requires this document to assess the reported loss. {rule['purpose']}"))
    required_fields = {
        "policy_id": ("Policy number", "The policy number is needed to retrieve the terms that apply to the claim."),
        "registration": ("Vehicle registration", "The registration is needed to match the reported vehicle to the insured vehicle."),
        "loss_type": ("Type of loss", "The type of loss is needed to check which cover applies."),
        "narrative": ("Incident description", "The description is needed to determine what happened and whether the loss and related costs are covered."),
        "gross_amount": ("Amount claimed", "The amount is needed to reconcile the submitted costs and check the policy limit."),
        "currency": ("Claim currency", "The currency is needed to compare the amount claimed with the policy limit."),
    }
    for key, (label, reason) in required_fields.items():
        if claim.get(key) in (None, ""):
            missing.append(required_gap(key, "Claim form", label, reason))
    if requirements and requirements["chassis_required"] and not (claim.get("vehicle") or {}).get("chassis_number"):
        missing.append(required_gap("vehicle.chassis_number", "Claim form", "Vehicle details",
                                    "The chassis number is needed to confirm the reported vehicle matches the vehicle identified in the policy."))
    internal_issue = None
    if requirements and not requirements["policy_available"]:
        internal_issue = "Intake could not confirm a saved policy for the policy ID on the claim. Staff must resolve the policy reference before Coverage can start."
    elif requirements and requirements["missing_policy_terms"]:
        internal_issue = "The saved policy is missing terms needed for Coverage."
    elif requirements and not requirements["wording_ready"]:
        internal_issue = "The manual claim has no indexed policy wording available for Coverage."
    elif claim.get("loss_type") and claim["loss_type"] not in COST_DOCUMENT_TYPES:
        internal_issue = "Staff must assess which cost evidence is required for this loss type."
    gaps = {gap["field"] for gap in missing}
    policy_available = bool(requirements and requirements["policy_available"])
    missing_terms = set(requirements["missing_policy_terms"]) if requirements else set()
    checks = {
        "policy_period": policy_available and missing_terms.isdisjoint({"starts_on", "ends_before"}) and "loss_date" not in gaps,
        "vehicle_match": policy_available and "registration" not in missing_terms and gaps.isdisjoint({"registration", "vehicle.chassis_number"}),
        "loss_and_costs": policy_available and requirements["wording_ready"]
        and claim.get("loss_type") in COST_DOCUMENT_TYPES
        and missing_terms.isdisjoint({"covered_losses", "limit", "deductible", "clause"})
        and not cost_gaps and not police_report_missing and not supporting_gaps
        and gaps.isdisjoint({"loss_type", "narrative", "gross_amount", "currency"}),
    }
    summary = f"Checked required fields in the claim form and {len(claim.get('documents') or [])} structured attachment(s)."
    evidence = ["submission:/narrative", "submission:/documents"]
    if requirements:
        evidence.append("tool:read_coverage_requirements")
        if policy_available:
            summary += " Confirmed the submitted policy ID exists in the policy database."
    if missing:
        summary += f" {len(missing)} document or field requirement(s) outstanding."
    if internal_issue:
        summary += " " + internal_issue
    return Finding(agent="intake", status="review" if internal_issue else "information" if missing else "pass",
                   summary=summary, evidence=evidence,
                   details={"missing": missing, "coverage_readiness": {
                       "ready": all(checks.values()), "checks": checks, "internal_issue": internal_issue,
                       "policy_available": policy_available if requirements else None,
                   },
                            "facts": [{"field": key, "value": claim.get(key), "source": f"submission:/{key}", "provenance": "structured claim form"}
                                      for key in ["policy_id", "loss_type", "loss_date", "registration", "gross_amount", "currency"]]})
