"""Check whether the claim has enough evidence for Coverage and record blocking gaps."""
from decimal import Decimal

from backend.claim_schemas import Finding
from backend.evidence.required_claim_evidence import (
    COST_DOCUMENT_TYPES, SUPPORTING_DOCUMENT_LABELS, additional_charge_receipts, additional_charge_totals, has_police_report, required_gap,
)
from backend.evidence.synthetic_claim_text import TEXT_REF_PREFIX, available_text_refs, document_text_refs


def cost_evidence(claim):
    documents = claim.get("documents") or []
    document_type, name = (("motor_benefit_receipt", "Itemised motor benefit receipt") if claim.get("motor_benefit")
                           else COST_DOCUMENT_TYPES.get(claim.get("loss_type"), (None, None)))
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
            "motor_benefit_receipt": "The itemised receipt is needed to substantiate the benefit expense and verify payment.",
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
        if field == "motor_benefit_receipt" and document.get("payment_status") != "paid":
            missing.append(required_gap("motor_benefit_receipt.payment_status", name, "Payment status",
                                        "Provide a paid receipt or other payment confirmation for this reimbursement."))
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
    requirements = (tools["read_coverage_requirements"].invoke({})
                    if claim.get("policy_id") or claim.get("source") == "Synthetic generator" else None)
    missing = []
    if not claim.get("loss_date"):
        missing.append(required_gap("loss_date", "Claim form", "Loss details / incident date",
                                    "A readable loss date is needed to determine the effective policy period."))
    typed_exclusion = requirements and requirements.get("typed_loss_excluded")
    cost_gaps = [] if typed_exclusion else missing_cost_fields(claim)
    selected_policy = (requirements or {}).get("selected_policy")
    deferred_cost_gaps = []
    if selected_policy and not requirements.get("matched_cost_evidence_required"):
        deferred_cost_gaps = cost_gaps
        cost_gaps = []
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
        if key == "policy_id" and selected_policy:
            continue
        if claim.get(key) in (None, ""):
            missing.append(required_gap(key, "Claim form", label, reason))
    internal_issue = None
    if requirements and not requirements["policy_available"]:
        internal_issue = ("The selected policy PDF is unavailable for Coverage." if requirements.get("selected_policy") else
                          "Intake could not confirm a saved policy for the policy ID on the claim. Staff must resolve the policy reference before Coverage can start.")
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
        "vehicle_match": policy_available and "registration" not in missing_terms and "registration" not in gaps,
        "loss_and_costs": policy_available and requirements["wording_ready"]
        and claim.get("loss_type") in COST_DOCUMENT_TYPES
        and missing_terms.isdisjoint({"covered_losses", "limit", "deductible", "clause"})
        and not cost_gaps and not police_report_missing and not supporting_gaps
        and gaps.isdisjoint({"loss_type", "narrative", "gross_amount", "currency"}),
    }
    summary = f"Checked required fields in the claim form and {len(claim.get('documents') or [])} structured attachment(s)."
    evidence = ["submission:/narrative", "submission:/documents"]
    text_evidence = ({"selected_policy": selected_policy, "deferred_cost_gaps": deferred_cost_gaps} if selected_policy else {})
    if claim.get("unstructured_inputs"):
        summary = f"Checked the claim form and fields extracted from {len(claim['documents'])} text attachment(s)."
        refs = available_text_refs(claim)
        evidence.extend(refs)
        text_evidence.update(text_sources={key: "available" if TEXT_REF_PREFIX + key in refs else "absent"
                                           for key in claim["unstructured_inputs"]}, document_text_refs=document_text_refs(claim))
    if requirements:
        evidence.append("tool:read_coverage_requirements")
        if policy_available:
            summary += (" Confirmed the selected policy PDF is indexed." if requirements.get("selected_policy") else
                        " Confirmed the submitted policy ID exists in the policy database.")
        if selected_policy and selected_policy.get("document_kind") == "product_summary":
            summary += " The selected PDF is a product summary. Coverage needs the full wording for a contract conclusion."
    if missing:
        summary += f" {len(missing)} document or field requirement(s) outstanding."
    if internal_issue:
        summary += " " + internal_issue
    return Finding(agent="intake", status="review" if internal_issue else "information" if missing else "pass",
                   summary=summary, evidence=evidence,
                   details={**text_evidence, "missing": missing, "coverage_readiness": {
                       "ready": all(checks.values()), "checks": checks, "internal_issue": internal_issue,
                       "policy_available": policy_available if requirements else None,
                   },
                            "facts": [{"field": key, "value": claim.get(key), "source": f"submission:/{key}", "provenance": "structured claim form"}
                                      for key in ["policy_id", "loss_type", "loss_date", "registration", "gross_amount", "currency"]]})
