"""Check policy and claim facts and validate the payable amount against coverage controls."""
import re
from datetime import date
from decimal import Decimal

from backend.agents.intake_specialist import cost_breakdown, cost_evidence, missing_cost_fields
from backend.claim_schemas import Finding, REQUIRED_POLICY_FIELDS
from backend.evidence.required_claim_evidence import (
    additional_charge_receipts, additional_charge_totals, cost_line_total,
    policy_lists_loss, required_claimant_gaps, required_gap,
)
from backend.evidence.synthetic_claim_text import TEXT_REF_PREFIX, available_text_refs, document_text_refs
from backend.evidence.vehicle_market_value import assess_repair_market_cap, repair_market_cap_applies


POLICY_SILENCE = re.compile(
    r"\b(?:policy|wording|schedule|contract) (?:is silent|does not (?:address|specify|state|mention|say))\b"
    r"|\bno (?:police report required for (?:field|list)|(?:applicable )?(?:clause|term|provision) "
    r"(?:addresses|specifies|requires|mandates|found))\b"
    r"|\b(?:not (?:specified|addressed|stated)|absent) in (?:the )?(?:policy|wording|schedule)\b"
)
CHASSIS_IDENTIFIER = re.compile(r"\b(?:chassis[ _]+(?:number|no|identifier)|vehicle identification number|vin)\b", re.I)


def describes_policy_silence(term):
    normalized = " ".join((term or "").casefold().replace("_", " ").split())
    return bool(POLICY_SILENCE.search(normalized))


def separate_policy_omissions(items, omissions):
    """A policy term that describes silence cannot support a coverage outcome."""
    remaining, gaps = [], list(omissions)
    for item in items:
        if not describes_policy_silence(item.get("policy_term")):
            remaining.append(item)
            continue
        label = item["item"]
        topic = label.casefold().replace("_", " ")
        existing_topics = [gap["item"].casefold().replace("_", " ") for gap in gaps]
        if topic not in existing_topics and not ("police report" in topic
                and any("police report" in existing for existing in existing_topics)):
            gaps.append({"item": label, "impact": item["reason"]})
    return remaining, gaps


def document_vehicle_matches(document, policy):
    return (document.get("registration") or "").strip() == policy["registration"]


def additional_charge_review_reasons(claim, terms):
    benefits = {item["category"]: item for item in terms["eligible_charges"]}
    exclusions = {item["category"]: item for item in terms.get("exclusions", [])}
    reasons = []
    if any(not cost.get("category") for cost in claim.get("additional_costs", [])):
        reasons.append("Match the uncategorised additional charges to the policy schedule." if terms["covered"]
                       else "The repair policy excludes all separately claimed additional charges.")
    for category, amount in additional_charge_totals(claim).items():
        benefit = benefits.get(category)
        if exclusion := exclusions.get(category):
            reasons.append(f"{exclusion['name']} is listed in the repair policy exclusions. {exclusion['reason']}")
            continue
        if benefit is None:
            reasons.append(f"The repair policy does not cover {category.replace('_', ' ')} as an additional charge.")
            continue
        limit, name = benefit["limit_per_incident"], benefit["name"]
        if limit is not None and amount > Decimal(limit):
            reasons.append(f"{name} of LKR {amount:.2f} exceeds the LKR {Decimal(limit):.2f} per-incident limit.")
        receipts = additional_charge_receipts(claim, category)
        if benefit["receipt_required"] and not receipts:
            reasons.append(f"The repair policy requires a receipt for {name.lower()}.")
        if receipts and (sum(Decimal(receipt["total"]) for receipt in receipts) != amount or
                         any(receipt["registration"] != claim["registration"] or receipt["date"] < claim["loss_date"]
                             for receipt in receipts)):
            reasons.append(f"The {name.lower()} receipts do not reconcile with the claimed charge, vehicle or loss date.")
    return reasons


def investigate(tools):
    context = tools["read_context"].invoke({})
    claim = tools["read_submission"].invoke({})
    retrieved = tools["retrieve_policy"].invoke({})
    if selected := retrieved.get("selected_policy"):
        intake = context["findings"].get("intake", {}).get("details", {})
        details = {"coverage_status": "PENDING_ANALYSIS", "missing": [], "selected_policy": selected,
                   "coverage_scope": "selected policy wording", "document_text_refs": document_text_refs(claim),
                   "deferred_cost_gaps": intake.get("deferred_cost_gaps", [])}
        evidence = ["tool:retrieve_policy"]
        if (not claim.get("motor_benefit") and (lookup := tools.get("retrieve_vehicle_market_value"))
                and repair_market_cap_applies(claim, context.get("policy_schedule"))):
            details["settlement_market_cap"] = market_cap = assess_repair_market_cap(claim, lookup.invoke({}))
            evidence.append(market_cap["evidence_ref"])
        return Finding(agent="coverage", status="pass", summary="Prepared the claim for review against the selected policy PDF.",
                       evidence=evidence, details=details)
    policy = retrieved["policy"]
    if policy is None:
        return Finding(agent="coverage", status="review", summary="Authoritative policy record is unavailable; internal review required.",
                       evidence=["tool:retrieve_policy"], details={"coverage_status": "UNRESOLVED",
                       "coverage_reason": "The saved policy could not be retrieved."})
    missing_terms = [name for name in REQUIRED_POLICY_FIELDS if policy.get(name) in (None, "")]
    if missing_terms:
        return Finding(agent="coverage", status="review", summary="Saved policy terms are incomplete.",
                       evidence=["tool:retrieve_policy"], details={"coverage_status": "UNRESOLVED", "missing_policy_fields": missing_terms,
                       "coverage_reason": "The saved policy does not contain all terms needed to evaluate this claim."})
    if not claim.get("loss_date"):
        gap = required_gap("loss_date", "Claim form", "Loss details / incident date",
                           "A readable date of loss is needed to check when the policy was in force.")
        return Finding(agent="coverage", status="information", summary="Coverage cannot be established because the claim-form loss date is unreadable. Provide a clear copy of the incident-date field.",
                       evidence=["submission:/loss_date", "policy:/starts_on", "policy:/ends_before"],
                       details={"coverage_status": "NEEDS_INFORMATION", "missing": [gap],
                                "coverage_reason": gap["reason"]})
    loss_date = date.fromisoformat(claim["loss_date"])
    if not date.fromisoformat(policy["starts_on"]) <= loss_date < date.fromisoformat(policy["ends_before"]):
        reason = f"Loss on {claim['loss_date']} is outside [{policy['starts_on']}, {policy['ends_before']}). The end date is exclusive."
        return Finding(agent="coverage", status="review", summary=reason,
                       evidence=["submission:/loss_date", "tool:retrieve_policy"],
                       details={"coverage_status": "INACTIVE", "policy": policy,
                                "coverage_reason": reason,
                                "coverage_items": [{"item": "Reported loss", "outcome": "not_covered",
                                    "policy_term": f"Cover starts on {policy['starts_on']} and ends before {policy['ends_before']}.",
                                    "reason": f"The loss on {claim['loss_date']} occurred outside the policy period."}]})
    missing = required_claimant_gaps(context["findings"].get("intake", {}).get("details", {}))
    if not policy.get("insurance_type") or policy_lists_loss(policy, claim["loss_type"]):
        for gap in missing_cost_fields(claim):
            if gap not in missing:
                missing.append(gap)
    sources = document_text_refs(claim)
    evidence = ["submission:/loss_date", "tool:retrieve_policy", *dict.fromkeys(sources.values())]
    if (fnol := TEXT_REF_PREFIX + "fnol_transcript") in available_text_refs(claim):
        evidence.append(fnol)
    details = {"coverage_status": "PENDING_ANALYSIS", "policy": policy, "missing": missing}
    if sources:
        details["document_text_refs"] = sources
    return Finding(agent="coverage", status="information" if missing else "pass",
                   summary="Checked required policy fields and the effective period. The loss date is within the policy period.",
                   evidence=evidence, details=details)


def apply_approval_controls(finding, claim, retrieved, conclusion):
    """Apply exact policy and financial gates to the specialist's assessment."""
    if finding.details["coverage_status"] != "PENDING_ANALYSIS":
        return
    policy = retrieved["policy"]
    if claim.get("person_id") and not retrieved.get("policyholder", {}).get("matches"):
        finding.status = "review"
        finding.details["coverage_status"] = "UNRESOLVED"
        finding.details["coverage_reason"] = "The claimant does not match the saved policyholder."
        finding.summary += " The claim's person ID does not match the saved policyholder."
        return
    listed = policy_lists_loss(policy, claim["loss_type"])
    if not listed or conclusion == "NOT_COVERED":
        finding.status = "review"
        finding.details["coverage_status"] = "NOT_COVERED"
        if not listed:
            loss = claim["loss_type"].replace("_", " ")
            terms = f"The policy covers these loss types: {', '.join(policy['covered_losses']).replace('_', ' ')}."
            reason = f"The policy does not list the reported {loss} loss among its covered losses."
            if policy.get("insurance_type") == "third_party" and claim["loss_type"] in {"repair", "theft", "glass"}:
                terms = "Third-party cover applies to the listed liabilities for harm to others. Own-vehicle damage and theft are excluded."
                reason = "The claim concerns the insured vehicle itself, so the own-vehicle exclusion applies."
            finding.details["coverage_reason"] = reason
            finding.details["coverage_items"] = [{"item": "Reported loss", "outcome": "not_covered",
                "policy_term": terms, "reason": reason}]
        finding.summary += f" Coverage is not supported for the declared {claim['loss_type']} incident."
        return
    registration = claim.get("registration")
    if registration != policy["registration"]:
        reason = (f"The claim reports registration {registration}, but the policy identifies {policy['registration']}."
                  if registration else "The claim does not give a vehicle registration to match to the insured vehicle.")
        finding.status = "review"
        finding.details.update(coverage_status="UNRESOLVED", vehicle_mismatches=["registration"], coverage_reason=reason)
        finding.details["coverage_items"] = [{"item": "Insured vehicle identity", "outcome": "undetermined",
            "policy_term": f"The policy identifies the insured vehicle as {policy['registration']}.",
            "reason": reason}]
        finding.summary += f" {reason}"
        return
    if conclusion != "COVERED":
        finding.status = "review"
        finding.details["coverage_status"] = "UNRESOLVED"
        finding.summary += " The specialist could not confirm coverage."
        return
    finding.details["coverage_status"] = "COVERED"
    missing = finding.details["missing"]
    if claim.get("gross_amount") in (None, ""):
        finding.status = "information"
        return
    gross = Decimal(claim["gross_amount"])
    deductible, limit = Decimal(policy["deductible"]), Decimal(policy["limit"])
    reasons = []
    documents, _, _ = cost_evidence(claim)
    if gross <= 0 or gross <= deductible:
        reasons.append("Claim has no positive supported payable above the deductible.")
    if gross > limit:
        reasons.append(f"Gross claim {gross} exceeds policy limit {limit} LKR.")
    if claim["loss_type"] == "repair" and (terms := (policy.get("repair_terms") or {}).get("additional_charges")):
        reasons.extend(additional_charge_review_reasons(claim, terms))
    document_total, additional_total = None, None
    if documents and all(doc.get("total") not in (None, "") for doc in documents):
        document_total, additional_total = cost_breakdown(claim, documents)
    if (claim["loss_type"] == "theft" and (policy.get("theft_terms") or {}).get("basis") == "vehicle_valuation_only"
            and claim.get("additional_costs")):
        reasons.append("Theft cover is limited to the vehicle valuation. The separately claimed additional charges are outside this cover.")
        finding.details["coverage_items"] = [
            {"item": "Vehicle valuation", "outcome": "covered",
             "policy_term": "Theft cover is limited to the vehicle valuation.",
             "reason": "The theft policy covers the vehicle valuation subject to its limit and deductible."},
            {"item": "Separately claimed additional charges", "outcome": "not_covered",
             "policy_term": "Theft cover is limited to the vehicle valuation.",
             "reason": "This theft policy covers the vehicle valuation only."},
        ]
    if not missing:
        if documents and document_total + additional_total != gross:
            reasons.append(f"Cost documents (LKR {document_total:.2f}) and additional costs (LKR {additional_total:.2f}) "
                           f"do not reconcile to the LKR {gross:.2f} claimed.")
        for document in documents:
            if cost_line_total(document["line_items"]) != Decimal(document["total"]):
                reasons.append(f"Document {document['id']} does not reconcile to its line items.")
            if not document_vehicle_matches(document, policy) or document["date"] < claim["loss_date"]:
                reasons.append(f"Document {document['id']} has inconsistent vehicle or incident-date evidence.")
    payable = min(max(gross - deductible, Decimal(0)), limit) if not missing and not reasons and claim["currency"] == "LKR" else None
    finding.status = "review" if reasons else "information" if missing else "pass"
    finding.summary += " " + (" ".join(reasons) if reasons else "Required cost evidence outstanding." if missing else "Financial controls reconciled.")
    finding.details.update(review_reasons=reasons, financial={"gross": str(gross), "deductible": str(deductible), "limit": str(limit),
        "cost_documents_total": str(document_total) if document_total is not None else None,
        "additional_costs_total": str(additional_total) if additional_total is not None else None,
        "payable": str(payable) if payable is not None else None, "formula": "min(max(gross - deductible, 0), limit)"})
