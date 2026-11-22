"""Choose one triage route by applying the claim's evidence, fraud, coverage and authority controls."""
from decimal import Decimal

from backend.agents.fraud_specialist import high_severity_concerns, screened_finding
from backend.claim_schemas import AUTHORITY, Decision, Route
from backend.evidence.required_claim_evidence import committed_required_gaps


def information_requests(findings):
    requests = [(gap["document"], gap.get("location"), gap["reason"]) for gap in committed_required_gaps(findings)]
    return [{"document": document, "location": location, "reason": reason}
            for document, location, reason in dict.fromkeys(requests)]


def select_route(claim, context, finding_ids):
    findings = {stage: screened_finding(finding, claim) for stage, finding in context["findings"].items()}
    flags = [indicator for finding in findings.values() for indicator in finding["details"].get("fraud_indicators", [])]
    high_severity = high_severity_concerns(findings, claim)
    intake = findings.get("intake", {})
    coverage = findings.get("coverage", {})
    historical = findings.get("historical", {})
    gross = Decimal(claim["gross_amount"])
    payable = coverage.get("details", {}).get("financial", {}).get("payable")
    readiness_gaps = context.get("readiness_gaps", [])
    missing = information_requests(findings)
    coverage_status = coverage.get("details", {}).get("coverage_status")
    leads = historical.get("details", {}).get("fraud_investigation", {}).get("leads", [])
    required_stages = ["intake", "fraud_initial", "coverage", "historical"]
    if leads:
        required_stages.append("fraud_followup")
    complete = all(findings.get(stage, {}).get("status") == "pass"
                   and findings[stage]["details"].get("checks_executed") for stage in required_stages)
    coverage_details = coverage.get("details", {})
    wording_conflicts = [item for item in coverage_details.get("policy_conflicts", [])
                         if item["materiality"] in {"material", "uncertain"}]
    if context.get("synthetic_wording_document_id"):
        wording = coverage_details.get("policy_rag", {})
        complete = complete and wording.get("cited") and wording.get("applicability") == "applicable"
    if context.get("execution_mode") != "deterministic-tools" and claim.get("source") == "Manual submission":
        complete = complete and coverage.get("details", {}).get("policy_rag", {}).get("cited", False)
        if context.get("policy_search_mode") == "uploaded-corpus":
            complete = complete and coverage.get("details", {}).get("policy_rag", {}).get("applicability") == "applicable"
    if flags:
        route, code, reason = Route.FRAUD, "FRAUD_INDICATOR", "An evidence-backed fraud indicator requires human fraud review; no allegation is treated as proven."
    elif high_severity:
        route, code, reason = Route.FRAUD, "HIGH_SEVERITY_FRAUD_CONCERN", "A high-severity finding from the Fraud specialist requires review by the fraud team."
    elif missing:
        route, code, reason = Route.INFO, "MISSING_EVIDENCE", "Request the information below to complete the required claim checks."
    elif intake.get("details", {}).get("duplicate_matches"):
        route, code, reason = Route.REVIEW, "POSSIBLE_DUPLICATE", "An earlier submission matches this vehicle and incident; an adjuster must reconcile the files."
    elif claim["currency"] != "LKR":
        route, code, reason = Route.REVIEW, "UNSUPPORTED_CURRENCY", "Currency requires human validation; no automated conversion is performed."
    elif gross > AUTHORITY:
        route, code, reason = Route.REVIEW, "GROSS_AUTHORITY", f"Gross claim {gross} LKR exceeds the mandatory 250000.00 LKR threshold before deductibles."
    elif context.get("deadline_exceeded"):
        route, code, reason = Route.REVIEW, "TRIAGE_DEADLINE", "The 15-minute processing budget was exceeded; human handling is required."
    elif context.get("investigation_error"):
        route, code, reason = Route.REVIEW, "INVESTIGATION_FAILURE", "The model investigation did not complete; human review is required."
    elif context.get("readiness_review") or any(g["owner"] == "internal" for g in readiness_gaps):
        route, code, reason = Route.REVIEW, "READINESS_REVIEW", "Specialist readiness requires human handling or unavailable internal records."
    elif wording_conflicts:
        route, code, reason = Route.REVIEW, "POLICY_WORDING_CONFLICT", "The adjuster must reconcile the wording and the saved policy terms. " + " ".join(
            f"{item['item']}. {item['impact']}" for item in wording_conflicts)
    elif any(f["status"] == "review" for f in findings.values()) or coverage_status in {"INACTIVE", "NOT_COVERED", "UNRESOLVED"}:
        route, code, reason = Route.REVIEW, "SPECIALIST_REVIEW", "A specialist identified an unresolved coverage, valuation or operational concern requiring an adjuster."
    elif coverage_status == "NEEDS_INFORMATION":
        route, code, reason = Route.REVIEW, "UNSPECIFIED_EVIDENCE_GAP", "The coverage check did not identify what the claimant needs to supply."
    elif context.get("historical_available") is False:
        route, code, reason = Route.REVIEW, "NO_COMPARABLE_HISTORY", "No eligible staff-approved claims were available for a cost comparison. An adjuster must validate the claimed amount."
    elif (not complete or coverage_status != "COVERED" or payable is None or Decimal(payable) <= 0 or gross <= 0
          or historical.get("details", {}).get("amount_justification", {}).get("conclusion") != "SUPPORTED"):
        route, code, reason = Route.REVIEW, "INCOMPLETE_CHECKS", "Required checks or a supported positive payable are not available."
    else:
        route, code, reason = Route.AUTO, "PROVISIONAL_CHECKS_PASSED", "All configured synthetic checks passed within provisional automated authority. This triage outcome does not authorize payment."
    recommendation = context.get("recommendation")
    proposed = recommendation.get("route") if recommendation else None
    if route == Route.AUTO:
        if proposed == Route.REVIEW.value:
            route, code, reason = Route.REVIEW, "MODEL_REVIEW", "The judge recommends human review based on the recorded investigation."
        elif proposed == Route.FRAUD.value:
            route, code, reason = Route.REVIEW, "UNVERIFIED_SUSPICION", "The judge raised a suspicion without a verified fraud indicator; an adjuster must investigate."
        elif proposed == Route.INFO.value:
            route, code, reason = Route.REVIEW, "UNRESOLVED_MODEL_GAP", "The judge requested more evidence without a validated claimant evidence gap; an adjuster must reconcile it."
        elif not recommendation:
            route, code, reason = Route.REVIEW, "MISSING_RECOMMENDATION", "A completed judge recommendation is required."
    override = reason if recommendation and recommendation.get("route") != route.value else None
    if claim.get("source") == "Manual submission" and route in {Route.AUTO, Route.REVIEW}:
        route, code = Route.REVIEW, "MANUAL_STAFF_DECISION"
        reason = "Manual claim investigation completed; an authenticated staff member must record the final decision. " + reason
        override = reason if recommendation and recommendation.get("route") != route.value else None
    rationale = [reason]
    if high_severity:
        rationale.extend(dict.fromkeys(concern["reason"] for concern in high_severity))
    if missing:
        items = []
        for gap in missing:
            label = gap["document"] + (f" · {gap['location']}" if gap["location"] else "")
            items.append(f"- {label} — {gap['reason']}")
        rationale.append("Missing information and why it is needed\n" + "\n".join(items))
    if route == Route.INFO and claim["currency"] == "LKR" and gross > AUTHORITY:
        rationale.append("An adjuster must oversee this claim because its gross amount exceeds LKR 250,000 before deductibles.")
    if route != Route.INFO:
        rationale.extend(f"{stage}: {f['summary']} Evidence: {', '.join(f['evidence'])}" for stage, f in findings.items())
        rationale.extend(f"{stage}: not assessed — {explanation}" for stage, explanation in context.get("skipped", {}).items())
        if recommendation:
            rationale.append("Judge recommendation: " + recommendation["explanation"])
    return Decision(route=route, justification="\n\n".join(rationale), reason_codes=[code], gross_amount=gross,
                    payable_amount=Decimal(payable) if payable is not None and route != Route.INFO else None,
                    information_required=missing,
                    finding_ids=finding_ids, context_version=context["version"], recommendation=recommendation, override_reason=override)
