"""Run mandatory checks and apply verified evidence controls in both investigation modes."""
from backend.agents import coverage_specialist as coverage, fraud_specialist as fraud, historical_comparer as historical, intake_specialist as intake
from backend.claim_schemas import THIRD_PARTY_LOSSES


def execute_checks(role, tools, follow_up=False):
    if role == "intake":
        return intake.investigate(tools)
    if role == "coverage":
        return coverage.investigate(tools)
    if role == "historical":
        return historical.investigate(tools)
    if role == "fraud":
        return fraud.investigate(tools, follow_up=follow_up)
    raise ValueError("Unknown specialist")


def apply_verified_controls(role, finding, tools, evidence, schedule=None):
    """Record objective duplicate and fraud signals before any model failure."""
    if role == "intake" and "find_duplicates" in tools:
        claim = evidence["read_submission"]
        identifiers = ("registration", "policy_id", "loss_date") if claim.get("source") == "Manual submission" else ("registration", "incident_id")
        if not all(claim.get(field) for field in identifiers):
            return
        matches = tools["find_duplicates"].invoke({})["matches"]
        finding.details["duplicate_matches"] = matches
        finding.evidence.append("tool:find_duplicates")
        if matches:
            finding.status = "review"
            finding.summary += " An earlier submission matches this vehicle and incident. An adjuster must reconcile the files."
        flags = fraud.scheduled_cost_indicators(claim, schedule) if schedule else []
        if "verify_reused_cost_document" in tools:
            verified = tools["verify_reused_cost_document"].invoke({})["verified"]
            reused = fraud.verified_submission_indicators(verified)
            if reused:
                finding.evidence.append("tool:verify_reused_cost_document")
                flags.extend(reused)
            finding.details["fraud_indicators"] = flags
        if flags:
            finding.details["fraud_indicators"] = flags
            finding.status = "fraud"
            finding.summary += " " + " ".join(flag["reason"] for flag in flags)
    elif role == "fraud":
        flags = fraud.verified_indicators(evidence.get("verify_historical_leads", {}).get("verified", []))
        finding.details["fraud_indicators"] = flags
        if flags:
            finding.status = "fraud"
            finding.summary += " A verified indicator requires human fraud review."


def finish_offline_checks(role, finding, evidence):
    if role == "coverage":
        claim, retrieved = evidence["read_submission"], evidence["retrieve_policy"]
        if selected := retrieved.get("selected_policy"):
            finding.status = "review"
            finding.summary = "The selected policy PDF requires a coverage review. Offline checks could not interpret its wording."
            finding.details.update(coverage_status="UNRESOLVED", policy_rag={"document_id": selected["id"], "cited": False, "retrievals": []})
            return
        liability = claim["loss_type"] in THIRD_PARTY_LOSSES
        coverage.apply_approval_controls(finding, claim, retrieved, "UNRESOLVED" if liability else "COVERED")
        if liability and finding.details["coverage_status"] == "UNRESOLVED":
            finding.details.setdefault("coverage_reason", "Assessing third-party liability requires reviewing evidence of responsibility and the third party's loss.")
        policy = retrieved["policy"] or {}
        if finding.status == "pass" and evidence.get("read_context", {}).get("synthetic_wording_document_id"):
            finding.status = "review"
            finding.details["financial"]["payable"] = None
            finding.details["review_reasons"].append("An adjuster must assess the linked policy wording in offline mode.")
            finding.summary += " The linked policy wording requires review."
        elif finding.status == "pass" and policy.get(f"{claim['loss_type']}_terms"):
            finding.status = "review"
            finding.details["financial"]["payable"] = None
            finding.details["review_reasons"].append("An adjuster must assess item eligibility under the detailed policy terms in offline mode.")
            finding.summary += " The detailed policy terms require an item-by-item coverage review."
    elif role == "historical":
        claim, records = evidence["read_submission"], evidence["retrieve_comparable_claims"]["records"]
        historical.apply_comparison(finding, claim, records, None)
