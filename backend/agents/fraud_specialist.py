"""Evaluate supported fraud indicators, plausibility and severity using submitted or verified evidence."""
import re
from datetime import date

from backend.claim_schemas import Finding

def investigate(tools, follow_up=False):
    if follow_up:
        tools["verify_historical_leads"].invoke({})
    return Finding(agent="fraud_followup" if follow_up else "fraud_initial",
                   status="pass",
                   summary="Historical cost document references checked." if follow_up else "Claim available for fraud screening.",
                   evidence=["tool:verify_historical_leads", "finding:historical"] if follow_up else ["submission:/narrative"],
                   details={})


def verified_indicators(verified):
    """Translate exact verification results into mandatory human referrals."""
    flags = []
    for lead in verified:
        flags.append({"code": "VERIFIED_COST_DOCUMENT_REUSE", "reason": "The current cost document reference is linked to at least two other settled incidents.",
                      "alternative": lead["alternative"], "evidence": lead})
    return flags


def verified_submission_indicators(verified):
    return [{"code": "VERIFIED_COST_DOCUMENT_REUSE",
             "reason": f"Cost document {match['document_reference']} is an exact copy of one submitted for a different vehicle and incident.",
             "alternative": "The unrelated document could have been attached in error.",
             "evidence": match} for match in verified]


def observed_fact(presented, path, evidence):
    """Resolve a visible scalar from the submission or a cited retrieval."""
    source, separator, pointer = path.partition(":/")
    if not separator or source not in presented or (source != "submission" and source not in evidence):
        return None
    value = presented[source]
    for key in pointer.split("/"):
        if key in {"gaps", "status", "error_type", "attachment_id", "claim_id", "incident_id", "provenance", "source"}:
            return None
        if isinstance(value, dict):
            if "attachment_id" in value and value.get("status") != "extracted":
                return None
            value = value.get(key)
        elif isinstance(value, list) and key.isascii() and key.isdecimal() and len(key) <= 4 and int(key) < len(value):
            value = value[int(key)]
        else:
            return None
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        return None
    text = str(value).strip()
    if text.casefold() in {"", "null", "none", "n/a", "unknown", "unavailable", "unreadable", "not provided", "not supplied"}:
        return None
    return text


def observation_matches(presented, fact, evidence):
    value = observed_fact(presented, fact["path"], evidence)
    quote = fact["value"]
    if value == quote:
        return True
    field = fact["path"].rsplit("/", 1)[-1]
    return bool(value and field in {"narrative", "summary", "description", "reason"}
                and re.search(r"(?<!\w)" + re.escape(quote) + r"(?!\w)", value))


def supported_concerns(concerns, presented):
    """A file citation alone does not establish a conflict between present facts."""
    accepted = []
    for concern in concerns:
        if concern.get("basis") != "conflicting_evidence":
            continue
        observations = concern.get("observed_facts", [])
        paths = {fact["path"] for fact in observations}
        if len({fact["value"] for fact in observations}) < 2:
            continue
        if not all(observation_matches(presented, fact, concern["evidence"]) for fact in observations):
            continue
        if len(paths) == 1:
            if paths != {"submission:/narrative"}:
                continue
            narrative = presented["submission"]["narrative"].strip()
            spans = sorted((narrative.find(fact["value"]), len(fact["value"])) for fact in observations)
            if not any(start + length <= later and re.search(r"[.!?](?=\s|$)", narrative[start:later])
                       for (start, length), (later, _) in zip(spans, spans[1:])):
                continue
        accepted.append(concern)
    return accepted


def plausibility_assessment(assessment, concerns, presented, evidence):
    if concerns:
        return {"assessment": "contradicted", "reason": "The submitted evidence contains the conflicting statements described below."}
    if assessment is None:
        return {"assessment": "inconclusive", "reason": "A comparison between the narrative and the attachments is unavailable."}
    facts = assessment["observed_facts"]
    if (assessment["assessment"] == "consistent"
            and any(fact["path"] == "submission:/narrative" for fact in facts)
            and any(any(source in fact["path"] for source in ("/documents/", "/uploaded_evidence/", ":/evidence/")) for fact in facts)
            and all(observation_matches(presented, fact, evidence) for fact in facts)):
        return assessment
    if assessment["assessment"] == "inconclusive" and all(observation_matches(presented, fact, evidence) for fact in facts):
        return assessment
    return {"assessment": "inconclusive", "reason": "The cited evidence was insufficient to confirm the narrative against the documents."}


def legacy_timeline_conflict(concern, claim):
    """Recognize archived timeline concerns when both cited dates can be corroborated."""
    try:
        loss = date.fromisoformat(claim.get("loss_date") or "")
    except ValueError:
        return False
    reason = concern["reason"].casefold()

    def mentioned(value):
        return value.isoformat() in reason or f"{value.day} {value:%B %Y}".casefold() in reason

    if not mentioned(loss):
        return False
    for document in claim.get("documents", []):
        if document.get("type") != "repair_estimate":
            continue
        try:
            estimate = date.fromisoformat(document.get("date") or "")
        except ValueError:
            continue
        if estimate < loss and mentioned(estimate):
            return True
    return False


def evidence_backed_concerns(details, claim=None):
    if not details.get("checks_executed"):
        return []
    concerns = []
    sources = set(details.get("sources", []))
    for concern in details.get("fraud_concerns", []):
        if not concern.get("reason") or not concern.get("evidence") or not sources.issuperset(concern["evidence"]):
            continue
        if details.get("fraud_concerns_validated") is True and concern.get("basis") == "conflicting_evidence":
            # New concerns are recorded only after validating their values against the specialist input.
            concerns.append(concern)
        elif "basis" not in concern and claim is not None and legacy_timeline_conflict(concern, claim):
            concerns.append(concern)
    return concerns


def high_severity_concerns(findings, claim=None):
    concerns = []
    for stage in ("fraud_initial", "fraud_followup"):
        details = findings.get(stage, {}).get("details", {})
        concerns.extend(concern for concern in evidence_backed_concerns(details, claim) if concern["severity"] == "high")
    return concerns


def screening_summary(details, concerns):
    if details.get("fraud_indicators"):
        return "Fraud screening recorded an independently verified indicator."
    if concerns:
        return "Fraud screening identified conflicting facts in the submitted evidence."
    if details.get("error_type"):
        return "The Fraud specialist could not complete its screening."
    return "Fraud screening found no supported indicator in the submitted evidence."


def screened_finding(finding, claim):
    """Project supported fraud concerns from a saved finding."""
    if finding["agent"] not in {"fraud_initial", "fraud_followup"}:
        return finding
    details = finding["details"]
    concerns = evidence_backed_concerns(details, claim)
    display = {**details, "fraud_concerns": concerns}
    status = finding["status"]
    summary = screening_summary(details, concerns)
    if not details.get("error_type"):
        status = ("fraud" if details.get("fraud_indicators") or any(c["severity"] == "high" for c in concerns)
                  else "review" if concerns else "pass")
        display["review_reasons"] = []
    if isinstance(details.get("analysis"), dict):
        display["analysis"] = {**details["analysis"], "fraud_concerns": concerns, "summary": summary, "review_reasons": []}
    return {**finding, "status": status, "summary": summary, "details": display}
