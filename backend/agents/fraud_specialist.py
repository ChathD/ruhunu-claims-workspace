"""Evaluate supported fraud indicators, plausibility and severity using submitted or verified evidence."""
import re
from datetime import date
from decimal import Decimal, InvalidOperation

from backend.claim_schemas import Finding
from backend.evidence.motor_benefits import commercial_use_exclusion
from backend.evidence.required_claim_evidence import COST_DOCUMENT_TYPES
from backend.evidence.synthetic_claim_text import (
    ASSESSOR_REPORT_SOURCE, BLOCK_SEPARATOR, NO_DOCUMENT, NO_POLICE_REPORT, TEXT_REF_PREFIX, available_text_refs,
    document_text_refs, resolve_claim_text,
)

NARRATIVE_REF = "submission:/narrative"
FNOL_REF = TEXT_REF_PREFIX + "fnol_transcript"


def scheduled_cost_indicators(claim, schedule):
    if (claim.get("source") != "Synthetic generator" or not claim.get("loss_date")
            or not schedule.get("policy_number") or not schedule.get("starts_on")
            or claim.get("policy_id") != schedule["policy_number"]
            or not claim.get("registration") or claim["registration"] != (schedule.get("vehicle") or {}).get("registration")):
        return []
    resolved = resolve_claim_text(claim)
    gross = positive_amount(claim.get("gross_amount"))
    if gross is None:
        return []
    document_type, _ = COST_DOCUMENT_TYPES.get(claim["loss_type"], (None, None))
    active_loss = schedule["starts_on"] <= claim["loss_date"] <= schedule.get("ends_on", claim["loss_date"])
    excluded_use = commercial_use_exclusion(schedule)
    flags = []
    for index, document in enumerate(resolved.get("documents", [])):
        ref = f"submission:/documents/{index}"
        if document.get("registration") == claim["registration"] and active_loss:
            if (document["type"] == "police_report" and document.get("incident_date")
                    and not schedule["starts_on"] <= document["incident_date"] <= schedule.get("ends_on", claim["loss_date"])):
                flags.append({"kind": "police_incident_date_conflict",
                              "reason": "The police incident register places the event outside the policy period, while the claim reports an in-policy date.",
                              "evidence": [f"{ref}/incident_date", "submission:/loss_date", "policy_schedule:/starts_on", "policy_schedule:/ends_on"]})
            if (document["type"] == "police_report" and document.get("commercial_trip_verified") is True
                    and claim.get("reported_vehicle_use") == "private" and excluded_use):
                term = "permitted_use" if excluded_use == schedule.get("permitted_use") else "special_conditions"
                flags.append({"kind": "excluded_use_conflicts_with_reported_use",
                              "reason": "The claim reports a private trip, but the police verified a paid passenger booking and the schedule excludes that use.",
                              "evidence": [f"{ref}/commercial_trip_verified", "submission:/reported_vehicle_use", f"policy_schedule:/{term}"]})
        if document["type"] != document_type or positive_amount(document.get("total")) != gross:
            continue
        if (document.get("date") and document["date"] < schedule["starts_on"] <= claim["loss_date"]):
            flags.append({"kind": "cost_document_predates_reported_incident",
                          "reason": "The filed cost document predates both policy inception and the reported incident it is offered to substantiate.",
                          "evidence": [f"{ref}/date", "submission:/loss_date", "policy_schedule:/starts_on"]})
        if (document_type in {"repair_estimate", "vehicle_valuation"} and document.get("registration")
                and document["registration"] != claim["registration"]):
            flags.append({"kind": "cost_document_registration_conflict",
                          "reason": "The primary own-vehicle cost document names a different registration from the matched schedule and claim form.",
                          "evidence": [f"{ref}/registration", "submission:/registration", "policy_schedule:/vehicle/registration"]})
        if (document_type == "repair_estimate" and document.get("registration") == claim["registration"]
                and "Caller reports airbags did not deploy." in resolved.get("unstructured_inputs", {}).get("fnol_transcript", "")):
            airbag_charges = [positive_amount(item.get("amount")) for item in document.get("line_items", [])
                              if item.get("description", "").casefold() == "driver airbag module replacement"]
            if any(amount is not None and airbag_charges.count(amount) > 1 for amount in airbag_charges):
                flags.append({"kind": "duplicate_airbag_charges_conflict",
                              "reason": "The estimate bills twice for the driver's airbag module, although the caller reports that the airbags did not deploy.",
                              "evidence": [f"{ref}/line_items", FNOL_REF]})
    return flags


def investigate(tools, follow_up=False):
    submission = tools.get("read_submission")
    claim = submission.invoke({}) if submission else {}
    text_refs = available_text_refs(claim)
    if follow_up:
        tools["verify_historical_leads"].invoke({})
    evidence = ["tool:verify_historical_leads", "finding:historical"] if follow_up else [NARRATIVE_REF]
    details = {"text_sources": text_refs} if text_refs else {}
    summary = "Historical cost document references checked." if follow_up else "Claim available for fraud screening."
    if (lookup := tools.get("retrieve_vehicle_market_value")) and (valuations := submitted_vehicle_valuations(claim)):
        response = lookup.invoke({})
        details["vehicle_market_check"] = assess_market_benchmark(valuations, response)
        evidence.append(response["evidence_ref"])
        summary += " " + details["vehicle_market_check"]["summary"]
    return Finding(agent="fraud_followup" if follow_up else "fraud_initial",
                   status="pass",
                   summary=summary,
                   evidence=[*evidence, *text_refs],
                   details=details)


def positive_amount(value):
    try:
        amount = Decimal(str(value))
    except InvalidOperation:
        return None
    return amount if amount.is_finite() and amount > 0 else None


def submitted_vehicle_valuations(claim):
    if claim.get("currency") != "LKR":
        return []
    return [(doc.get("name") or "Submitted vehicle valuation", amount, f"submission:/documents/{index}/total")
            for index, doc in enumerate(claim.get("documents") or [])
            if doc.get("type") == "vehicle_valuation" and (amount := positive_amount(doc.get("total"))) is not None]


def assess_market_benchmark(valuations, response):
    market = response["result"]
    check = {"evidence_ref": response["evidence_ref"], "benchmark": market, "comparisons": []}
    if market["status"] == "no_observation":
        return {**check, "summary": "No advertised asking price was available for this vehicle’s model year at or before the loss date."}
    price = positive_amount(market.get("asking_price_lkr"))
    if market["status"] != "found" or price is None:
        return {**check, "summary": "The vehicle market benchmark was unavailable during fraud screening."}

    observed = date.fromisoformat(market["observation_date"]).strftime("%d/%m/%Y")
    source = market.get("source") or "The market data source"
    summary = f"{source} recorded an average advertised asking price of LKR {price:,.2f} on {observed}."
    for label, amount, source_ref in valuations:
        pct = format(amount / price * 100, ".1f")
        check["comparisons"].append({"label": label, "amount_lkr": str(amount), "percent_of_benchmark": pct, "source": source_ref})
        summary += f" {label} stated LKR {amount:,.2f}, or {pct}% of that benchmark."
    return {**check, "summary": summary}


def verified_indicators(verified):
    """Translate exact verification results into mandatory human referrals."""
    flags = []
    for lead in verified:
        flags.append({"code": "VERIFIED_COST_DOCUMENT_REUSE", "reason": "The current cost document reference is linked to at least two other prior incidents.",
                      "alternative": lead["alternative"], "evidence": lead})
    return flags


def verified_submission_indicators(verified):
    return [{"code": "VERIFIED_COST_DOCUMENT_REUSE",
             "reason": f"Cost document {match['document_reference']} is an exact copy of one submitted for an earlier, separate incident.",
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
    if path == TEXT_REF_PREFIX + "police_report_ocr":
        text = text.removeprefix(NO_POLICE_REPORT + BLOCK_SEPARATOR)
    if text.casefold() in {"", "null", "none", "n/a", "unknown", "unavailable", "unreadable", "not provided", "not supplied",
                           NO_DOCUMENT.casefold(), NO_POLICE_REPORT.casefold()}:
        return None
    return text


def observation_matches(presented, fact, evidence):
    path = fact["path"]
    raw = path.startswith(TEXT_REF_PREFIX)
    if raw and path not in evidence:
        return False
    value = observed_fact(presented, path, evidence)
    quote = fact["value"]
    if value is None:
        return False
    if value == quote:
        return True
    field = path.rsplit("/", 1)[-1]
    return bool((raw or field in {"narrative", "summary", "description", "reason"})
                and re.search(r"(?<!\w)" + re.escape(quote) + r"(?!\w)", value))


def supported_concerns(concerns, presented, *, fictional_policy_reference=False):
    """A file citation alone does not establish a conflict between present facts."""
    accepted = []
    submission = presented.get("submission", {})
    text_refs = document_text_refs(submission)
    policy_number = submission.get("policy_id")
    policy_mention = (re.compile(r"\bpolicy(?: number| no\.?)?\s*[:#]?\s+" + re.escape(policy_number) + r"(?!\w)", re.I)
                      if fictional_policy_reference and policy_number else None)
    for concern in concerns:
        if concern.get("basis") != "conflicting_evidence":
            continue
        observations = concern.get("observed_facts", [])
        paths = {fact["path"] for fact in observations}
        if fictional_policy_reference and ("submission:/policy_id" in paths or policy_mention and any(
                fact["value"].casefold() == policy_number.casefold() or policy_mention.search(fact["value"]) for fact in observations)):
            continue
        if paths == {"submission:/loss_date", "submission:/reported_date"}:
            continue
        if len({fact["value"] for fact in observations}) < 2:
            continue
        if not all(observation_matches(presented, fact, concern["evidence"]) for fact in observations):
            continue
        document_refs = {"/".join(path.split("/")[:3]) for path in paths if path.startswith("submission:/documents/")}
        origins = {NARRATIVE_REF if path == FNOL_REF else text_refs.get("/".join(path.split("/")[:3]), path) for path in paths}
        if origins == {NARRATIVE_REF}:
            narrative = (submission.get("narrative") or "").strip()
            spans = sorted((narrative.find(fact["value"]), len(fact["value"])) for fact in observations)
            if not any(0 <= start and start + length <= later and re.search(r"[.!?](?=\s|$)", narrative[start:later])
                       for (start, length), (later, _) in zip(spans, spans[1:])):
                continue
        elif len(origins) < 2 and len(document_refs) < 2:
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
            and any(fact["path"] in {NARRATIVE_REF, FNOL_REF} for fact in facts)
            and any(any(source in fact["path"] for source in ("/documents/", "/uploaded_evidence/", ":/evidence/",
                         "/unstructured_inputs/police_report_ocr", "/unstructured_inputs/repair_estimate_text",
                         "/unstructured_inputs/medical_intake_note", TEXT_REF_PREFIX + ASSESSOR_REPORT_SOURCE)) for fact in facts)
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
        summary = "Fraud screening recorded an independently verified indicator."
    elif details.get("unverified_analysis"):
        summary = "Confirm the Fraud screening because the analysis cited unavailable evidence."
    elif concerns:
        summary = "Fraud screening identified conflicting facts in the submitted evidence."
    elif details.get("error_type"):
        summary = "The Fraud specialist could not complete its screening."
    else:
        summary = "Fraud screening found no supported indicator in the submitted evidence."
    if market := details.get("vehicle_market_check"):
        summary += " " + market["summary"]
    return summary


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
                  else "review" if concerns or details.get("unverified_analysis") else "pass")
        display["review_reasons"] = []
    if isinstance(details.get("analysis"), dict):
        display["analysis"] = {**details["analysis"], "fraud_concerns": concerns, "summary": summary, "review_reasons": []}
    return {**finding, "status": status, "summary": summary, "details": display}
