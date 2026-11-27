"""Prepare findings and completed specialist assignments for the claim detail screens."""
import re
from datetime import date
from decimal import Decimal, InvalidOperation

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from backend.agents.coverage_specialist import separate_policy_omissions
from backend.agents.fraud_specialist import high_severity_concerns, screened_finding
from backend.claim_schemas import AUTHORITY, Route
from backend.evidence.required_claim_evidence import committed_required_gaps, required_claimant_gaps
from backend.evidence.synthetic_claim_text import DOCUMENT_TYPES
from backend.evidence.vehicle_market_value import market_cap_only_review, recorded_market_cap_assumption
from backend.storage.sqlite_store import AgentRun, ModelInvocation, ToolCall
from backend.workflow.specialist_context import is_specialist_stage

SELECTED_COVERAGE_REVIEW_CODES = {"SPECIALIST_REVIEW", "POLICY_SCHEDULE_REVIEW"}
PROVISIONAL_PAYABLE = re.compile(
    r"\b(?:(?:net|estimated|preliminary|provisional)\s+)?payable(?:\s+amount)?\s*(?::|\bis\b|\bof\b)"
    r"\s*(?:LKR|Rs\.?)\s*\d[\d,]*(?:\.\d+)?", re.IGNORECASE,
)
OLD_OCR_REVIEW_NOTES = {
    "Check OCR-derived terms against the original PDF.",
    "Confirm the policy wording behind the cited product summary or OCR passage.",
    "Verify the OCR citations against the original PDF before applying a restriction or requesting a policy-specific document.",
}
COVERAGE_CHECK_LABELS = {
    "brand-new parts contribution": "Policyholder’s share of new parts costs",
    "condition of average": "Vehicle market value and any reduction in payment",
}


def specialist_inputs(session, assessment_id):
    stages = {}
    for row in session.scalars(select(ToolCall).where(ToolCall.assessment_id == assessment_id).order_by(ToolCall.id)):
        if is_specialist_stage(row.stage):
            stage = stages.setdefault(row.stage, {"stage": row.stage, "requests": [], "checks": []})
            stage["checks"].append({"id": row.id, "tool": row.tool, "request": row.request, "response": row.response})
    for row in session.scalars(select(ModelInvocation).where(ModelInvocation.assessment_id == assessment_id)
                               .order_by(ModelInvocation.created_at, ModelInvocation.id)):
        if is_specialist_stage(row.stage):
            stage = stages.setdefault(row.stage, {"stage": row.stage, "requests": [], "checks": []})
            stage["requests"].append({"id": row.id, "status": row.status, "request": row.payload})
    return list(stages.values())


def open_coverage_items(details):
    items, _ = separate_policy_omissions(details.get("coverage_items", []), [])
    return [item for item in items if item.get("outcome") == "undetermined" and item.get("item") and item.get("reason")]


def coverage_unconfirmed_reason(details, policy=None):
    if error := details.get("error_type"):
        if error.endswith("Timeout") or error in {"TimeoutException", "TimeoutError"}:
            return "The Coverage check timed out before returning a result."
        if error == "ModelOutputTruncated":
            return "The Coverage response reached its length limit before returning a complete result."
        if details.get("error_message") == "Incomplete model response":
            return "The Coverage check returned an incomplete result."
        return "The Coverage check stopped before it could reach a conclusion."

    wording = details.get("policy_rag") or {}
    status = details.get("coverage_status")
    product_summary = (policy or {}).get("document_kind", wording.get("document_kind")) == "product_summary"
    schedule_unconfirmed = policy is not None and not details.get("schedule_confirmed")
    if status in {"COVERED", "NOT_COVERED", "INACTIVE", "PENDING_ANALYSIS"}:
        return None
    if status not in {"UNRESOLVED", "NEEDS_INFORMATION"} and not (product_summary or schedule_unconfirmed):
        return None
    if product_summary:
        return "The selected PDF is a product summary. Obtain the full policy wording and individual schedule to confirm coverage."
    if gaps := required_claimant_gaps(details):
        documents = list(dict.fromkeys(gap["document"].lower() for gap in gaps))
        return f"Required claim evidence is missing or incomplete, including {', '.join(documents[:3])}."
    if fields := details.get("missing_policy_fields"):
        return f"The saved policy is missing details needed for the coverage check, including {', '.join(field.replace('_', ' ') for field in fields[:3])}."
    reason = (details.get("schedule_restriction") or details.get("coverage_reason") or "").strip()
    if reason and reason not in OLD_OCR_REVIEW_NOTES:
        return reason
    if mismatches := details.get("vehicle_mismatches"):
        return f"The reported vehicle differs from the insured vehicle in {', '.join(mismatches)}."

    reasons = []
    if schedule_unconfirmed:
        reasons.append("No individual policy schedule was confirmed for the reported vehicle and loss date.")
    if details.get("amounts_reconciled") is False:
        reasons.append("The claimed amount and submitted cost records could not be reconciled.")
    if wording.get("applicability") in {"unclear", "irrelevant"}:
        reasons.append("The cited wording could not be matched to the reported loss.")
    elif schedule_unconfirmed and wording.get("cited") is False:
        reasons.append("The retrieved passages did not establish a policy term that applies to the reported loss.")
    if reasons:
        return " ".join(reasons[:2])
    if coverage_unconfirmed_checks(details):
        if details.get("schedule_confirmed") and wording.get("cited") and wording.get("applicability") == "applicable":
            return "The policy schedule is confirmed and the wording applies to this loss. The adjuster still needs to verify the open checks."
        return "The investigation identified coverage checks that still need evidence or verification."
    for issue in details.get("document_inconsistencies", []):
        if issue.get("document_refs") and (reason := issue.get("reason")):
            return reason
    if wording.get("cited") is False:
        return "The retrieved passages did not establish a policy term that applies to the reported loss."
    for note in details.get("review_reasons", []):
        note = note.strip()
        if note not in OLD_OCR_REVIEW_NOTES and coverage_evidence_gap(note):
            return INTERNAL_REVIEW_REFERENCE.sub("", note).strip()
    return "The investigation could not establish coverage and did not record which specific check blocked confirmation."


COVERAGE_STATE_LABELS = {
    "confirmed": "Confirmed", "partially_confirmed": "Partially confirmed", "unconfirmed": "Unconfirmed",
}


def coverage_outcome(details, policy=None):
    """Express contract findings independently from the payment and triage decisions."""
    status = details.get("coverage_status")
    items, omissions = separate_policy_omissions(details.get("coverage_items", []), details.get("policy_omissions", []))
    supported = [item for item in items if item["outcome"] == "covered"]
    open_items = [item for item in items if item["outcome"] != "covered"]
    reason = coverage_unconfirmed_reason(details, policy)
    state = "unconfirmed"
    if details.get("error_type"):
        return {"state": state, "reason": reason}
    if status in {"INACTIVE", "NOT_COVERED"}:
        exclusion = next((item["reason"] for item in open_items if item["outcome"] == "not_covered"), None)
        reason = (details.get("schedule_restriction") or details.get("coverage_reason") or exclusion
                  or ("The reported loss date falls outside the policy period." if status == "INACTIVE"
                      else "The policy excludes the reported loss."))
    elif status == "PENDING_ANALYSIS":
        reason = "The coverage investigation is in progress."
    elif not details.get("schedule_restriction") and (status == "COVERED" or supported):
        missing_schedule = policy is not None and not details.get("schedule_confirmed")
        positive = " ".join(f"{item['item']}. {item['reason']}" for item in supported)
        positive = positive or "The applicable policy terms support the reported loss."
        if open_items or omissions or missing_schedule or status != "COVERED":
            state = "partially_confirmed"
            limits = [f"{item['item']}. {item['reason']}" for item in open_items]
            limits.extend(f"{item['item']}. {item['impact']}" for item in omissions)
            if missing_schedule:
                limits.append("The individual policy schedule still needs confirmation for the insured vehicle and loss date.")
            if status != "COVERED" and not limits:
                limits.append(reason or "Some coverage checks remain open.")
            reason = " ".join([positive, *limits])
        else:
            state, reason = "confirmed", positive
    elif open_items and not details.get("coverage_reason") and not details.get("schedule_restriction"):
        reason = " ".join(f"{item['item']}. {item['reason']}" for item in open_items)
    return {"state": state, "reason": reason or "The recorded checks did not establish coverage."}


def omit_provisional_payable(text):
    return PROVISIONAL_PAYABLE.sub("Payable amount remains unverified", text)


def selected_coverage_summary(details, policy):
    outcome = coverage_outcome(details, policy)
    reason = outcome["reason"]
    if (details.get("financial") or {}).get("payable") is None:
        reason = omit_provisional_payable(reason)
    summary = f"Coverage {COVERAGE_STATE_LABELS[outcome['state']].lower()}. {reason}"
    if assumption := recorded_market_cap_assumption(details):
        summary += " " + assumption
    return summary


def selected_review_summary(context, claim):
    findings = context.get("findings", {})
    coverage = findings.get("coverage", {}).get("details", {})
    fraud = findings.get("fraud_followup") or findings.get("fraud_initial")
    parts = [selected_coverage_summary(coverage, context["selected_policy"])] if coverage else []
    if fraud and fraud["details"].get("checks_executed") and fraud["status"] in {"review", "fraud"}:
        parts.append(screened_finding(fraud, claim)["summary"])
    if history := findings.get("historical", {}).get("details", {}).get("review_assessment"):
        parts.append(history)
    return " ".join(parts) if parts else "Review the submitted evidence and specialist findings."


def staff_justification(decision, context, claim):
    justification = decision.get("justification", "")
    if not context.get("selected_policy") or decision["route"] == Route.INFO:
        return justification
    if decision["route"] == Route.REVIEW and SELECTED_COVERAGE_REVIEW_CODES.intersection(decision.get("reason_codes", ())):
        return selected_review_summary(context, claim)
    return justification.partition("\n\n")[0]


def display_finding(finding, claim, selected_policy=None):
    if finding["agent"] != "coverage":
        return screened_finding(finding, claim)
    details = finding["details"]
    items, omissions = separate_policy_omissions(details.get("coverage_items", []), details.get("policy_omissions", []))
    displayed = {**finding, "details": {**details, "coverage_items": items, "policy_omissions": omissions}}
    if "review_reasons" in details:
        displayed["details"]["review_reasons"] = [reason for reason in details["review_reasons"] if reason.strip() not in OLD_OCR_REVIEW_NOTES]
    displayed["details"]["coverage_outcome"] = coverage_outcome(details, selected_policy)
    if reason := coverage_unconfirmed_reason(details, selected_policy):
        displayed["details"]["coverage_unconfirmed_reason"] = reason
        if checks := coverage_unconfirmed_checks(details):
            displayed["details"]["coverage_unconfirmed_checks"] = checks
    if selected_policy:
        displayed["summary"] = selected_coverage_summary(details, selected_policy)
    return displayed


class AdjusterPriority(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_id: str
    sentence: str
    context_sentence: str | None = Field(default=None, description="Supporting facts when the summary focuses on one priority.")


def ranked_review_facts(claim, findings, schedule=None):
    """Order objective findings that can also be displayed without a model rewrite."""
    intake = findings.get("intake", {}).get("details", {})
    coverage = findings.get("coverage", {}).get("details", {})
    comparison = findings.get("historical", {}).get("details", {}).get("amount_justification", {})
    issues = []
    if any(indicator.get("code") == "VERIFIED_COST_DOCUMENT_REUSE" for finding in findings.values()
           for indicator in finding.get("details", {}).get("fraud_indicators", [])):
        issues.append("A submitted cost document is linked to another incident.")
    if intake.get("duplicate_matches"):
        issues.append("An earlier submission matches this vehicle and reported incident.")
    if coverage.get("coverage_status") == "INACTIVE" and schedule and (loss_date := claim.get("loss_date")):
        start, end = schedule.get("starts_on"), schedule.get("ends_on")
        if start and loss_date < start:
            issues.append(f"The reported incident date of {loss_date} precedes the recorded start date of {start}.")
        elif end and loss_date > end:
            issues.append(f"The reported incident date of {loss_date} follows the recorded end date of {end}.")

    financial = coverage.get("financial", {})
    gross, documents_total, additional_total = (financial.get(key) for key in ("gross", "cost_documents_total", "additional_costs_total"))
    if gross is not None and documents_total is not None and additional_total is not None:
        supported_total = Decimal(documents_total) + Decimal(additional_total)
        if supported_total != Decimal(gross):
            issues.append(f"The claim totals LKR {Decimal(gross):,.2f}, and the submitted costs total LKR {supported_total:,.2f}.")
    elif coverage.get("amounts_reconciled") is False:
        issues.append("The submitted costs do not reconcile to the claimed amount.")
    amount, observed = comparison.get("cost_documents_total"), comparison.get("observed_range")
    if comparison.get("conclusion") == "REQUIRES_REVIEW" and amount is not None and observed:
        low, high = map(Decimal, observed)
        if not low <= Decimal(amount) <= high:
            issues.append(f"The submitted costs are LKR {Decimal(amount):,.2f}, outside the comparable range of LKR {low:,.2f} to LKR {high:,.2f}.")
    for gap in committed_required_gaps(findings):
        document = gap["document"].lower()
        if not POLICY_REFERENCE.search(document) and not re.search(r"[.;:!?]", document):
            issues.append(f"The required {document} is missing or incomplete.")
    return list(dict.fromkeys(issues))


def adjuster_document_labels(claim, coverage):
    labels = dict(coverage.get("submitted_documents") or {})
    for index, document in enumerate(claim.get("documents") or []):
        kind = document.get("type")
        label = DOCUMENT_TYPES.get(kind)
        if not label:
            continue
        if kind == "additional_cost_receipt" and (category := document.get("category")):
            label = f"{category.replace('_', ' ').capitalize()} service receipt"
        facts = []
        if issued := document.get("date"):
            facts.append(f"dated {issued}")
        if document.get("total") is not None and claim.get("currency"):
            try:
                total = Decimal(str(document["total"]))
            except InvalidOperation:
                total = None
            if total is not None and total.is_finite():
                facts.append(f"{claim['currency']} {total:,.2f}")
        if facts:
            label += f" ({', '.join(facts)})"
        if kind == "additional_cost_receipt" and (description := document.get("description")):
            label += f". Service details: {description}"
        labels[f"submission:/documents/{index}"] = label
    return labels


def adjuster_review_candidates(claim, findings, schedule=None):
    """Provide material conflicts and unresolved findings for the model to rank."""
    intake = findings.get("intake", {}).get("details", {})
    coverage = findings.get("coverage", {}).get("details", {})
    issues = ranked_review_facts(claim, findings, schedule)
    documents = adjuster_document_labels(claim, coverage)

    def describe_issue(item, label=""):
        records = list(dict.fromkeys(documents[ref] for ref in item.get("document_refs", []) if documents.get(ref)))
        return label + item["reason"] + (f" Records: {' | '.join(records)}" if records else "")

    issues.extend(describe_issue(item) for item in coverage.get("document_inconsistencies", [])
                  if item.get("document_refs") and item.get("reason"))
    issues.extend(item["reason"] for item in intake.get("fraud_indicators", []))
    for stage in ("fraud_initial", "fraud_followup"):
        if finding := findings.get(stage):
            details = screened_finding(finding, claim)["details"]
            issues.extend(item["reason"] for key in ("fraud_indicators", "fraud_concerns") for item in details.get(key, []))
    issues.extend(describe_issue(item, f"{item['item']}: ") for item in coverage.get("coverage_items", [])
                  if item.get("outcome") in {"undetermined", "not_covered"} and item.get("reason")
                  and (item.get("document_refs") or item.get("passage_refs")))
    for details in (coverage, findings.get("historical", {}).get("details", {})):
        if details.get("checks_executed"):
            issues.extend(reason for reason in details.get("review_reasons", [])
                          if reason.strip() and reason.strip() not in OLD_OCR_REVIEW_NOTES)
    return {f"D{index}": issue for index, issue in enumerate(dict.fromkeys(issues), 1)}


ADJUSTER_TASK = re.compile(
    r"^(?:please\s+)?(?:review|check|confirm|verify|compare|obtain|request|clarify|reconcile|investigate|assess|validate|contact)\b"
    r"|\b(?:should|must|needs? to|recommend(?:s|ed)?|requires? (?:review|verification|confirmation))\b", re.I,
)
POLICY_REFERENCE = re.compile(
    r"\b(?:policy|policies|wording|endorsements?|schedules?|clauses?|coverage|deductibles?"
    r"|sections?\s+(?:[IVX]+|\d+[A-Z]?)|conditions?\s+\d+|W\d{3,}|M\d{2,})\b", re.I,
)
TRAILING_POLICY_CITATION = re.compile(
    r"\s+(?:as required by|under|per) (?:the )?"
    r"(?:section (?:[IVX]+|\d+[A-Z]?)(?: (?:clause|condition) \d+[A-Z]?)?|(?:clause|condition) \d+[A-Z]?)"
    r"(?:\s*\([WM]\d+\))?$", re.I,
)
VAGUE_REVIEW_REFERENCE = re.compile(r"\b(?:the record|the file|(?:the|this) work (?:was|is|had been|has been))\b", re.I)
UNRESOLVED_FACT = re.compile(
    r"\b(?:unclear|uncertain|unconfirmed|unverified|missing|absent|unnamed|cannot be (?:confirmed|verified|established)"
    r"|(?:is |was |has been )?not (?:recorded|stated|provided|confirmed|established|supported)"
    r"|(?:does|do) not (?:establish|show|identify))\b", re.I,
)
UNRESOLVED_QUESTION = re.compile(r"^whether (.+?) requires (?:adjuster |staff )?(?:confirmation|verification)\.?$", re.I)
SETTLEMENT_TOPIC = re.compile(
    r"\b(?:settle(?:ment|d)?|pay(?:able|ment|ing)?|amounts?|charges?|costs?|excess(?:es)?|deductibles?|limits?"
    r"|valuations?|market value|estimates?|fees?|liability|receipts?|invoices?|reimburs\w*)\b", re.I,
)
SETTLEMENT_CHECK = re.compile(r"\b(?:confirm\w*|verif\w*|reconcil\w*|subject to|required)\b", re.I)
SETTLEMENT_GAP = re.compile(
    r"\b(?:no evidence|not been\b|not independently\b|without (?:confirmed|documented|recorded|expressly recording))", re.I,
)
INDEPENDENT_COST_CHECK = re.compile(r"\bindependent(?:ly)? (?:assess\w*|valu\w*|verif\w*)\b", re.I)
INTERNAL_REVIEW_REFERENCE = re.compile(r"\(?(?:submission:/|chunk:|tool:|pdf-page:)[\w:/.-]+\)?", re.I)
REVIEW_DATE = re.compile(r"(\b\d{1,2}/\d{1,2}/\d{4}\b)")
REVIEW_SENTENCE_BREAK = re.compile(r"(?<!\bRs\.)(?<!\bNo\.)(?<=[.!?])\s+|;\s*", re.I)
REVIEW_REQUEST_START = re.compile(r"^(?:(?:staff|the adjuster) should\s+)?(?:confirm|verify) whether\b", re.I)
REVIEW_AMOUNT = re.compile(r"\bLKR \d[\d,]*(?:\.\d+)?")
VEHICLE_VALUATION_TOPIC = re.compile(r"\b(?:market value|(?:vehicle|current|pre-loss) valuation|condition of average)\b", re.I)
VALUATION_GAP = re.compile(r"\b(?:not on file|needed|required|obtain|no (?:dated |current )?valuation)\b", re.I)
REPAIR_APPROVAL = re.compile(r"\bwritten (?:\w+ ){0,2}(?:approval|consent)\b", re.I)
VEHICLE_VALUE_CAP = re.compile(r"\b(?:lower of|lesser of|whichever is less)\b", re.I)
NET_REPAIR_COST = re.compile(rf"{REVIEW_AMOUNT.pattern}\s+net repair cost\b|\bnet repair cost(?: of)?\s+{REVIEW_AMOUNT.pattern}", re.I)
SETTLEMENT_TERM = re.compile(r"\b(?:Clause \d+[A-Z]?|[A-Z]\d{2,})\b", re.I)
SETTLEMENT_EFFECT = re.compile(r"\bto (?:determine|establish) whether\b", re.I)
EXCESS_ALREADY_APPLIED = re.compile(
    r"\b(?:net(?:s|ted)?\s+(?:off|of)|already (?:deducted|applied|included)|self[ -]applied)\b"
    r"[^.]{0,40}\b(?:excess|deductible)\b", re.I,
)
COVERAGE_EVIDENCE_GAP = re.compile(
    r"\b(?:no (?:submitted )?(?:documents?|records?|evidence) (?:confirm|show|establish|identif)\w*"
    r"|could not be (?:confirmed|verified|established)|remains? unresolved)\b", re.I,
)
COVERAGE_REVIEW_PREFIX = re.compile(r"^(?:staff|the adjuster|the insurer) (?:should|must)\s+", re.I)


def coverage_evidence_gap(text):
    return UNRESOLVED_FACT.search(text) or SETTLEMENT_GAP.search(text) or COVERAGE_EVIDENCE_GAP.search(text)


def coverage_unconfirmed_checks(details):
    if details.get("error_type"):
        return []
    checks = [(gap["document"], gap["reason"]) for gap in required_claimant_gaps(details)]
    items = [(item["item"], item["reason"]) for item in open_coverage_items(details)]
    items.extend((item["item"], item["impact"]) for item in details.get("policy_conflicts", [])
                 if item.get("materiality") in {"material", "uncertain"} and item.get("item") and item.get("impact"))
    for label, explanation in items:
        label = re.sub(r"\s+[–—]\s+(?:appendix|section|clause)\b.*$", "", label.strip(), flags=re.I)
        label = COVERAGE_CHECK_LABELS.get(label.casefold(), label)
        text = INTERNAL_REVIEW_REFERENCE.sub("", explanation)
        sentences = [sentence.strip() for sentence in REVIEW_SENTENCE_BREAK.split(text) if sentence.strip()]
        if not sentences:
            continue
        actions = [COVERAGE_REVIEW_PREFIX.sub("", sentence) for sentence in sentences]
        action = next((sentence for sentence in actions if ADJUSTER_TASK.match(sentence)), None)
        gap = next((sentence for sentence in sentences if coverage_evidence_gap(sentence)), None)
        reason = " ".join((gap or action or sentences[0]).split())
        reason = re.sub(r"^however,?\s+", "", reason, flags=re.I)
        reason = re.sub(r"\bM\s*&\s*E\s+vs\.?\s+body\b", "mechanical, electrical, or body parts", reason, flags=re.I)
        reason = reason[0].upper() + reason[1:]
        if not reason.endswith((".", "?", "!")):
            reason += "."
        checks.append((label, reason))
    return [{"label": label, "reason": reason} for label, reason in dict.fromkeys(checks)]


def adjuster_sentence(value):
    text = " ".join((value or "").split()).removesuffix(".")
    text = TRAILING_POLICY_CITATION.sub("", text)
    if (not text or len(text.split()) > 45 or re.search(r"[;:!?]|\.(?!\d)", text)
            or ADJUSTER_TASK.search(text) or POLICY_REFERENCE.search(text) or VAGUE_REVIEW_REFERENCE.search(text)):
        return None
    return text + "."


def select_adjuster_summary(priorities, candidates):
    sentences, used = [], set()
    supporting = None
    for priority in priorities:
        if priority.candidate_id in used or priority.candidate_id not in candidates:
            continue
        sentence = adjuster_sentence(priority.sentence)
        if sentence and sentence not in sentences:
            if not sentences:
                supporting = adjuster_sentence(priority.context_sentence)
            sentences.append(sentence)
            used.add(priority.candidate_id)
            if len(sentences) == 2:
                break
    if len(sentences) == 1 and supporting and supporting != sentences[0]:
        sentences.append(supporting)
    return sentences


def payable_review_facts(coverage):
    if reason := adjuster_sentence(coverage.get("coverage_reason")):
        return [reason]
    for issue in coverage.get("document_inconsistencies", []):
        if issue.get("document_refs") and (reason := adjuster_sentence(issue.get("reason"))):
            return [reason]
    unresolved_item = None
    for item in coverage.get("coverage_items", []):
        if item.get("outcome") != "undetermined" or not item.get("item"):
            continue
        label = item["item"]
        if label[:2].istitle():
            label = label[0].lower() + label[1:]
        subject = adjuster_sentence(f"The claim includes the unresolved {label}")
        unresolved_item = unresolved_item or subject
        for part in re.split(r"(?<=[.!?])\s+", item.get("reason") or ""):
            if question := UNRESOLVED_QUESTION.fullmatch(part):
                part = f"It is unclear whether {question[1]}"
            if UNRESOLVED_FACT.search(part) and (reason := adjuster_sentence(part)):
                return [sentence for sentence in (subject, reason) if sentence]
    for note in coverage.get("review_reasons", []):
        for part in re.split(r"(?<=[.!?])\s+|;\s*", note):
            if UNRESOLVED_FACT.search(part) and (reason := adjuster_sentence(part)):
                return [reason]
    return [unresolved_item] if unresolved_item else []


def plain_review_text(text):
    text = INTERNAL_REVIEW_REFERENCE.sub("", text)
    text = re.sub(r"\b(claimant|insurer|company|insured|policyholder)['’]s\b", r"\1", text, flags=re.I)
    text = re.sub(r"\s*[–—]\s+|:\s+", ", ", text)
    text = text.replace("%", " percent").replace("&", " and ").replace("’", "").replace("'", "")
    text = re.sub(r"\b(\d{4})-(\d{2})-(\d{2})\b", r"\3/\2/\1", text)
    text = "".join(part if index % 2 else re.sub(r"[^\w,.]+|_+", " ", part)
                   for index, part in enumerate(REVIEW_DATE.split(text)))
    text = re.sub(r"\s+([,.])", r"\1", " ".join(text.split()))
    return text.strip(" ,.") + "." if text.strip(" ,.") else ""


def concise_settlement_reason(reason, label=""):
    parts = [part.strip() for part in REVIEW_SENTENCE_BREAK.split(INTERNAL_REVIEW_REFERENCE.sub("", reason.strip()))]
    requests = [part for part in parts if REVIEW_REQUEST_START.match(part)]
    gaps = [part for part in parts if UNRESOLVED_FACT.search(part) or SETTLEMENT_GAP.search(part)]
    checks = [part for part in parts if SETTLEMENT_CHECK.search(part)]
    sentence = (requests or sorted(gaps, key=lambda part: len(part.split())) or checks or parts)[0]
    sentence = re.sub(r"^(?:however,?\s+|staff should\s+|the adjuster should\s+)", "", sentence, flags=re.I)
    sentence = re.sub(r"\([^()]*\)", lambda match: f", {amount[0]}," if (amount := REVIEW_AMOUNT.search(match[0])) else "", sentence)
    sentence = sentence.partition(", as ")[0]
    sentence = plain_review_text(sentence)
    if sentence:
        sentence = sentence[0].upper() + sentence[1:]
    if re.search(r"\brepair estimate\b", sentence, re.I) and INDEPENDENT_COST_CHECK.search(sentence) and SETTLEMENT_GAP.search(sentence):
        sentence += (" An independent assessor checks whether the quoted work matches the damage "
                     "and whether the parts and labour costs are reasonable.")
    subject = plain_review_text(label).removesuffix(".")
    key = re.search(r"\b(?:LKR [\d,.]+|[A-Z]\d{2,}\b|Clause \d+)", subject)
    if not key or key[0].rstrip(".,") in sentence or len(subject.split()) > 10:
        return sentence
    charge_terms = {"costs", "charges", "fees", "damages", "estimate"}
    shared_terms = set(re.findall(r"\w+", subject.lower())) & set(re.findall(r"\w+", sentence.lower()))
    if not (key[0].startswith("LKR ") and shared_terms & charge_terms):
        return f"{subject}. {sentence}".strip()
    return sentence


def settlement_item_reason(item, notes):
    label = item.get("item", "")
    reason = concise_settlement_reason(item["reason"], label)
    if term := SETTLEMENT_TERM.search(label):
        for note in notes:
            if (REVIEW_REQUEST_START.match(note) and (effect := SETTLEMENT_EFFECT.search(note))
                    and term[0].casefold() in {value.casefold() for value in SETTLEMENT_TERM.findall(note)}):
                consequence = plain_review_text(f"This determines whether {note[effect.end():].strip()}")
                return f"{reason} {consequence}"
    return reason


def fraud_identification_brief(findings):
    for stage in ("fraud_followup", "fraud_initial"):
        details = findings.get(stage, {}).get("details", {})
        for issue in (*details.get("fraud_indicators", []), *details.get("fraud_concerns", [])):
            if reason := issue.get("reason"):
                return plain_review_text(REVIEW_SENTENCE_BREAK.split(reason, maxsplit=1)[0])
    return None


def fraud_escalation_brief(findings, claim):
    indicators = [issue for finding in findings.values()
                  for issue in finding.get("details", {}).get("fraud_indicators", []) if issue.get("reason")]
    for issue in indicators or high_severity_concerns(findings, claim):
        reason = issue["reason"]
        evidence = issue.get("evidence")
        if isinstance(evidence, dict) and (reference := evidence.get("document_reference")) and reference not in reason:
            return f"Cost document {reference}. {reason}"
        return reason
    return None


def theft_claim_amount_reason(claim, coverage, schedule):
    if claim.get("loss_type") != "theft" or claim.get("currency") != "LKR" or not coverage.get("schedule_confirmed"):
        return None
    valuations = [document["total"] for document in claim.get("documents", [])
                  if document.get("type") == "vehicle_valuation" and document.get("total") not in (None, "")]
    benefit = next((item for item in (schedule or {}).get("cover", []) if item.get("loss_type") == "theft"), {})
    if len(valuations) != 1 or benefit.get("compulsory_excess_lkr") in (None, ""):
        return None
    try:
        gross = Decimal(str(claim["gross_amount"]))
        valuation = Decimal(str(valuations[0]))
        excess = Decimal(str(benefit["compulsory_excess_lkr"]))
        charges = sum((Decimal(str(item["amount"])) for item in claim.get("additional_costs", [])), Decimal(0))
    except (InvalidOperation, KeyError):
        return None
    if not all(amount.is_finite() for amount in (gross, valuation, excess, charges)) or excess <= 0:
        return None

    reason = (f"The claim form lists LKR {gross:,.2f} and the vehicle valuation lists LKR {valuation:,.2f}. "
              f"Reconcile the claimed total to show whether it already deducts the LKR {excess:,.2f} policy excess")
    if charges > 0:
        reason += f" and whether it includes the separately listed charges of LKR {charges:,.2f}"
    return reason + "."


def vehicle_valuation_gap(text):
    return VEHICLE_VALUATION_TOPIC.search(text) and (coverage_evidence_gap(text) or VALUATION_GAP.search(text))


def repair_approval_gap(text):
    return REPAIR_APPROVAL.search(text) and re.search(r"\b(?:repairs?|dismantl\w*)\b", text, re.I) and coverage_evidence_gap(text)


def vehicle_valuation_reason(claim, schedule, finding):
    vehicle = claim.get("vehicle") or schedule.get("vehicle") or {}
    model = " ".join(str(vehicle[key]) for key in ("year", "make", "model") if vehicle.get(key))
    subject = f"the {model}" if model else "the insured vehicle"
    if registration := claim.get("registration") or vehicle.get("registration"):
        subject += f" ({registration})"
    try:
        loss_date = date.fromisoformat(str(claim.get("loss_date") or ""))
    except ValueError:
        loss_date = None
    when = f" on {loss_date:%d/%m/%Y}" if loss_date else " at the time of the loss"
    reason = f"The adjuster needs the market value of {subject}{when} to calculate the payable amount."
    benefit = next((item for item in schedule.get("cover", []) if item.get("loss_type") == claim.get("loss_type")), {})
    terms = " ".join((finding, benefit.get("conditions") or ""))
    capped = VEHICLE_VALUE_CAP.search(terms) and re.search(r"\b(?:schedule|sum insured)\b", terms, re.I)
    if capped:
        reason += " The policy caps payment at the lower of the scheduled value and the vehicle’s market value on the loss date."
    elif re.search(r"\b(?:condition of average|rateable proportion|underinsurance)\b", terms, re.I):
        reason += " The valuation determines whether underinsurance reduces the payment."
    if net := NET_REPAIR_COST.search(finding):
        before_cap = " before applying that cap" if capped else ""
        reason += f" The recorded net repair cost{before_cap} is {REVIEW_AMOUNT.search(net[0])[0]}."
    return reason


def settlement_review_reasons(context, decision, *, claim=None):
    if not decision or decision.get("payable_amount") is not None:
        return []
    coverage = context.get("findings", {}).get("coverage", {}).get("details", {})
    status = coverage.get("coverage_status")
    if status not in {"COVERED", "UNRESOLVED", "NEEDS_INFORMATION"} or coverage.get("error_type"):
        return []

    if third_party := coverage.get("third_party_assessment"):
        reasons = []
        if third_party["liability"]["status"] == "unconfirmed":
            reasons.append("The claims team must assess the insured’s legal responsibility from the collision evidence.")
        if basis := third_party.get("review_context"):
            reasons.append(basis)
        for cost in third_party["costs"]:
            if cost["amount_status"] == "unverified" and cost["evidence_requests"]:
                reasons.extend(request["reason"] for request in cost["evidence_requests"] if request["owner"] == "claimant")
        if total_gap := third_party.get("total_evidence_request"):
            reasons.append(total_gap["reason"])
        if reasons:
            return list(dict.fromkeys(reasons))[:3]

    notes = [note.strip() for note in coverage.get("review_reasons", [])
             if repair_approval_gap(note) or vehicle_valuation_gap(note)
             or (SETTLEMENT_TOPIC.search(note) and (coverage_evidence_gap(note) or SETTLEMENT_CHECK.search(note)))]
    items = [item for item in coverage.get("coverage_items", [])
             if item.get("outcome") == "undetermined" and (item.get("reason") or "").strip()]
    if recorded_market_cap_assumption(coverage):
        notes = [note for note in notes if not market_cap_only_review(note)]
        items = [item for item in items if not market_cap_only_review(f"{item.get('item', '')} {item['reason']}")]
    reasons = []
    if claim and claim.get("loss_type") in {"repair", "glass", "theft"}:
        valuation_notes = [note for note in notes if vehicle_valuation_gap(note)]
        valuation_items = [item for item in items if vehicle_valuation_gap(f"{item.get('item', '')} {item['reason']}")]
        if valuation_notes or valuation_items:
            evidence = " ".join(dict.fromkeys([*valuation_notes, *(item["reason"] for item in valuation_items)]))
            reasons.append(vehicle_valuation_reason(claim, context.get("policy_schedule") or {}, evidence))
            notes = [note for note in notes if note not in valuation_notes]
            items = [item for item in items if item not in valuation_items]
    reasons.extend(settlement_item_reason(item, notes) for item in items)
    reasons.extend(concise_settlement_reason(item["impact"], item.get("item", "")) for item in coverage.get("policy_conflicts", [])
                   if item.get("materiality") in {"material", "uncertain"} and (item.get("impact") or "").strip())
    if coverage.get("amounts_reconciled") is False:
        reasons.append("The claimed amount and submitted cost records could not be reconciled.")
    if reasons:
        notes = [note for note in notes if INDEPENDENT_COST_CHECK.search(note) or repair_approval_gap(note)]
    for note in notes:
        if EXCESS_ALREADY_APPLIED.search(note):
            reasons.append(theft_claim_amount_reason(claim or {}, coverage, context.get("policy_schedule")) or
                           "Reconcile the claimed total to show which charges and deductions it already includes.")
        else:
            reasons.append(concise_settlement_reason(note))
    reasons = list(dict.fromkeys(filter(None, reasons)))
    if reasons:
        return reasons[:3]
    if status == "COVERED" and context.get("selected_policy"):
        return ["The adjuster must verify the settlement calculation before a payable amount can be recorded."]
    return []


def adjuster_review_brief(claim, context, *, decision=None):
    if not decision or decision["route"] != Route.REVIEW:
        return None
    gross = claim.get("gross_amount")
    exceeds_authority = claim.get("currency") == "LKR" and gross not in (None, "") and Decimal(gross) > AUTHORITY
    if summary := context.get("adjuster_summary"):
        return " ".join(summary[:2])
    findings = context.get("findings", {})
    if facts := ranked_review_facts(claim, findings, context.get("policy_schedule")):
        return " ".join(facts[:2])
    reasons = decision.get("reason_codes", ())
    if "NO_COMPARABLE_HISTORY" in reasons:
        return "No comparable claims were available to benchmark the submitted amount."
    if "UNSUPPORTED_CURRENCY" in reasons:
        return "The submitted amount is in a currency without a verified LKR equivalent."
    coverage = findings.get("coverage", {}).get("details", {})
    if decision.get("payable_amount") is None:
        status = coverage.get("coverage_status")
        if status == "INACTIVE":
            return "The reported loss was outside the effective dates, so a settlement amount was not calculated."
        if status == "NOT_COVERED":
            return adjuster_sentence(coverage.get("coverage_reason")) or "The assessment found the reported loss ineligible for payment."
        if facts := payable_review_facts(coverage):
            return " ".join(facts)
        match status:
            case "UNRESOLVED" | "NEEDS_INFORMATION":
                if context.get("selected_policy") and not coverage.get("schedule_confirmed"):
                    return "The individual cover for the reported vehicle and incident date could not be verified, so the settlement amount is undetermined."
                return "The assessment has yet to establish which part of the reported loss qualifies for payment."
            case "COVERED":
                return "Coverage is confirmed for the reported loss."
    if "GROSS_AUTHORITY" in reasons and exceeds_authority:
        return f"The LKR {Decimal(gross):,.2f} claim exceeds the LKR {AUTHORITY:,.2f} review threshold."
    if decision.get("payable_amount") is None:
        return "The investigation did not identify supported charges from which to calculate a settlement amount."
    return "The findings did not establish a material discrepancy."


def completed_model_assignments(session, assessment, runs=None):
    plans = assessment.context.get("readiness_history", {})
    if not any(plan.get("next_action") == "DISPATCH" for plan in plans.values()):
        return []
    if runs is None:
        runs = session.execute(select(AgentRun.stage, AgentRun.instruction, AgentRun.finding)
                               .where(AgentRun.assessment_id == assessment.id)).all()
    completed = {run.stage: run for run in runs
                 if run.finding["details"].get("checks_executed") and not run.finding["details"].get("error_type")}
    if not completed:
        return []

    invocations = session.execute(select(ModelInvocation.stage, ModelInvocation.response).where(
        ModelInvocation.assessment_id == assessment.id, ModelInvocation.status == "complete",
        ModelInvocation.stage.like("readiness:%")))
    rounds = {}
    for stage, response in invocations:
        number = stage.removeprefix("readiness:")
        plan = plans.get(number, {})
        if (not number.isdecimal() or plan.get("next_action") != "DISPATCH"
                or not isinstance(response, dict) or response.get("stop_reason") != "tool_use"):
            continue
        calls = [block for block in (response.get("content") or [])
                 if isinstance(block, dict) and block.get("type") == "tool_use"]
        tasks = plan["tasks"]
        if len(calls) != 1 or calls[0].get("name") != "assess_readiness":
            continue
        submitted = calls[0].get("input")
        if not isinstance(submitted, dict) or submitted.get("tasks") != tasks:
            continue
        assignments = []
        for task in tasks:
            run = completed.get(f"r{number}:{task['specialist']}")
            if run and run.instruction == task["task"]:
                assignments.append({"specialist": run.finding["agent"], "task": run.instruction})
        if assignments:
            rounds[int(number)] = {"round": int(number), "assignments": assignments}
    return [rounds[number] for number in sorted(rounds)]
