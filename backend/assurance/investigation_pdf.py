"""Render the staff-facing results of a motor claim investigation."""
import re
from decimal import Decimal
from html import escape
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, SimpleDocTemplate

from backend.assurance.investigation_display import (
    COVERAGE_STATE_LABELS, REVIEW_SENTENCE_BREAK, SELECTED_COVERAGE_REVIEW_CODES, coverage_outcome, open_coverage_items,
)
from backend.claim_schemas import Route

ROUTE_LABELS = {
    Route.AUTO: "Auto-approve",
    Route.REVIEW: "Adjuster review",
    Route.INFO: "Request more information",
    Route.FRAUD: "Fraud escalation",
}
SPECIALIST_LABELS = {
    "intake": "Intake",
    "coverage": "Coverage",
    "fraud": "Fraud",
    "fraud_initial": "Fraud",
    "fraud_followup": "Fraud follow-up",
    "historical": "Historical comparison",
}
INK = colors.HexColor("#152637")
ACCENT = colors.HexColor("#215b67")
MUTED = colors.HexColor("#47596a")
RULE = colors.HexColor("#d2dbe1")
BODY = ParagraphStyle("Report body", fontName="Helvetica", fontSize=10.5, leading=15, textColor=INK, spaceAfter=6)
TITLE = ParagraphStyle("Report title", parent=BODY, fontName="Helvetica-Bold", fontSize=28, leading=34,
                       spaceAfter=8, keepWithNext=True)
OUTCOME = ParagraphStyle("Report outcome", parent=BODY, fontName="Helvetica-Bold", fontSize=14, leading=19,
                         textColor=ACCENT, spaceAfter=12, keepWithNext=True)
HEADING = ParagraphStyle("Report heading", parent=BODY, fontName="Helvetica-Bold", fontSize=14, leading=19,
                         spaceBefore=14, spaceAfter=7, keepWithNext=True)
SUBHEADING = ParagraphStyle("Report subheading", parent=BODY, fontName="Helvetica-Bold", spaceBefore=5,
                            spaceAfter=3, keepWithNext=True)
STATUS_LINE = ParagraphStyle("Report status", parent=SUBHEADING, keepWithNext=False)
HYPHENS = str.maketrans("‐‑‒–—", "-----")
POLICY_LOCATOR = r"(?:Section\s+[IVX]+(?:[\s.-]*\d+(?:\([a-z]\))?)?|(?:Clause|Condition)\s+\d+[a-z]?)"
INLINE_SOURCE = r"(?:(?:chunk|pdf-page|tool|submission):[\w:./-]+|\b[WM]\d{2,3}\s*(?:,?\s*lines?\s+\d+(?:\s*[-–—]\s*\d+)?)?)"
PARENTHETICAL_CITATION = re.compile(rf"\s*\((?=[^)]*(?:{INLINE_SOURCE}|{POLICY_LOCATOR}))(?:[^()]|\([^()]*\))*\)", re.I)
ATTRIBUTED_CITATION = re.compile(rf"\s+\b(?:under|per|see|pursuant to|as required by|set out in)\s+{POLICY_LOCATOR}", re.I)
SOURCE_TOKEN = re.compile(INLINE_SOURCE, re.I)
POLICY_CITATION_SUBJECT = re.compile(rf"\b{POLICY_LOCATOR}(?=\s+(?:requires?|covers?|says?|states?|provides?|limits?|applies)\b)", re.I)


def _text(value):
    return escape(str(value).translate(HYPHENS)).replace("\n", "<br/>")


def _money(value, currency="LKR"):
    return f"{currency} {Decimal(str(value)):,.2f}" if value is not None else "Undetermined"


def _report_prose(value):
    text = PARENTHETICAL_CITATION.sub("", value)
    text = ATTRIBUTED_CITATION.sub("", text)
    text = POLICY_CITATION_SUBJECT.sub("The policy", text)
    text = re.sub(rf"\b{POLICY_LOCATOR}(?!\w)", "", text, flags=re.I)
    text = SOURCE_TOKEN.sub("", text)
    return re.sub(r" +([,.;])", r"\1", re.sub(r" {2,}", " ", text)).strip()


LOSS_COVER = re.compile(r"\b(?:accidental|collision|theft|bodily injury|third.party (?:injury|property)|fire|flood|glass)\b", re.I)
AUTHORISED_REPAIRER = re.compile(r"\bauthori[sz]ed (?:agent|repairer)\b", re.I)
USAGE_DEDUCTION = re.compile(r"\busage[ -]deduction\b", re.I)


def _first_sentence(text):
    sentence = REVIEW_SENTENCE_BREAK.split(_report_prose(text or ""), maxsplit=1)[0].strip()
    if sentence and not sentence.endswith((".", "?", "!")):
        sentence += "."
    return sentence


def _coverage_summary(coverage, policy, *, settlement_issue=None):
    if not coverage:
        return "The investigation did not record a coverage finding."
    outcome = coverage.get("coverage_outcome") or coverage_outcome(coverage, policy)
    state = outcome["state"]
    label = COVERAGE_STATE_LABELS[state]
    if state == "unconfirmed":
        reason = _first_sentence(outcome["reason"])
        if (coverage.get("coverage_status") == "UNRESOLVED" and not coverage.get("coverage_reason")
                and not coverage.get("schedule_restriction") and (open_items := open_coverage_items(coverage))):
            sentences = REVIEW_SENTENCE_BREAK.split(_report_prose(open_items[0]["reason"]))
            reason = " ".join(filter(None, (_first_sentence(sentence) for sentence in sentences[:2])))
        return " ".join(filter(None, (label + ".", reason)))

    items = coverage.get("coverage_items", [])
    supported = [item for item in items if item.get("outcome") == "covered"]
    loss = next((item for item in supported if LOSS_COVER.search(item.get("item", ""))), None)
    if loss:
        description = _report_prose(loss["item"]).strip(" –—-")
        description = re.sub(r"^accidental external means\b", "accidental damage", description, flags=re.I)
        primary = f"The evidence supports cover for {description[0].lower() + description[1:]}." if description else ""
    else:
        primary = _first_sentence(supported[0]["reason"]) if supported else _first_sentence(outcome["reason"])

    limits = []
    if state == "partially_confirmed":
        limits = [item["reason"] for item in open_coverage_items(coverage)]
        omissions = coverage.get("policy_omissions", [])
        if omissions and not limits and omissions[0] != settlement_issue and omissions[0].get("impact"):
            limits.append(omissions[0]["impact"])
        if not limits and policy and not coverage.get("schedule_confirmed"):
            limits = ["The individual policy schedule still needs confirmation for the reported vehicle and loss date."]
    limitation = _first_sentence(limits[0]) if limits else ""
    return " ".join(dict.fromkeys(filter(None, (label + ".", primary, limitation))))


def _repairer_settlement_issue(decision, coverage):
    if not decision or "POLICY_SCHEDULE_REVIEW" not in decision.get("reason_codes", ()):
        return None
    return next((item for item in coverage.get("policy_omissions", [])
                 if AUTHORISED_REPAIRER.search(item.get("item", "")) and USAGE_DEDUCTION.search(item.get("impact", ""))), None)


def _historical_summary(finding, currency):
    comparison = finding.get("details", {}).get("amount_justification", {})
    count, median = comparison.get("sample_size"), comparison.get("median")
    if count and median is not None:
        documents = "document" if count == 1 else "documents"
        return f"{count} comparable cost {documents} with a median total of {_money(median, currency)}."
    return _first_sentence(finding.get("summary"))


def _decision_summary(decision, requests, *, fraud_reason=None, adjuster_reason=None, settlement_reasons=(),
                      settlement_issue=None):
    sentences = []
    if decision["route"] == Route.INFO:
        gaps = [_report_prose(request["reason"]) for request in requests if request.get("reason")]
        sentences = [REVIEW_SENTENCE_BREAK.split(gap, maxsplit=1)[0] for gap in gaps if gap]
    if not sentences:
        reason = decision.get("justification", "")
        if decision["route"] == Route.FRAUD and fraud_reason:
            reason = fraud_reason
        if decision["route"] == Route.REVIEW:
            codes = decision.get("reason_codes", ())
            review_codes = SELECTED_COVERAGE_REVIEW_CODES | {"MODEL_REVIEW", "INCOMPLETE_CHECKS"}
            if settlement_issue:
                reason = ("The adjuster must confirm whether the repairer is authorised under the policy. "
                          "This affects whether a deduction for prior use applies to replacement parts.")
            elif "POLICY_SCHEDULE_REVIEW" in codes and settlement_reasons:
                reason = settlement_reasons[0]
            elif adjuster_reason and (not reason or review_codes.intersection(codes)):
                reason = adjuster_reason
        justification = _report_prose(reason.strip().partition("\n\n")[0])
        sentences = REVIEW_SENTENCE_BREAK.split(justification) if justification else []
    return " ".join(sentence if sentence.endswith((".", "?", "!")) else sentence + "."
                    for sentence in list(dict.fromkeys(sentences))[:2])


def render_investigation_pdf(claim, wording_url=None):
    """Render one completed or interrupted investigation from its staff detail."""
    reference = claim["reference"]
    submission = claim["submission"]
    currency = submission.get("currency") or "LKR"
    policy = claim["selected_policy"]
    schedule = claim["policy_schedule"] or {}
    decision = claim["triage"]
    findings = claim["findings"]
    coverage = findings.get("coverage", {}).get("details", {})
    settlement_issue = _repairer_settlement_issue(decision, coverage)

    outcome = ROUTE_LABELS[decision["route"]] if decision else "Investigation interrupted"
    story = [
        Paragraph("Investigation report", TITLE),
        Paragraph(f"{reference} &nbsp; / &nbsp; {outcome}", OUTCOME),
    ]

    def add(value, style=BODY):
        if value:
            story.append(Paragraph(_text(_report_prose(str(value))), style))

    add("Claim details", HEADING)
    policy_number = schedule.get("policy_number") or submission.get("policy_id")
    loss_type = submission.get("loss_type")
    for label, value in (
        ("Policyholder", submission.get("insured_name")),
        ("Motor policy number", policy_number),
        ("Vehicle registration", submission.get("registration")),
        ("Loss type", loss_type.replace("_", " ").capitalize() if loss_type else None),
        ("Loss date", submission.get("loss_date")),
        ("Claimed amount", _money(submission.get("gross_amount"), currency)),
    ):
        if value:
            story.append(Paragraph(f"<b>{label}</b> &nbsp; {_text(value)}", BODY))
    if policy:
        document = _text(f"{policy['title']} ({policy['version']})")
        if wording_url:
            document = f'<link href="{escape(wording_url)}" color="#215b67">{document}</link>'
        story.append(Paragraph(f"<b>Policy wording</b> &nbsp; {document}", BODY))

    add("Decision summary", HEADING)
    summary = ""
    if decision:
        story.append(Paragraph(f"<b>Payable amount</b> &nbsp; {_money(decision.get('payable_amount'), currency)}", BODY))
        summary = _decision_summary(decision, claim["information_required"], fraud_reason=claim["fraud_identification"],
                                    adjuster_reason=claim["adjuster_review"], settlement_reasons=claim["settlement_review_reasons"],
                                    settlement_issue=settlement_issue)
        add(summary or "The recorded decision has no accompanying rationale.")
    else:
        add(_first_sentence(claim.get("processing_error")) or "The investigation stopped before a routing decision was recorded.")

    if requests := claim["information_required"]:
        add("Information requested", HEADING)
        for request in requests:
            add(request.get("document"), SUBHEADING)
            reason = _first_sentence(request.get("reason"))
            if reason and reason not in summary:
                add(reason)

    if findings:
        add("Specialist findings", HEADING)
        for stage, name in SPECIALIST_LABELS.items():
            if not (finding := findings.get(stage)):
                continue
            style = SUBHEADING if stage in {"coverage", "historical"} else STATUS_LINE
            add(f"{name} / {finding['status'].capitalize()}", style)
            if stage == "coverage":
                add(_coverage_summary(coverage, policy, settlement_issue=settlement_issue))
            elif stage == "historical":
                add(_historical_summary(finding, currency))

    historical = findings.get("historical", {}).get("details", {})
    if incidents := historical.get("review_history"):
        add("Similar incidents and decisions", HEADING)
        for incident in incidents:
            add(" / ".join(filter(None, (incident.get("claim_reference"), incident.get("loss_date")))), SUBHEADING)
            if triage := incident.get("triage_decision"):
                route = triage["route"]
                add(f"Published triage / {ROUTE_LABELS.get(route, route.capitalize())}")
            if review := incident.get("staff_decision"):
                add(f"Staff decision / {review['outcome'].replace('_', ' ').capitalize()}")

    def page_header(canvas, document):
        width, height = document.pagesize
        left, right = document.leftMargin, width - document.rightMargin
        canvas.setFont("Helvetica-Bold", 9)
        canvas.setFillColor(ACCENT)
        canvas.drawString(left, height - 39, "RUHUNU CLAIMS")
        canvas.setStrokeColor(RULE)
        canvas.line(left, height - 48, right, height - 48)
        canvas.line(left, 42, right, 42)
        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(MUTED)
        canvas.drawString(left, 27, reference)
        canvas.drawRightString(right, 27, f"Page {document.page}")

    buffer = BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=54, rightMargin=54, topMargin=68,
                                 bottomMargin=57, title=f"{reference} Investigation report", author="Ruhunu Claims")
    document.build(story, onFirstPage=page_header, onLaterPages=page_header)
    return buffer.getvalue()
