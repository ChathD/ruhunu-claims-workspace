"""Render motor policy schedules as PDF page previews without serving a PDF file."""
from decimal import Decimal
from html import escape
from io import BytesIO

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, SimpleDocTemplate

from backend.assurance.pdf_preview import pdf_page_previews

BODY = ParagraphStyle("Schedule body", fontName="Helvetica", fontSize=10.5, leading=15,
                      textColor=HexColor("#152637"), spaceAfter=5)
TITLE = ParagraphStyle("Schedule title", parent=BODY, fontName="Helvetica-Bold", fontSize=24, leading=29,
                       spaceAfter=10, keepWithNext=True)
HEADING = ParagraphStyle("Schedule heading", parent=BODY, fontName="Helvetica-Bold", fontSize=13, leading=17,
                         textColor=HexColor("#215b67"), spaceBefore=12, spaceAfter=5, keepWithNext=True)
BENEFIT = ParagraphStyle("Schedule benefit", parent=HEADING, fontSize=11, leading=15, spaceBefore=7)
LIMIT_LABELS = {
    "wording_formula": "Calculated from wording", "market_value": "Market value", "statutory": "Statutory limit",
    "unlimited": "Unlimited", "excluded": "Excluded", "unspecified": "Limit unspecified", "unconfirmed": "Cover unconfirmed",
}
COVER_LABELS = {"comprehensive": "Comprehensive", "third_party": "Third-party only",
                "third_party_fire_theft": "Third-party, fire and theft"}


def _text(value):
    return escape(str(value)).replace("\n", "<br/>")


def _amount(value, currency="LKR"):
    return "Not stated" if value is None else f"{currency} {Decimal(str(value)):,.2f}"


def _add_facts(story, heading, facts):
    story.append(Paragraph(heading, HEADING))
    for label, value in facts:
        story.append(Paragraph(f"<b>{_text(label)}</b> &nbsp; {_text(value if value not in (None, '') else 'Not supplied')}", BODY))


def _saved_schedule(story, schedule):
    vehicle = schedule["vehicle"]
    _add_facts(story, "Policyholder and period", [
        ("Policy number", schedule["policy_number"]), ("Policyholder", schedule["insured_name"]),
        ("Postal address", schedule["insured_address"]), ("District", schedule["district"]),
        ("Postal code", schedule["postal_code"]), ("Issued on", schedule["issued_on"]),
        ("Period of insurance", f'{schedule["starts_on"]} to {schedule["ends_on"]}'),
        ("Cover type", COVER_LABELS[schedule["cover_type"]]),
    ])
    _add_facts(story, "Insured vehicle", [
        ("Registration", vehicle["registration"]), ("Make and model", f'{vehicle["make"]} {vehicle["model"]} ({vehicle["year"]})'),
        ("Class and body", f'{vehicle["vehicle_class"]}, {vehicle["body_type"]}'),
        ("Chassis number", vehicle["chassis_number"]), ("Engine number", vehicle["engine_number"]),
        ("Fuel and capacity", f'{vehicle["fuel"]}, {vehicle["engine_capacity_cc"]} cc'), ("Seats", vehicle["seats"]),
    ])
    story.append(Paragraph("Cover, limits and excesses", HEADING))
    for benefit in schedule["cover"]:
        label = benefit.get("benefit_name") or benefit["loss_type"].replace("_", " ").capitalize()
        basis = benefit["limit_basis"]
        limit = _amount(benefit["limit_lkr"]) if basis in {"scheduled_amount", "wording_amount"} else LIMIT_LABELS[basis]
        story.append(Paragraph(_text(label), BENEFIT))
        story.append(Paragraph(f"<b>Limit</b> &nbsp; {_text(limit)}", BODY))
        if basis != "excluded":
            story.append(Paragraph(f'<b>Compulsory excess</b> &nbsp; {_amount(benefit["compulsory_excess_lkr"])}'
                                   f' &nbsp; <b>Voluntary excess</b> &nbsp; {_amount(benefit["voluntary_excess_lkr"])}', BODY))
        story.append(Paragraph(_text(benefit["conditions"]), BODY))
    _add_facts(story, "Use, drivers and endorsements", [
        ("Territorial limits", schedule["territorial_limits"]), ("Permitted use", schedule["permitted_use"]),
        ("Authorised drivers", schedule["authorised_drivers"]), ("Named drivers", ", ".join(schedule["named_drivers"])),
        ("Endorsements", "\n".join(schedule["endorsements"]) or "None selected"),
        ("Special conditions", "\n".join(schedule["special_conditions"]) or "Standard wording applies"),
    ])
    _add_facts(story, "Premium breakdown", [
        ("Base premium", _amount(schedule["base_premium_lkr"])), ("No-claim discount", _amount(schedule["no_claim_discount_lkr"])),
        *((charge["label"], _amount(charge["amount_lkr"])) for charge in schedule["charges"]),
        ("Total premium", _amount(schedule["total_premium_lkr"])),
    ])
    story.append(Paragraph("Claims instructions", HEADING))
    story.append(Paragraph(_text(schedule["claims_instructions"]), BODY))


def _draft_schedule(story, schedule):
    reported = schedule["reported"]
    _add_facts(story, "Reported policy and vehicle", [
        ("Policy number", reported["policy_number"]), ("Insured name", reported["insured_name"]),
        ("Insured address", reported["insured_address"]), ("Registration", reported["registration"]),
        ("Vehicle", " ".join(str(reported[key]) for key in ("make", "model", "year") if reported.get(key))),
        ("Chassis number", reported["chassis_number"]),
    ])
    _add_facts(story, "Reported loss", [
        ("Loss type", (reported["loss_type"] or "").replace("_", " ").capitalize()), ("Loss date", reported["loss_date"]),
        ("Reported on", reported["reported_date"]),
        ("Claim amount", _amount(reported["claim_amount"], reported["claim_currency"] or schedule["currency"])),
    ])
    story.append(Paragraph("Individual terms to confirm", HEADING))
    story.append(Paragraph("Verify the period of insurance, level of cover, sum insured, excess, premium and endorsements "
                           "against the issued policy schedule.", BODY))


def policy_schedule_preview(schedule):
    saved = "id" in schedule
    wording = schedule["wording"]
    title = f'Motor policy schedule / {schedule["policy_number"]}' if saved else "Draft motor policy schedule"
    story = [Paragraph(_text(title), TITLE)]
    if saved:
        story.append(Paragraph("Synthetic Sri Lankan motor schedule", BODY))
    else:
        story.append(Paragraph(_text(f'{schedule["country"]} / {schedule["currency"]} / Generated {schedule["generated_at"][:10]}'), BODY))
    story.append(Paragraph(f'<b>Policy wording</b> &nbsp; {_text(wording["title"])} ({_text(wording["version"])})', BODY))
    if wording.get("document_kind") == "product_summary":
        story.append(Paragraph("Provisional schedule from a product summary. Confirm terms against the full policy wording.", BODY))
    (_saved_schedule if saved else _draft_schedule)(story, schedule)

    def footer(canvas, document):
        canvas.setFillColor(HexColor("#47596a"))
        canvas.setFont("Helvetica", 9)
        canvas.drawString(48, 27, "Ruhunu Claims")
        canvas.drawRightString(A4[0] - 48, 27, f"Page {document.page}")

    buffer = BytesIO()
    SimpleDocTemplate(buffer, pagesize=A4, leftMargin=48, rightMargin=48, topMargin=42, bottomMargin=48,
                      title=title).build(story, onFirstPage=footer, onLaterPages=footer)
    return {"title": title, "wording": wording, "pages": pdf_page_previews(buffer.getvalue())}
