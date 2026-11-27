"""Preserve legacy comparison drafts and apply saved schedules during claim assessment."""
import re
from decimal import Decimal

from sqlalchemy import func, select

from backend.evidence.motor_benefits import fixed_motor_benefit, motor_benefit_excess, towing_excess_unresolved
from backend.evidence.required_claim_evidence import cost_line_total
from backend.evidence.synthetic_claim_text import resolve_claim_text
from backend.storage.sqlite_store import PolicyChunk, now


WORDING_TOPICS = {
    "schedule_conditions": r"period of (?:insurance|cover)|effective date|expiry date|specified in the schedule|stated in the schedule",
    "cover": r"insuring clause|own (?:damage|vehicle)|loss or damage|covered|indemnif\w*|third.party liability|comprehensive",
    "excess_and_limits": r"sum insured|market value|limit of (?:liability|indemnity)|(?:compulsory|voluntary) excess|deductible|\bexcess\b",
    "use_and_territory": r"geographical|territorial|limitations? (?:as to|of) use|permitted use|authori[sz]ed driver|driving licen[cs]e",
    "endorsements": r"endorsements?|optional (?:cover|extension)|additional cover|extensions?|special conditions",
    "exclusions_and_duties": r"exclusions?|not covered|shall not (?:cover|pay|indemnify)|claims? conditions?|notice of (?:loss|accident)|police report|notify",
}
LOSS_TERMS = {
    "repair": r"collision|impact|repair|accidental damage|windscreen|flood",
    "glass": r"windscreen|window glass|glass breakage",
    "theft": r"theft|stolen|burglary|vehicle valuation",
    "third_party_property_damage": r"damage to property|property damage|third.party property",
    "third_party_injury": r"third.party injury|bodily injury|death or injury|medical expenses",
}
EXCESS_ITEM = re.compile(r"^(?:[A-Z]\d{1,4}\s*[:–-]?\s*)?(?:(?:compulsory|voluntary|accidental damage)\s+)*(?:excess|deductible)\b", re.I)


def wording_references(passages, submission, ocr_pages):
    entries = {}
    loss_type = submission.get("loss_type")
    relevance = LOSS_TERMS.get(loss_type)
    for section, terms in WORDING_TOPICS.items():
        matches = (passage for passage in passages if re.search(terms, passage.text, re.IGNORECASE))
        best = max(matches, key=lambda passage: bool(relevance and re.search(relevance, passage.text, re.IGNORECASE)), default=None)
        if best:
            entries.setdefault(best, []).append(section)

    policy_number, registration = submission.get("policy_id"), submission.get("registration")
    if policy_number and registration:
        identity = [re.compile(rf"(?<!\w){re.escape(value)}(?!\w)", re.IGNORECASE) for value in (policy_number, registration)]
        match = next((passage for passage in passages if all(pattern.search(passage.text) for pattern in identity)), None)
        if match:
            entries.setdefault(match, []).append("reported_identifiers")
    if not entries:
        entries[passages[0]] = ["wording"]

    # Keep the indexed passages intact so their qualifiers and headings stay with the wording.
    return [{"sections": sections, "page": passage.page, "passage_id": f"chunk:{passage.id}",
             "text": passage.text, "ocr": passage.page in ocr_pages} for passage, sections in entries.items()]


def draft_policy_schedule(session, claim, document, kind):
    query = select(PolicyChunk).where(PolicyChunk.document_id == document.id)
    if chunk_format := document.embedding.get("chunk_format"):
        query = query.where(PolicyChunk.id.startswith(f"{document.id}:{chunk_format}:"))
    passages = session.scalars(query.order_by(PolicyChunk.page, func.length(PolicyChunk.id), PolicyChunk.id)).all()
    if not passages:
        raise ValueError("The selected PDF has no indexed wording to use for the policy schedule.")
    submission = claim.payload
    vehicle = submission.get("vehicle") or {}
    return {
        "generated_at": now(),
        "status": "draft",
        "claim_payload_hash": claim.payload_hash,
        "country": "Sri Lanka",
        "currency": "LKR",
        "wording": {"id": document.id, "title": document.title, "product": document.product,
                    "version": document.version, "file_hash": document.file_hash, "document_kind": kind},
        "reported": {
            "policy_number": submission.get("policy_id"), "insured_name": submission.get("insured_name"),
            "insured_address": submission.get("insured_address"), "registration": submission.get("registration"),
            "make": vehicle.get("make"), "model": vehicle.get("model"), "year": vehicle.get("year"),
            "chassis_number": vehicle.get("chassis_number"), "loss_type": submission.get("loss_type"),
            "loss_date": submission.get("loss_date"), "reported_date": submission.get("reported_date"),
            "claim_amount": submission.get("gross_amount"), "claim_currency": submission.get("currency"),
        },
        "wording_references": wording_references(passages, submission, set(document.embedding.get("ocr_pages", []))),
    }


def schedule_restriction(schedule, claim):
    if schedule["wording"].get("document_kind") == "product_summary":
        return "UNRESOLVED", "The matched schedule is provisional. Obtain the full policy wording and individual schedule to confirm coverage."
    loss_date = claim.get("loss_date")
    if not loss_date:
        return "UNRESOLVED", "A loss date is needed to check the matched policy period."
    if not schedule["starts_on"] <= loss_date <= schedule["ends_on"]:
        return "INACTIVE", "The reported loss date falls outside the matched schedule’s period of insurance."
    if kind := claim.get("motor_benefit"):
        if fixed_motor_benefit(schedule, kind) is None:
            return "UNRESOLVED", "The requested motor benefit does not match fixed terms on the saved schedule."
        return None
    benefit = next((item for item in schedule["cover"] if item["loss_type"] == claim.get("loss_type")), None)
    if not benefit:
        return "UNRESOLVED", "The reported loss type could not be matched to a scheduled benefit."
    if benefit["limit_basis"] == "excluded":
        return "NOT_COVERED", "The matched schedule excludes this type of loss."
    return None


def apply_matched_schedule(finding, claim, schedule):
    details = finding["details"]
    provisional = schedule["wording"].get("document_kind") == "product_summary"
    details.update(schedule_confirmed=not provisional, policy_schedule_id=schedule["id"], scheduled_policy_number=schedule["policy_number"])
    if restriction := schedule_restriction(schedule, claim):
        status, reason = restriction
        finding.update(status="review", summary=reason)
        details.update(coverage_status=status, schedule_restriction=reason, missing=[], coverage_items=[],
                       financial={"payable": None})
        if assessment := details.get("third_party_assessment"):
            applicability = "excluded" if status in {"INACTIVE", "NOT_COVERED"} else "unresolved"
            for cost in assessment["costs"]:
                cost.update(policy_applicability=applicability, evidence_requests=[])
            assessment["liability"]["evidence_requests"] = []


def fixed_repair_terms(benefit):
    if (benefit["loss_type"] not in {"repair", "glass"}
            or benefit["limit_basis"] not in {"scheduled_amount", "wording_amount"}
            or any(benefit.get(field) is None for field in ("limit_lkr", "compulsory_excess_lkr", "voluntary_excess_lkr"))
            or re.search(r"\b(?:market value|rateable|proportional|condition of average|sub[ -]?limit|contribution|depreciat\w*|wear and tear)\b",
                         benefit.get("conditions", ""), re.I)):
        return None
    return (Decimal(benefit["limit_lkr"]),
            Decimal(benefit["compulsory_excess_lkr"]) + Decimal(benefit["voluntary_excess_lkr"]))


def verify_scheduled_repair(finding, claim, schedule):
    details = finding["details"]
    wording = details.get("policy_rag", {})
    items = details.get("coverage_items", [])
    if (details.get("coverage_status") != "COVERED" or not details.get("schedule_confirmed")
            or not wording.get("cited") or wording.get("applicability") != "applicable"
            or any(details.get(key) for key in ("missing", "deferred_cost_gaps", "review_reasons", "policy_omissions", "document_inconsistencies"))):
        return
    if not (financial := scheduled_financials(claim, schedule)):
        return
    waived_terms = []
    if claim.get("motor_benefit") == "glass_waiver" and Decimal(financial["deductible"]) == 0:
        waived_terms = [item for item in items if item["outcome"] == "not_covered" and EXCESS_ITEM.search(item.get("item", ""))]
    coverage_items = [item for item in items if item not in waived_terms]
    if not coverage_items or any(item["outcome"] != "covered" for item in coverage_items):
        return
    if waived_terms:
        details.update(coverage_items=coverage_items, waived_excess_terms=waived_terms)
    details.update(scheduled_repair_verified=True, financial=financial)
    summary = ("The applicable wording, paid itemised receipt and matched schedule support the motor benefit."
               if claim.get("motor_benefit") else "The applicable wording and matched schedule support the repair. "
               "The itemised estimate and independent physical assessment agree with the claimed amount.")
    finding.update(status="pass", summary=summary)


def scheduled_financials(claim, schedule):
    if (claim.get("source") != "Synthetic generator" or claim.get("currency") != "LKR"
            or claim.get("additional_costs") or claim.get("occasional_driver")
            or claim.get("policy_id") != schedule["policy_number"]
            or claim.get("registration") != schedule["vehicle"]["registration"] or schedule_restriction(schedule, claim)):
        return None
    if claim.get("motor_benefit"):
        return motor_benefit_financials(claim, schedule)
    terms = next((fixed_repair_terms(item) for item in schedule["cover"] if item["loss_type"] == claim["loss_type"]), None)
    if terms is None:
        return
    limit, excess = terms
    gross = Decimal(claim["gross_amount"])
    documents = resolve_claim_text(claim).get("documents", [])
    estimates = [doc for doc in documents if doc["type"] == "repair_estimate"]
    assessments = [doc for doc in documents if doc["type"] == "independent_assessor_report"]
    if len(estimates) != 1 or len(assessments) != 1 or not excess < gross <= limit:
        return
    estimate, report = estimates[0], assessments[0]
    if (any(doc.get("registration") != claim["registration"] or (doc.get("date") or "") < claim["loss_date"]
            for doc in (estimate, report)) or report.get("inspection_basis") != "physical_inspection"
            or report.get("prior_damage_recorded_on")
            or (report.get("inspection_date") or "") < claim["loss_date"]
            or not estimate.get("line_items") or cost_line_total(estimate["line_items"]) != gross
            or Decimal(estimate.get("total") or "0") != gross or Decimal(report.get("assessed_amount_lkr") or "0") != gross):
        return
    return financial_amounts(gross, limit, excess)


def motor_benefit_financials(claim, schedule):
    kind = claim["motor_benefit"]
    benefit = fixed_motor_benefit(schedule, kind)
    if not benefit or (kind == "roadside_towing" and towing_excess_unresolved(schedule, benefit)):
        return
    documents = resolve_claim_text(claim).get("documents", [])
    receipts = [doc for doc in documents if doc["type"] == "motor_benefit_receipt"]
    if len(receipts) != 1:
        return
    receipt = receipts[0]
    gross = Decimal(claim["gross_amount"])
    limit = Decimal(benefit["limit_lkr"])
    excess = motor_benefit_excess(schedule, benefit, kind)
    if (excess is None or not excess < gross <= limit or receipt.get("service") != kind or receipt.get("payment_status") != "paid"
            or receipt.get("registration") != claim["registration"] or (receipt.get("date") or "") < claim["loss_date"]
            or not receipt.get("provider") or not receipt.get("service_location") or not receipt.get("line_items")
            or cost_line_total(receipt["line_items"]) != gross or Decimal(receipt.get("total") or "0") != gross):
        return
    if kind == "motor_medical" and (receipt.get("beneficiary") != schedule["insured_name"]
                                    or not receipt.get("service_code") or receipt.get("date") != claim["loss_date"]):
        return
    if kind == "scheduled_equipment" and not any(
            doc["type"] == "purchase_receipt" and doc.get("date", "") < claim["loss_date"]
            and doc.get("registration") == claim["registration"]
            and doc.get("part_serial") and doc["part_serial"] == receipt.get("part_serial")
            and doc.get("asset_serial") == receipt.get("asset_serial") == schedule["vehicle"]["chassis_number"] for doc in documents):
        return
    if kind in {"scheduled_equipment", "roadside_towing"} and not any(
            doc["type"] == "police_report" and doc.get("registration") == claim["registration"]
            and (kind == "scheduled_equipment" or (doc.get("incident_location") == receipt["service_location"]
                                                    and doc.get("incident_date") == receipt["date"] == claim["loss_date"] and gross == limit))
            for doc in documents):
        return
    if kind == "glass_waiver" and (not all(re.search(r"\b(?:glass|windscreen|windshield)\b", item["description"], re.I)
                                           for item in receipt["line_items"]) or not any(
            doc["type"] == "independent_assessor_report" and doc.get("inspection_basis") == "physical_inspection"
            and doc.get("registration") == claim["registration"] and Decimal(doc.get("assessed_amount_lkr") or "0") == gross
            for doc in documents)):
        return
    return financial_amounts(gross, limit, excess)


def financial_amounts(gross, limit, excess):
    return {"gross": str(gross), "limit": str(limit), "deductible": str(excess),
            "payable": str(gross - excess), "formula": "gross - scheduled excess"}
