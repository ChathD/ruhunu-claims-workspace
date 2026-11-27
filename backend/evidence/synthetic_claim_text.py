"""Render and extract the text sources used in synthetic claim submissions."""
import random
from datetime import date
from decimal import Decimal

from backend.evidence.required_claim_evidence import INCOME_LOSS_EVIDENCE, SUPPORTING_DOCUMENT_LABELS


DOCUMENT_TYPES = {
    "repair_estimate": "Motor repair quotation",
    "vehicle_valuation": "Pre-loss vehicle valuation",
    "third_party_demand": "Third-party claim for costs",
    "independent_assessor_report": "Independent assessor report",
    "additional_cost_receipt": "Service receipt",
    "motor_benefit_receipt": "Motor benefit invoice",
    "purchase_receipt": "Original equipment purchase receipt",
    "police_report": "Police Traffic Branch extract",
    **SUPPORTING_DOCUMENT_LABELS,
    **INCOME_LOSS_EVIDENCE,
}
DOCUMENT_FIELDS = {
    "id": "Filing ID",
    "name": "Filename",
    "document_reference": "Reference",
    "registration": "Vehicle registration",
    "date": "Issued on",
    "incident_date": "Incident recorded on",
    "incident_location": "Incident location",
    "inspection_date": "Inspected on",
    "prior_damage_recorded_on": "Earlier condition record",
    "commercial_trip_verified": "Paid passenger booking found",
    "inspection_basis": "Assessment basis",
    "garage_name": "Garage",
    "garage_address": "Garage address",
    "station": "Police station",
    "author": "Prepared by",
    "chassis_number": "Chassis number",
    "category": "Service category",
    "service": "Service",
    "provider": "Provider",
    "service_location": "Service location",
    "payment_status": "Payment status",
    "beneficiary": "Patient or beneficiary",
    "service_code": "Treatment code",
    "asset_serial": "Asset serial number",
    "part_serial": "Fitted part serial number",
    "total": "Total",
    "assessed_amount_lkr": "Assessed amount",
    "summary": "Report",
    "description": "Service details",
    "source": "Source",
    "provenance": "Provenance",
}
BLOCK_SEPARATOR = "\n\n--- Next document ---\n\n"
NO_DOCUMENT = "No document submitted."
NO_POLICE_REPORT = "No police report submitted."
TEXT_REF_PREFIX = "submission:/unstructured_inputs/"
INCIDENT_REPORT_TYPES = {"police_report", *SUPPORTING_DOCUMENT_LABELS}
ASSESSOR_REPORT_SOURCE = "independent_assessor_report"
DOCUMENT_SOURCES = ("police_report_ocr", "repair_estimate_text", ASSESSOR_REPORT_SOURCE)
SOURCE_BY_DOCUMENT_TYPE = {**dict.fromkeys(DOCUMENT_TYPES, "repair_estimate_text"),
                           **dict.fromkeys(INCIDENT_REPORT_TYPES, "police_report_ocr"),
                           "independent_assessor_report": ASSESSOR_REPORT_SOURCE}
DOCUMENT_DATE_FIELDS = {"date", "incident_date", "inspection_date", "prior_damage_recorded_on"}
DOCUMENT_AMOUNT_FIELDS = {"total", "assessed_amount_lkr"}
FNOL_MARKER = "\n[00:15] Caller\n"
CALL_END = "\n[End of call]"


def _noisy(text, rng, level):
    if level <= 0:
        return text
    substitutions = {"0": "O", "1": "l", "5": "S", "8": "B", " ": ""}
    output = []
    for char in text:
        if rng.random() < level * 0.15:
            char = substitutions.get(char, char.swapcase() if char.isascii() and char.isalpha() else char)
        output.append(char)
    result = "".join(output)
    return result.replace("m", "rn", 1) if rng.random() < level else result


def _local_date(value):
    return date.fromisoformat(value).strftime("%d/%m/%Y")


def _money(value):
    return f"LKR {Decimal(value):,f}"


def _single_line(value):
    return value.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")


def _document_text(document, rng, noise, initial_estimate=None):
    lines = [DOCUMENT_TYPES[document["type"]], _noisy("Scanned copy filed with motor claims", rng, noise)]
    total = document.get("total")
    if initial_estimate is not None and (total is None or Decimal(initial_estimate) != Decimal(total)):
        lines.append(f"Original quotation: {_money(initial_estimate)}.")
    for field, label in DOCUMENT_FIELDS.items():
        if field not in document:
            continue
        value = document[field]
        if value is None:
            value = "[blank]"
        elif field in DOCUMENT_DATE_FIELDS:
            value = _local_date(value)
        elif field in DOCUMENT_AMOUNT_FIELDS:
            value = _money(value)
        elif field == "inspection_basis":
            value = value.replace("_", " ").capitalize()
        elif field == "commercial_trip_verified":
            value = "Yes" if value else "No"
        elif field == "summary" and SOURCE_BY_DOCUMENT_TYPE[document["type"]] != "repair_estimate_text":
            value = _noisy(value, rng, noise)
        value = _single_line(value)
        lines.append(f"{label}: {value}")
    if document.get("line_items"):
        lines.append("Itemised charges:")
        lines.extend(f"  {_money(item['amount'])} | {_single_line(item['description'])}" for item in document["line_items"])
    return "\n".join(lines)


def render_claim_text(claim, person, details, noise_level):
    rng = random.Random(claim["claim_id"])
    fnol = [
        _noisy("Motor claims hotline | first notification of loss", rng, noise_level * 0.4),
        f"[00:04] Agent: May I confirm your name and vehicle number?\n[00:08] Caller: {person['full_name']}, {claim['registration']}.",
    ]
    if claim["loss_date"]:
        fnol.append(f"[00:11] Agent: I've recorded the incident date as {_local_date(claim['loss_date'])}.")
    if details.get("weather"):
        fnol.append(f"Caller on road conditions: {details['weather']}.")
    if details.get("collision_type"):
        fnol.append(f"Caller's description of impact: {details['collision_type'].replace('_', ' ')}.")
    if details.get("other_party"):
        fnol.append(f"Other party named by caller: {details['other_party']}.")
    if details.get("airbags_deployed") is not None:
        fnol.append("Caller reports airbags " + ("deployed." if details["airbags_deployed"] else "did not deploy."))
    if details.get("ems_at_scene") is not None:
        fnol.append("Caller reports " + ("an ambulance attended." if details["ems_at_scene"] else "no ambulance attended."))
    injuries = details.get("claimed_injuries")
    if injuries and not details.get("medical_intake_filed"):
        fnol.append(f"Caller reports injuries: {', '.join(injuries)}.")

    filings = {source: [] for source in DOCUMENT_SOURCES}
    for document in claim["documents"]:
        source = SOURCE_BY_DOCUMENT_TYPE[document["type"]]
        noise = noise_level * 0.8 if source == "repair_estimate_text" else noise_level
        initial = details.get("initial_repair_estimate_lkr") if document["type"] == "repair_estimate" else None
        filings[source].append(_document_text(document, rng, noise, initial))
    sources = {"fnol_transcript": "\n".join(fnol) + FNOL_MARKER + claim["narrative"] + CALL_END}
    sources.update({source: BLOCK_SEPARATOR.join(blocks) for source, blocks in filings.items() if blocks})
    if details.get("medical_intake_filed"):
        diagnoses = ", ".join(details.get("icd10_codes", []))
        patient = (details.get("other_party") or "Third-party claimant") if claim["loss_type"] == "third_party_injury" else person["full_name"]
        intake = f"Patient: {patient}"
        if injuries:
            intake += f"\nPresenting complaints: {', '.join(injuries)}"
        if diagnoses:
            intake += f"\nICD-10 entries: {diagnoses}"
        if details.get("medical_facility"):
            intake = f"{details['medical_facility']}\n{intake}"
        if details.get("medical_intake_date"):
            intake += f"\nSeen on: {_local_date(details['medical_intake_date'])}"
        sources["medical_intake_note"] = _noisy(intake, rng, noise_level * 0.7)

    if details.get("adjuster_note_filed"):
        notes = [f"{_local_date(claim['reported_date'])} | Motor claims desk diary"]
        if policy_id := claim.get("policy_id"):
            notes.append(f"Policy {policy_id}.")
        notes.append(f"Customer reports {_money(claim['gross_amount'])} claimed.")
        if details.get("law_firm"):
            notes.append(f"Letter from Attorney-at-Law: {details['law_firm']}.")
        if details.get("estimated_impact_speed_kmh"):
            notes.append(f"Estimated impact speed noted on file: {details['estimated_impact_speed_kmh']} km/h.")
        if details.get("liability_split_insured_pct") is not None:
            notes.append(f"Insured driver's share recorded in case correspondence: {details['liability_split_insured_pct']}%.")
        if details.get("at_fault_party"):
            parties = {"insured": "insured driver", "other_party": "other party", "shared": "both drivers"}
            notes.append(f"Case correspondence attributes responsibility to the {parties[details['at_fault_party']]}.")
        sources["adjuster_notes"] = _noisy("\n".join(notes), rng, noise_level * 0.3)
    return sources


def extract_claim_text(inputs):
    """Read the simulated form fields and line items without interpreting the case outcome."""
    types = {title: kind for kind, title in DOCUMENT_TYPES.items()}
    fields = {label: key for key, label in DOCUMENT_FIELDS.items()}
    documents = []
    for source in DOCUMENT_SOURCES:
        for block in inputs.get(source, "").split(BLOCK_SEPARATOR):
            title, _, text = block.partition("\n")
            if title not in types:
                continue
            document = {"type": types[title]}
            for line in text.splitlines():
                if line == "Itemised charges:":
                    document["line_items"] = []
                elif line.startswith("  LKR ") and "line_items" in document:
                    amount, _, description = line[6:].partition(" | ")
                    document["line_items"].append({"description": description, "amount": amount.replace(",", "")})
                else:
                    label, _, value = line.partition(": ")
                    if (field := fields.get(label)) is None:
                        continue
                    if value == "[blank]":
                        value = None
                    elif field in DOCUMENT_DATE_FIELDS:
                        day, month, year = map(int, value.split("/"))
                        value = date(year, month, day).isoformat()
                    elif field in DOCUMENT_AMOUNT_FIELDS:
                        value = value.removeprefix("LKR ").replace(",", "")
                    elif field == "inspection_basis":
                        value = value.lower().replace(" ", "_")
                    elif field == "commercial_trip_verified":
                        value = value == "Yes"
                    document[field] = value
            documents.append(document)
    documents.sort(key=lambda document: int(document["id"].rsplit("-", 1)[-1]))
    transcript = inputs["fnol_transcript"]
    return {"narrative": transcript.split(FNOL_MARKER, 1)[1].removesuffix(CALL_END), "documents": documents}


def resolve_claim_text(claim):
    """Use the filed text to derive the investigation view of a synthetic claim."""
    if claim.get("source") == "Synthetic generator" and (inputs := claim.get("unstructured_inputs")):
        return {**claim, **extract_claim_text(inputs)}
    return claim


def available_text_refs(claim):
    """References to actual text, excluding placeholders for absent filings."""
    return [TEXT_REF_PREFIX + key for key, value in (claim.get("unstructured_inputs") or {}).items()
            if value.strip() not in {"", NO_DOCUMENT, NO_POLICE_REPORT}]


def document_text_refs(claim):
    """Link extracted document positions to the streams they were read from."""
    available = set(available_text_refs(claim))
    if not available:
        return {}
    refs = {}
    for index, document in enumerate(claim.get("documents") or []):
        key = SOURCE_BY_DOCUMENT_TYPE.get(document["type"], "repair_estimate_text")
        if (source := TEXT_REF_PREFIX + key) in available:
            refs[f"submission:/documents/{index}"] = source
    return refs
