"""Identify motor benefits and mandatory reports from the saved policy terms."""
import re
from decimal import Decimal
from typing import Literal


MotorBenefit = Literal["motor_medical", "scheduled_equipment", "roadside_towing", "glass_waiver"]
BENEFIT_NAMES = {
    "motor_medical": r"\bmedical expenses\b",
    "scheduled_equipment": r"\b(?:parts\s*(?:(?:and|&)(?:/or)?|,)\s*accessories|scheduled equipment|vehicle accessories)\b",
    "roadside_towing": r"\b(?:towing|roadside assistance|protection and removal)\b",
    "glass_waiver": r"\b(?:glass breakage|glass waiver|windscreen|windshield)\b",
}
GLASS_WAIVER = re.compile(r"\b(?:irrespective of any excess|no excess applies|zero[ -]deductible|without (?:an? )?excess|excess (?:is )?waived)\b", re.I)
SECTION_COST_EXCESS = re.compile(r"\bexcess\b[^.]{0,300}\bsection\s+[i1]\b[^.]{0,120}\binclud\w*\s+(?:costs|expenses)", re.I)
ADDITIONAL_EXCESS = re.compile(r"\bin addition to (?:any )?other (?:applicable )?excess(?:es)?\b", re.I)
COMMERCIAL_EXCLUSION = re.compile(
    r"\b(?:private use only|exclusively for private use|(?:exclud\w*|except|not permitted|not covered|does not cover|not for|no use for)"
    r"[^.;\n]{0,70}\b(?:hire (?:or|and) reward|ride[ -]?share|commercial (?:use|passenger)))\b", re.I,
)
POLICE_DUTY = (r"(?:police report (?:is )?(?:required|mandatory)|must (?:submit|provide|attach) a police report|"
               r"police report must be (?:submitted|provided|attached))")
THRESHOLD_AMOUNT = r"(?:above|over|exceed(?:s|ing)?)\s+(?:LKR|Rs\.?|රු\.?)\s*([\d,]+(?:\.\d{1,2})?)"
POLICE_THRESHOLDS = (
    re.compile(rf"\b{POLICE_DUTY}[^.;\n]{{0,80}}\b{THRESHOLD_AMOUNT}", re.I),
    re.compile(rf"\b{THRESHOLD_AMOUNT}[^.;\n]{{0,80}}\b{POLICE_DUTY}", re.I),
)


def quoted_terms(benefit):
    return " ".join((benefit.get("conditions") or "", benefit.get("wording", {}).get("quote") or ""))


def fixed_motor_benefit(schedule, kind):
    pattern = BENEFIT_NAMES.get(kind)
    if not pattern or schedule["wording"].get("document_kind") == "product_summary":
        return None
    matches = []
    for benefit in schedule["cover"]:
        if (benefit["loss_type"] != "additional_benefit" or not re.search(pattern, benefit.get("benefit_name") or "", re.I)
                or benefit["limit_basis"] not in {"scheduled_amount", "wording_amount"} or not benefit.get("limit_lkr")
                or any(benefit.get(key) is None for key in ("compulsory_excess_lkr", "voluntary_excess_lkr"))):
            continue
        terms = quoted_terms(benefit)
        if kind == "glass_waiver" and (not GLASS_WAIVER.search(terms) or any(
                Decimal(benefit[key]) for key in ("compulsory_excess_lkr", "voluntary_excess_lkr"))):
            continue
        if kind == "motor_medical" and not re.search(r"\b(?:insured|you)\b", terms, re.I):
            continue
        matches.append(benefit)
    return matches[0] if len(matches) == 1 else None


def towing_excess_unresolved(schedule, benefit):
    wording = (benefit.get("wording") or {}).get("quote") or ""
    return (any(SECTION_COST_EXCESS.search(" ".join(endorsement.split())) for endorsement in schedule.get("endorsements", []))
            and not GLASS_WAIVER.search(wording))


def motor_benefit_excess(schedule, benefit, kind):
    fields = ("compulsory_excess_lkr", "voluntary_excess_lkr")
    if any(benefit.get(field) is None for field in fields):
        return None
    excess = sum(Decimal(benefit[field]) for field in fields)
    # Some parts-theft endorsements expressly add their own excess to the
    # existing theft excess. A blank base excess cannot support auto triage.
    if kind == "scheduled_equipment" and ADDITIONAL_EXCESS.search(" ".join(quoted_terms(benefit).split())):
        theft = next((row for row in schedule["cover"] if row["loss_type"] == "theft"), None)
        if not theft or any(theft.get(field) is None for field in fields):
            return None
        excess += sum(Decimal(theft[field]) for field in fields)
    return excess


def commercial_use_exclusion(schedule):
    return next((term for term in (schedule.get("permitted_use", ""), *schedule.get("special_conditions", []))
                 if COMMERCIAL_EXCLUSION.search(term)), None)


def mandatory_police_threshold(schedule, benefit):
    text = "\n".join((schedule.get("claims_instructions", ""), quoted_terms(benefit), *schedule.get("special_conditions", [])))
    thresholds = police_thresholds(text)
    return thresholds.pop() if len(thresholds) == 1 else None


def police_thresholds(text):
    return {Decimal(value.replace(",", "")) for pattern in POLICE_THRESHOLDS for value in pattern.findall(text)}
