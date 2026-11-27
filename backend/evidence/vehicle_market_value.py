"""Retrieve dated Sri Lankan asking price benchmarks for an exact model year."""
import json
import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

import httpx
from pydantic import BaseModel, Field, field_validator


PRICE_TREND_URL = "https://pricemart.lk/dashboard/trend"
PRICE_SOURCE = "PriceMart.lk"
PAGE_DATA = re.compile(r"self\.__next_f\.push\((\[.*?\])\)</script>")
VEHICLE_SLUG = re.compile(r"[^a-z0-9]+")
PROPORTIONAL_REDUCTION = re.compile(r"\b(?:condition of average|rateable proportion|proportionate reduction|underinsur\w*)\b", re.I)
MARKET_CAP = re.compile(r"\b(?:lower of|lesser of|whichever is less|market[- ]value (?:cap|ceiling|limit)|(?:policy|settlement|payment) cap|capped)\b", re.I)
VALUATION_REQUIREMENT = re.compile(r"\b(?:mandatory (?:vehicle )?valuation|(?:valuation|appraisal) (?:is mandatory|must be (?:submitted|provided))"
                                   r"|(?:requires?|must (?:submit|supply|provide)) (?:an? )?(?:independent |vehicle |written )?(?:valuation|appraisal))\b", re.I)
SEPARATE_REQUIREMENT = re.compile(r"\b(?:approval|consent|total loss|towing|(?:scheduled value|sum insured) (?:is )?(?:missing|unconfirmed|unavailable)"
                                  r"|must (?:submit|supply|provide))\b", re.I)
MAX_CAP_BENCHMARK_AGE_DAYS = 180
MAX_CLAIM_SHARE_FOR_CAP_ASSUMPTION = Decimal("0.50")


class VehicleMarketQuery(BaseModel):
    make: str = Field(min_length=1, max_length=60)
    model: str = Field(min_length=1, max_length=80)
    year: int = Field(ge=1886, le=2100)
    as_of: date

    @field_validator("make", "model")
    @classmethod
    def validate_vehicle_name(cls, value):
        if not re.search(r"[a-z0-9]", value, re.I):
            raise ValueError("Enter a vehicle make and model")
        return value.strip()

    @field_validator("as_of", mode="before")
    @classmethod
    def read_as_of_date(cls, value):
        if isinstance(value, str) and "/" in value:
            return datetime.strptime(value, "%d/%m/%Y").date()
        return value


def published_trend(html, make, model):
    parts = []
    for match in PAGE_DATA.finditer(html):
        try:
            packet = json.loads(match[1])
        except ValueError:
            continue
        if len(packet) == 2 and packet[0] == 1 and isinstance(packet[1], str):
            parts.append(packet[1])
    content = "".join(parts)
    decoder = json.JSONDecoder()
    fields = {}
    for key in ("initialMake", "initialModel", "initialTrend"):
        match = re.search(rf'"{key}"\s*:\s*', content)
        if not match:
            raise ValueError("Missing published price series")
        fields[key], _ = decoder.raw_decode(content, match.end())
    if fields["initialMake"] != make or fields["initialModel"] != model or not isinstance(fields["initialTrend"], list):
        raise ValueError("Unexpected vehicle price series")
    return fields["initialTrend"]


def lookup_vehicle_market_value(query: VehicleMarketQuery):
    make, model = (VEHICLE_SLUG.sub("-", name.lower()).strip("-") for name in (query.make, query.model))
    url = f"{PRICE_TREND_URL}/{make}/{model}"
    result = {**query.model_dump(mode="json"), "source": PRICE_SOURCE, "source_url": url,
              "basis": "Average advertised asking price", "currency": "LKR"}
    try:
        response = httpx.get(url, timeout=8)
        if response.status_code == 404:
            return {**result, "status": "no_observation"}
        response.raise_for_status()
        rows = published_trend(response.text, make, model)
    except (httpx.HTTPError, ValueError):
        return {**result, "status": "unavailable"}

    selected = None
    for row in rows:
        if not isinstance(row, dict) or (value := row.get(str(query.year))) is None:
            continue
        try:
            observed_at = date.fromisoformat(row["date"])
            amount = Decimal(str(value))
        except (KeyError, TypeError, ValueError, InvalidOperation):
            continue
        if (observed_at <= query.as_of and amount.is_finite() and amount > 0
                and (selected is None or observed_at > selected[0])):
            selected = (observed_at, amount)

    if selected is None:
        return {**result, "status": "no_observation"}
    observed_at, amount = selected
    return {**result, "status": "found", "observation_date": observed_at.isoformat(),
            "days_before_requested": (query.as_of - observed_at).days, "asking_price_lkr": format(amount, "f")}


def repair_market_cap_applies(claim, schedule):
    if (not schedule or claim.get("loss_type") not in {"repair", "glass"} or claim.get("currency") != "LKR"
            or not claim.get("loss_date") or not claim.get("registration")
            or schedule.get("wording", {}).get("document_kind") == "product_summary"
            or not schedule.get("starts_on", "9999-12-31") <= claim["loss_date"] <= schedule.get("ends_on", "")
            or schedule.get("vehicle", {}).get("registration") != claim.get("registration")):
        return False
    benefit = next((item for item in schedule.get("cover", []) if item["loss_type"] == claim["loss_type"]), {})
    terms = benefit.get("conditions") or ""
    return (not PROPORTIONAL_REDUCTION.search(terms) and not VALUATION_REQUIREMENT.search(terms)
            and (benefit.get("limit_basis") == "market_value"
                 or (benefit.get("limit_basis") == "scheduled_amount" and "market value" in terms.casefold() and bool(MARKET_CAP.search(terms)))))


def assess_repair_market_cap(claim, lookup):
    benchmark = lookup["result"]
    assessment = {"status": "unconfirmed", "benchmark": benchmark, "evidence_ref": lookup["evidence_ref"]}
    if benchmark.get("status") != "found" or benchmark.get("currency") != "LKR":
        return assessment
    try:
        gross = Decimal(str(claim["gross_amount"]))
        asking = Decimal(str(benchmark["asking_price_lkr"]))
        observed = date.fromisoformat(benchmark["observation_date"])
        age = (date.fromisoformat(claim["loss_date"]) - observed).days
    except (InvalidOperation, KeyError, TypeError, ValueError):
        return assessment
    if (not gross.is_finite() or not asking.is_finite() or not 0 < gross <= asking * MAX_CLAIM_SHARE_FOR_CAP_ASSUMPTION
            or not 0 <= age <= MAX_CAP_BENCHMARK_AGE_DAYS):
        return assessment
    vehicle = f"{benchmark['year']} {benchmark['make']} {benchmark['model']}"
    return {**assessment, "status": "unlikely_to_limit", "gross_claim_lkr": str(gross), "observation_age_days": age,
            "summary": f"{benchmark['source']} recorded an average asking price of LKR {asking:,.2f} for a {vehicle} on {observed:%d/%m/%Y}. "
                       f"The market-value cap is unlikely to reduce payment because the benchmark is well above the full LKR {gross:,.2f} claim. "
                       "This is a provisional settlement assumption."}


def market_cap_only_review(text):
    return ("market value" in text.casefold().replace("-", " ") and MARKET_CAP.search(text)
            and not PROPORTIONAL_REDUCTION.search(text) and not VALUATION_REQUIREMENT.search(text) and not SEPARATE_REQUIREMENT.search(text))


def recorded_market_cap_assumption(coverage):
    assessment = coverage.get("settlement_market_cap", {})
    if (coverage.get("coverage_status") == "COVERED" and coverage.get("schedule_confirmed")
            and assessment.get("status") == "unlikely_to_limit"):
        return assessment["summary"]
    return None
