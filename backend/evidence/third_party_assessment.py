"""Associate third-party findings and evidence requests with frozen demand lines."""

import re
from decimal import Decimal, InvalidOperation
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CoverageEvidenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    owner: Literal["claimant", "internal"]
    document: str = Field(min_length=1, max_length=200)
    reason: str = Field(min_length=1, max_length=350)
    location: str | None = Field(default=None, max_length=200)


class ThirdPartyCostAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    line_ref: str
    policy_item: str | None = None
    amount_evidence_refs: list[str] = Field(default_factory=list, max_length=4)
    evidence_requests: list[CoverageEvidenceRequest] = Field(default_factory=list, max_length=3)


def claimed_costs(claim):
    demand = [{"line_ref": f"submission:/documents/{doc_index}/line_items/{line_index}",
               "description": line["description"], "claimed_amount": line["amount"]}
              for doc_index, doc in enumerate(claim.get("documents", [])) if doc.get("type") == "third_party_demand"
              for line_index, line in enumerate(doc.get("line_items") or [])]
    return demand + [{"line_ref": f"submission:/additional_costs/{index}", "description": line["reason"],
                      "claimed_amount": line["amount"]} for index, line in enumerate(claim.get("additional_costs", []))]


FACTUAL_LIABILITY_EVIDENCE = re.compile(r"\b(?:police (?:report|findings|records?)|witness|dash[ -]?cam|CCTV|footage|photographs?|"
                                        r"photos?|incident report|(?:scene|accident) (?:sketch|diagram|report))\b", re.I)
LIABILITY_DETERMINATION = re.compile(r"\b(?:admission|judg(?:e)?ment|court (?:order|decision|ruling)|motor traffic act|"
                                    r"(?:formal |legal )?liability (?:assessment|determination|confirmation)|insurer consent)\b", re.I)
EXPENSE_AUTHORIZATION = re.compile(r"\b(?:consent|authorization|authorisation|approval)\b", re.I)
TOWING_RECORD = re.compile(r"\btowing (?:invoice|receipt)\b", re.I)


def liability_evidence_requests(requests):
    accepted = []
    needs_internal_assessment = False
    for request in requests:
        if request["owner"] == "claimant":
            document = request["document"]
            if not FACTUAL_LIABILITY_EVIDENCE.search(document) or LIABILITY_DETERMINATION.search(document + " " + request["reason"]):
                needs_internal_assessment = True
                continue
        accepted.append(request)
    if needs_internal_assessment and not any(request["owner"] == "internal" for request in accepted):
        accepted.append({"owner": "internal", "document": "Insurer assessment of liability", "location": None,
                         "reason": "The claims team must assess legal responsibility using the available collision evidence."})
    return accepted


def assess_third_party(analysis, costs, policy_items, document_refs, liability_refs, *, policy_applies, claim):
    proposals = {candidate.line_ref: candidate for candidate in analysis.cost_assessments}
    terms = {item["item"]: item for item in policy_items}
    confirmed = bool(liability_refs)
    liability = {"status": "confirmed" if confirmed else "unconfirmed", "evidence_refs": sorted(liability_refs),
                 "evidence_requests": []}
    if not confirmed and policy_applies:
        liability["evidence_requests"] = liability_evidence_requests([request.model_dump() for request in analysis.liability_evidence_requests])
    assessed = []
    for cost in costs:
        ref = cost["line_ref"]
        proposal = proposals[ref]
        term = terms.get(proposal.policy_item) if policy_applies else None
        applicability = {"covered": "applies", "not_covered": "excluded"}.get(term["outcome"] if term else None, "unresolved")
        requests = [request.model_dump() for request in proposal.evidence_requests
                    if request.owner != "claimant" or not EXPENSE_AUTHORIZATION.search(request.document)] if term and applicability != "excluded" else []
        for request in requests:
            if request["owner"] == "claimant" and TOWING_RECORD.search(request["document"]):
                request["reason"] = "An invoice or receipt would verify the towing charge and link the service to this incident."
                try:
                    amount = Decimal(str(cost["claimed_amount"]))
                except InvalidOperation:
                    continue
                if amount.is_finite() and amount > 0:
                    request["reason"] = request["reason"].replace("the towing charge", f"the {claim.get('currency', 'LKR')} {amount:,.2f} towing charge")
        evidence = document_refs.intersection(proposal.amount_evidence_refs)
        supported = bool(term and applicability != "excluded" and evidence and not requests)
        assessed.append({**cost, "policy_applicability": applicability, "policy_term": term["policy_term"] if term else None,
                         "passage_refs": term.get("passage_refs", []) if term else [],
                         "amount_status": "supported" if supported else "unverified",
                         "amount_evidence_refs": sorted(evidence) if supported else [], "evidence_requests": requests})
    result = {"liability": liability, "costs": assessed}
    if policy_applies:
        if request := third_party_total_gap(claim):
            result["total_evidence_request"] = request
    return result


def third_party_total_gap(claim):
    demands = [doc for doc in claim.get("documents", []) if doc.get("type") == "third_party_demand"]
    if not demands or any(not doc.get("line_items") for doc in demands):
        return None
    costs = claimed_costs(claim)
    if claim.get("gross_amount") is None or any(cost.get("claimed_amount") is None for cost in costs):
        return None
    try:
        gross = Decimal(str(claim["gross_amount"]))
        accounted = sum((Decimal(str(cost["claimed_amount"])) for cost in costs), Decimal(0))
    except InvalidOperation:
        return None
    if not gross.is_finite() or not accounted.is_finite() or gross == accounted:
        return None
    currency = claim.get("currency", "LKR")
    return {"owner": "claimant", "document": "Breakdown of the total claim amount", "location": None,
            "reason": f"The itemised demand and separate charges total {currency} {accounted:,.2f}. "
                      f"The claim form states {currency} {gross:,.2f}, leaving a difference of {currency} {abs(gross - accounted):,.2f}."}


def supported_payable(assessment):
    return (assessment["liability"]["status"] == "confirmed"
            and not assessment.get("total_evidence_request")
            and bool(assessment["costs"])
            and all(cost["policy_applicability"] == "applies" and cost["amount_status"] == "supported"
                    for cost in assessment["costs"]))
