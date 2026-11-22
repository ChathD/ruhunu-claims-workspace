"""Compare similar approved claims to assess costs and identify leads for fraud follow-up."""
from decimal import Decimal
from statistics import median

from backend.agents.intake_specialist import cost_breakdown, cost_evidence, missing_cost_fields
from backend.claim_schemas import Finding
from backend.evidence.approved_claim_history import uses_document_reference

def investigate(tools):
    claim = tools["read_submission"].invoke({})
    records = tools["retrieve_comparable_claims"].invoke({})["records"]
    available = claim.get("gross_amount") not in (None, "") and bool(records)
    return Finding(agent="historical", status="pass" if available else "review",
                   summary="Comparable records gathered for analysis." if available else "The claimed amount or comparable records are unavailable.",
                   evidence=["tool:retrieve_comparable_claims", "submission:/gross_amount"], details={})


def cost_document_reference(claim, document):
    reference = document.get("document_reference")
    # Manual file IDs are not printed estimate references.
    if reference is None and claim.get("source") != "Manual submission":
        reference = document["id"]
    return reference


def reference_leads(claim, records):
    """Exact-match candidates for offline fixture runs and scripted providers."""
    leads = []
    for document in cost_evidence(claim)[0]:
        reference = cost_document_reference(claim, document)
        linked = [row for row in records if reference and uses_document_reference(row, reference)]
        if len({row["incident_id"] for row in linked}) >= 2:
            leads.append({"document_id": document["id"], "history_ids": [row["id"] for row in linked]})
    return leads


def cost_document_amount(claim):
    documents, _, _ = cost_evidence(claim)
    if not documents or any(
        document.get("total") in (None, "") or not document.get("line_items")
        or any(item.get("amount") in (None, "") or not item.get("description") for item in document["line_items"])
        or sum(Decimal(item["amount"]) for item in document["line_items"]) != Decimal(document["total"])
        for document in documents
    ):
        return None
    total, _ = cost_breakdown(claim, documents)
    return total if total > 0 else None


def exact_scope_ids(claim, records):
    """Conservative fixture/offline fallback when no model can interpret damage."""
    def scope(item):
        documents, _, _ = cost_evidence(item)
        return sorted(line["description"].strip().lower() for doc in documents for line in doc.get("line_items", []))
    current_scope = scope(claim)
    return [row["id"] for row in records if current_scope and row.get("narrative") == claim.get("narrative")
            and scope(row) == current_scope]


def apply_comparison(finding, claim, records, conclusion, proposed_leads, comparable_ids=None):
    """Reproduce the benchmark from the cost evidence the specialist selected."""
    records = [row for row in records if Decimal(row["gross_cost"]) > 0]
    by_id = {row["id"]: row for row in records}
    selected = exact_scope_ids(claim, records) if comparable_ids is None else list(dict.fromkeys(comparable_ids))
    if any(identifier not in by_id for identifier in selected):
        raise ValueError("Cost comparison cites a claim outside the retrieved history")
    comparable_amounts = {identifier: cost_document_amount(by_id[identifier]) for identifier in selected if by_id[identifier].get("narrative")}
    comparable_amounts = {identifier: amount for identifier, amount in comparable_amounts.items() if amount is not None}
    costs = sorted(comparable_amounts.values())
    middle = median(costs) if costs else None
    current_cost = cost_document_amount(claim)
    _, additional_total = cost_breakdown(claim, [])
    if len(costs) < 5 or current_cost is None:
        conclusion = "INSUFFICIENT_EVIDENCE"
    elif conclusion is None:
        conclusion = "REQUIRES_REVIEW" if additional_total or current_cost > middle * 2 else "SUPPORTED"
    leads = []
    documents = {document["id"]: document for document in cost_evidence(claim)[0]}
    for proposed in proposed_leads:
        document = documents.get(proposed["document_id"])
        history_ids = list(dict.fromkeys(proposed["history_ids"]))
        reference = cost_document_reference(claim, document) if document else None
        if not reference or any(identifier not in by_id for identifier in history_ids):
            raise ValueError("Historical lead cites a document or comparison outside the specialist evidence")
        matches = {by_id[identifier]["incident_id"] for identifier in history_ids if uses_document_reference(by_id[identifier], reference)}
        if len(matches) < 2 or any(not uses_document_reference(by_id[identifier], reference) for identifier in history_ids):
            raise ValueError("Historical lead does not identify matching estimate references across distinct incidents")
        leads.append({"pattern": "Current cost document reference occurs in multiple settled incidents",
                      "document_reference": reference, "document_id": document["id"],
                      "history_ids": history_ids,
                      "alternative": "Reference collision or issuer clerical error. Independently verify before referral."})
    amount = {"conclusion": conclusion, "claimed_gross": claim.get("gross_amount"),
              "cost_documents_total": str(current_cost) if current_cost is not None else None,
              "additional_costs_total": str(additional_total), "cost_basis": "itemised cost document totals in LKR",
              "sample_size": len(costs), "comparable_claim_ids": list(comparable_amounts),
              "median": str(middle) if middle is not None else None,
              "observed_range": [str(costs[0]), str(costs[-1])] if costs else None,
              "normalization": "Comparable damage and work, same loss type and currency. No inflation adjustment.",
              "limitation": "Historic cost document amounts are submitted amounts. Additional costs are assessed separately."}
    if middle is None:
        finding.summary += " No usable cost documents for a like-for-like comparison."
    else:
        finding.summary += f" {len(costs)} comparable cost documents with a median total of LKR {middle:.2f}."
    if 0 < len(costs) < 5:
        finding.summary += " At least five usable comparables are required to support the benchmark."
    if leads:
        finding.summary += " The repeated cost document reference needs independent fraud verification."
    if conclusion != "SUPPORTED":
        finding.status = "information" if missing_cost_fields(claim) else "review"
    finding.details.update(amount_justification=amount, fraud_investigation={"leads": leads})
