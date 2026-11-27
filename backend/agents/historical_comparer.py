"""Compare similar prior claims to assess costs and identify leads for fraud follow-up."""
from decimal import Decimal
from statistics import median

from backend.agents.intake_specialist import cost_breakdown, cost_evidence, missing_cost_fields
from backend.claim_schemas import Finding
from backend.evidence.claim_history import uses_document_reference
from backend.evidence.required_claim_evidence import cost_line_total
from backend.evidence.synthetic_claim_text import TEXT_REF_PREFIX, available_text_refs

def investigate(tools):
    claim = tools["read_submission"].invoke({})
    records = tools["retrieve_comparable_claims"].invoke({})["records"]
    available = claim.get("gross_amount") not in (None, "") and bool(records)
    evidence = ["tool:retrieve_comparable_claims", "submission:/gross_amount"]
    if cost_evidence(claim)[0] and (cost_text := TEXT_REF_PREFIX + "repair_estimate_text") in available_text_refs(claim):
        evidence.append(cost_text)
    return Finding(agent="historical", status="pass" if available else "review",
                   summary="Comparable records gathered for analysis." if available else "The claimed amount or comparable records are unavailable.",
                   evidence=evidence, details={})


def cost_document_reference(claim, document):
    reference = document.get("document_reference")
    # Manual file IDs are not printed estimate references.
    if reference is None and claim.get("source") != "Manual submission":
        reference = document["id"]
    return reference


def reference_leads(claim, records):
    """Find exact repeats of a current cost document reference across distinct prior incidents."""
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
        or cost_line_total(document["line_items"]) != Decimal(document["total"])
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


def apply_comparison(finding, claim, records, conclusion, proposed_leads=None, comparable_ids=None, *, review_assessment=None):
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
    if not costs or current_cost is None:
        conclusion = "INSUFFICIENT_EVIDENCE"
    elif conclusion is None:
        conclusion = "REQUIRES_REVIEW" if additional_total or current_cost > middle * 2 else "SUPPORTED"
    leads = []
    documents = {document["id"]: document for document in cost_evidence(claim)[0]}
    if proposed_leads is None:
        proposed_leads = reference_leads(claim, records)
    for proposed in proposed_leads:
        document = documents.get(proposed["document_id"])
        history_ids = list(dict.fromkeys(proposed["history_ids"]))
        reference = cost_document_reference(claim, document) if document else None
        if not reference or any(identifier not in by_id for identifier in history_ids):
            raise ValueError("Historical lead cites a document or comparison outside the specialist evidence")
        matches = {by_id[identifier]["incident_id"] for identifier in history_ids if uses_document_reference(by_id[identifier], reference)}
        if len(matches) < 2 or any(not uses_document_reference(by_id[identifier], reference) for identifier in history_ids):
            raise ValueError("Historical lead does not identify matching estimate references across distinct incidents")
        leads.append({"pattern": "Current cost document reference occurs in multiple prior incidents",
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
    if current_cost is not None and (cost_text := TEXT_REF_PREFIX + "repair_estimate_text") in available_text_refs(claim):
        amount["current_cost_text_ref"] = cost_text
    comparable_sources = {identifier: by_id[identifier]["cost_document_text_refs"] for identifier in comparable_amounts
                          if by_id[identifier].get("cost_document_text_refs")}
    if comparable_sources:
        amount["comparable_cost_text_refs"] = comparable_sources
    review_history = [{"claim_id": row["id"], "claim_reference": row.get("claim_reference", row["id"]),
                       "loss_date": row.get("loss_date"), "narrative": row.get("narrative"),
                       "triage_decision": row.get("triage_decision"), "staff_decision": row.get("staff_decision")}
                      for identifier in selected if (row := by_id[identifier]).get("triage_decision") or row.get("staff_decision")]
    if middle is None:
        finding.summary += " No usable cost documents for a like-for-like comparison."
    else:
        document_count = "1 comparable cost document" if len(costs) == 1 else f"{len(costs)} comparable cost documents"
        finding.summary += f" {document_count} with a median total of LKR {middle:,.2f}."
    if leads:
        finding.summary += " The repeated cost document reference needs independent fraud verification."
    if conclusion != "SUPPORTED":
        finding.status = "information" if missing_cost_fields(claim) else "review"
    finding.details.update(amount_justification=amount, fraud_investigation={"leads": leads})
    if review_history:
        finding.details["review_history"] = review_history
        if review_assessment:
            finding.details["review_assessment"] = review_assessment
