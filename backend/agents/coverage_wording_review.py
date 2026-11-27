"""Validate policy citations and reconcile the wording with the structured policy schedule."""
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from backend.agents.coverage_specialist import describes_policy_silence
from backend.evidence.required_claim_evidence import SUPPORTING_DOCUMENT_LABELS, required_gap

class PolicyConflict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    item: str = Field(min_length=1)
    structured_term: str = Field(min_length=1)
    wording_term: str = Field(min_length=1)
    passage_refs: list[str] = Field(min_length=1, max_length=5)
    materiality: Literal["material", "immaterial", "uncertain"]
    impact: str = Field(min_length=1)


def assess_bound_wording(analysis, retrievals, document_id):
    """Accept only passages retrieved for this assessment's frozen wording version."""
    passages = {passage["source"]: passage for result in retrievals if result.get("document_id") == document_id
                for passage in result.get("passages", [])}
    references = set(analysis.applicable_passages)
    applicable = (analysis.policy_applicability == "applicable" and bool(references)
                  and references.issubset(passages))
    waiting_on_facts = (
        not retrievals and not references and not analysis.policy_conflicts and not analysis.policy_omissions
        and analysis.coverage_conclusion == "UNRESOLVED" and bool(analysis.coverage_items)
        and all(item.basis == "schedule" and item.outcome == "undetermined" and not item.passage_refs
                and not describes_policy_silence(item.policy_term) for item in analysis.coverage_items)
    )
    reasons, items, conflicts = [], [], []
    if not applicable:
        if not waiting_on_facts:
            reasons.append("Coverage could not verify a citation to applicable passages in the linked wording.")
        references = set()

    for item in analysis.coverage_items:
        if not describes_policy_silence(item.policy_term) and (
                not references.issuperset(item.passage_refs) or (item.basis != "schedule" and not item.passage_refs)):
            reasons.append(f"{item.item} requires an applicable wording citation.")
            continue
        items.append(item)

    for conflict in analysis.policy_conflicts:
        if not references.issuperset(conflict.passage_refs):
            reasons.append(f"The reported policy difference for {conflict.item} could not be verified from the cited wording.")
            continue
        conflicts.append(conflict.model_dump())
        if conflict.materiality != "immaterial":
            reasons.append(f"Policy wording and structured terms need reconciliation for {conflict.item}. {conflict.impact}")

    wording = {"document_id": document_id, "applicability": analysis.policy_applicability, "cited": applicable,
               "applicable_passages": sorted(references), "retrievals": retrievals}
    if waiting_on_facts:
        wording["search_skipped"] = "An unresolved claim fact prevented a decision under the structured policy terms."
    return items, conflicts, wording, reasons


def assess_selected_wording(analysis, items, retrievals, selected, claim, conclusion, deferred_cost_gaps=()):
    passages = {p["source"]: p for result in retrievals if result.get("document_id") == selected["id"]
                and result.get("file_hash") == selected["file_hash"] for p in result.get("passages", [])}
    applicable = analysis.policy_applicability == "applicable"
    cited_items = [item for item in items if analysis.policy_applicability != "irrelevant"
                   and item.passage_refs and all(ref in passages for ref in item.passage_refs)]
    summary_only = selected.get("document_kind") == "product_summary"
    if summary_only or not applicable:
        conclusion = "UNRESOLVED"
    supported = [item.model_copy(update={"outcome": "undetermined"}) if summary_only or not applicable else item for item in cited_items]
    required_outcome = {"COVERED": "covered", "NOT_COVERED": "not_covered"}.get(conclusion)
    if required_outcome and not any(item.outcome == required_outcome for item in supported):
        conclusion = "UNRESOLVED"

    gaps = list(deferred_cost_gaps) if conclusion == "COVERED" else []
    wording_refs = {ref for item in supported for ref in item.passage_refs}
    applicable_refs = {ref for item in supported if item.outcome != "undetermined" for ref in item.passage_refs}
    attached = {doc["type"] for doc in claim.get("documents", [])}
    if applicable and not summary_only and analysis.coverage_conclusion != "NOT_COVERED" and analysis.classified_loss in (None, claim["loss_type"]):
        for requirement in analysis.required_documents:
            if requirement.document_type in attached or requirement.passage_ref not in passages:
                continue
            label = {"police_report": "Police report", **SUPPORTING_DOCUMENT_LABELS}[requirement.document_type]
            passage = " ".join(passages[requirement.passage_ref]["text"].casefold().split())
            quote = " ".join(requirement.requirement_quote.casefold().split())
            mandatory = re.search(r"\b(?:must|shall|required|mandatory|requires?)\b", quote)
            optional = re.search(r"\b(?:may (?:be required|require|be requested)|if (?:requested|available)|on request|where necessary|"
                                 r"optional|not (?:required|mandatory)|need not)\b", quote)
            document_action = re.search(r"\b(?:submit\w*|provid\w*|furnish\w*|suppl\w*|send\w*|produc\w*|attach\w*|obtain\w*|"
                                        r"(?:is|are) (?:required|mandatory))\b", quote)
            if quote not in passage or label.casefold() not in quote or not mandatory or optional or not document_action:
                continue
            wording_refs.add(requirement.passage_ref)
            gap = required_gap(requirement.document_type, label, None, f"The selected policy requires this document for the reported loss. {requirement.reason}")
            gap["policy_citation"] = {"id": requirement.passage_ref, "quote": requirement.requirement_quote}
            gaps.append(gap)
            applicable_refs.add(requirement.passage_ref)
    if gaps:
        conclusion = "NEEDS_INFORMATION"
    wording = {"document_id": selected["id"], "file_hash": selected["file_hash"], "applicability": analysis.policy_applicability,
               "document_kind": selected.get("document_kind", "wording"), "cited": bool(applicable_refs),
               "wording_passages": sorted(wording_refs), "applicable_passages": sorted(applicable_refs), "retrievals": retrievals}
    ocr_pages = sorted({passages[ref]["page"] for ref in wording_refs if passages[ref].get("extraction") == "ocr"})
    if ocr_pages:
        wording["ocr_pages"] = ocr_pages
    return supported, conclusion, wording, gaps
