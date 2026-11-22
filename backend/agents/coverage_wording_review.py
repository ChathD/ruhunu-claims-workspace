"""Validate policy citations and reconcile the wording with the structured policy schedule."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from backend.agents.coverage_specialist import describes_policy_silence

class PolicyConflict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    item: str = Field(min_length=1, max_length=160)
    structured_term: str = Field(min_length=1, max_length=350)
    wording_term: str = Field(min_length=1, max_length=350)
    passage_refs: list[str] = Field(min_length=1, max_length=5)
    materiality: Literal["material", "immaterial", "uncertain"]
    impact: str = Field(min_length=1, max_length=500)


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
