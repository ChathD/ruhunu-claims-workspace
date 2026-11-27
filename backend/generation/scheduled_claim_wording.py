"""Resolve saved schedule citations back to the applicable linked policy text."""
from dataclasses import dataclass
from decimal import Decimal

from sqlalchemy import select

from backend.generation.policy_schedule_generator import ScheduleCitation, checked_citation, lkr_amounts, wording_lines, wording_sources
from backend.storage.sqlite_store import PolicyChunk


def normalized(text):
    return " ".join(text.split())


@dataclass(frozen=True)
class LinkedWording:
    metadata: dict
    pages: dict
    chunks: dict

    def matches(self, schedule):
        wording = schedule["wording"]
        return (all(wording.get(key) == self.metadata[key] for key in ("id", "version", "file_hash"))
                and wording.get("document_kind") in {None, self.metadata["document_kind"]})

    def citation(self, saved):
        reference, quote = saved.get("reference", ""), saved.get("quote")
        if not reference or not quote or saved.get("page") is None:
            return None
        if reference.startswith("chunk:"):
            chunk = self.chunks.get(reference)
            return saved if chunk == (saved["page"], quote) else None
        if not (source := self.pages.get(f"W{saved['page']:03d}")) or source["reference"] != reference:
            return None
        checked = checked_citation(ScheduleCitation(reference=f"W{saved['page']:03d}",
            start_line=saved.get("start_line"), end_line=saved.get("end_line")), self.pages)
        return checked if checked and normalized(checked["quote"]) == normalized(quote) else None

    def benefit_clause(self, benefit):
        citation = self.citation(benefit.get("wording") or {})
        if not citation:
            return None
        if benefit["limit_basis"] == "wording_amount":
            limit = Decimal(benefit["limit_lkr"])
            if limit not in lkr_amounts(citation["quote"], ocr=citation.get("ocr", False))[0]:
                return None
        return citation

    def find_clause(self, predicate):
        for source in self.pages.values():
            lines = wording_lines(source["text"])
            for index in range(len(lines)):
                end = min(index + 4, len(lines))
                quote = "\n".join(lines[index:end])
                if predicate(normalized(quote)):
                    return {"reference": source["reference"], "page": source["page"],
                            "start_line": index + 1, "end_line": end, "quote": quote}
        return None


def linked_wording(session, document_id):
    try:
        metadata, pages = wording_sources(session, document_id, already_selected=True)
    except ValueError:
        return None
    chunks = {f"chunk:{row.id}": (row.page, row.text) for row in session.scalars(
        select(PolicyChunk).where(PolicyChunk.document_id == document_id))}
    return LinkedWording(metadata, pages, chunks)
