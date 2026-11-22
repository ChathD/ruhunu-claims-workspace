"""Find policy language copied into generated claims and supporting documents."""
import re
import unicodedata


PASSAGE_WORDS = 6


def words(text):
    return tuple(re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold()))


def phrases(tokens, length):
    return zip(*(tokens[offset:] for offset in range(length)))


def schedule_text(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for field in value.values():
            yield from schedule_text(field)
    elif isinstance(value, list):
        for item in value:
            yield from schedule_text(item)


def policy_phrases(policy, wording):
    sources = [wording, *schedule_text(policy.get("repair_terms")), *schedule_text(policy.get("theft_terms")),
               *schedule_text(policy.get("liability_terms")), *schedule_text(policy.get("use_conditions"))]
    excerpts = {phrase for source in sources for phrase in phrases(words(source), PASSAGE_WORDS)}
    clause = policy.get("clause") or ""
    # Named clauses are policy citations, even when shorter than a passage.
    if any(char.isdigit() for char in clause) and any(char.isalpha() for char in clause):
        excerpts.add(words(clause))
    return excerpts


def copied_fields(fields, excerpts):
    lengths = {len(excerpt) for excerpt in excerpts}
    copied = []
    for field, value in fields:
        tokens = words(value)
        if any(phrase in excerpts for length in lengths for phrase in phrases(tokens, length)):
            copied.append(field)
    return copied


def cost_text(items, charges):
    yield from (("line item description", item.description) for item in items)
    yield from (("additional charge reason", charge.reason) for charge in charges)


def plan_text(plan):
    yield "incident narrative", plan.incident
    yield from cost_text(plan.cost_items, plan.additional_costs)
    if plan.estimate_discrepancy:
        yield "conflicting line item description", plan.estimate_discrepancy.estimate_description
    if plan.narrative_variation and plan.narrative_variation.conflicting_sentence:
        yield "contradictory incident statement", plan.narrative_variation.conflicting_sentence


def draft_text(draft):
    yield "claim narrative", draft.narrative
    yield from cost_text([], draft.additional_costs)
    for document in draft.documents:
        yield "document name", document.name
        if document.type == "repair_estimate":
            yield "garage name", document.garage_name
            yield "garage address", document.garage_address
        if document.type == "police_report":
            yield "police station", document.station
            yield "police report summary", document.summary
        elif document.type == "additional_cost_receipt":
            yield "receipt description", document.description
        elif document.type in {"incident_report", "witness_statement"}:
            yield "document author", document.author
            yield "incident evidence summary", document.summary
        else:
            yield from cost_text(document.line_items, [])


def stored_cost_text(document):
    yield "document name", document.get("name") or ""
    if document.get("type") == "repair_estimate":
        yield "garage name", document.get("garage_name") or ""
        yield "garage address", document.get("garage_address") or ""
    yield from (("line item description", item.get("description") or "") for item in document["line_items"])
