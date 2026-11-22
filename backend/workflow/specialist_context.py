"""Build a role-specific view of the saved investigation findings and policy references."""
import copy

DEPENDENCIES = {
    "intake": (),
    "coverage": ("intake",),
    "historical": ("intake", "coverage"),
    "fraud_initial": ("intake",),
    "fraud_followup": ("intake", "coverage", "historical", "fraud_initial"),
}


def generated_policy_context(context):
    """Drop uploaded wording references from new synthetic model inputs."""
    return {**{key: value for key, value in context.items() if key not in {"policy_corpus_ids", "wording_document_id"}},
            "policy_search_mode": "bound-synthetic" if context.get("synthetic_wording_document_id") else "generated-policy"}


def specialist_context(context, stage, version=0):
    findings = context.get("findings", {})
    view = {"version": context.get("version", version),
            "findings": {name: copy.deepcopy(findings[name])
                         for name in DEPENDENCIES[stage] if name in findings}}
    if stage == "coverage":
        for name in ("policy_search_mode", "policy_corpus_ids", "wording_document_id", "synthetic_wording_document_id"):
            if name in context:
                view[name] = copy.deepcopy(context[name])
    return view
