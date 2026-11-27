"""Build a role-specific view of the saved investigation findings and policy references."""
import copy

DEPENDENCIES = {
    "intake": (),
    "coverage": ("intake", "historical"),
    "historical": ("intake",),
    "fraud_initial": ("intake",),
    "fraud_followup": ("intake", "coverage", "historical", "fraud_initial"),
}


def is_specialist_stage(stage):
    prefix, separator, role = stage.partition(":")
    if separator and (not prefix.startswith("r") or not prefix[1:].isdigit()):
        return False
    return (role if separator else prefix) in {*DEPENDENCIES, "fraud"}


def generated_policy_context(context):
    """Keep explicit selections and scope older synthetic investigations to their saved terms."""
    mode = ("selected-upload" if context.get("selected_policy") else
            "bound-synthetic" if context.get("synthetic_wording_document_id") else "generated-policy")
    return {**{key: value for key, value in context.items() if key not in {"policy_corpus_ids", "wording_document_id"}},
            "policy_search_mode": mode}


def specialist_context(context, stage, version=0):
    findings = context.get("findings", {})
    view = {"version": context.get("version", version),
            "findings": {name: copy.deepcopy(findings[name])
                         for name in DEPENDENCIES[stage] if name in findings}}
    if stage == "coverage":
        for name in ("policy_search_mode", "policy_corpus_ids", "wording_document_id", "synthetic_wording_document_id", "selected_policy"):
            if name in context:
                view[name] = copy.deepcopy(context[name])
        if (schedule := context.get("policy_schedule")) and schedule.get("id"):
            view["policy_schedule"] = copy.deepcopy(schedule)
    return view
