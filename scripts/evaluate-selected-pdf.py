"""Compare the selected-PDF workflows in isolated stores with frozen claims."""
import argparse
import base64
import io
import json
import os
import sqlite3
import sys
import time
import uuid
from collections import defaultdict
from contextlib import closing
from pathlib import Path

from dotenv import load_dotenv
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.assurance.claim_audit_export import export_audit, verify_audit
from backend.claim_schemas import Route
from backend.evidence.policy_wording_index import create_document, index_document
from backend.evidence.synthetic_claim_text import extract_claim_text, render_claim_text
from backend.generation.synthetic_claim_generator import commit_batch
from backend.model_settings import settings
from backend.storage.sqlite_store import Account, Assessment, Claim, DecisionRecord, PolicyDocument, Store, digest
from backend.workflow.claims_workflow import ClaimsWorkflow
from backend.workflow.selected_policy import generate_policy_schedule, queue_with_policy, selectable_policy_pdf
from backend.workflow.versions import V3


LABEL_VERSION = "motor-pdf-pilot-1"
REGISTRATION = "SYN-WP-CAB-2801"
REUSED_REFERENCE = "PILOT-SAMPLE-8842"
REVIEW, INFO, FRAUD = Route.REVIEW.value, Route.INFO.value, Route.FRAUD.value


def stable_id(name):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"ruhunu/{LABEL_VERSION}/{name}"))


def policy_pdf(lines):
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({
        NameObject("/F1"): DictionaryObject({NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"), NameObject("/BaseFont"): NameObject("/Helvetica")})})})
    contents = DecodedStreamObject()
    contents.set_data(("BT /F1 10 Tf 14 TL 40 744 Td " + " T* ".join(f"({line}) Tj" for line in lines) + " ET").encode("ascii"))
    page[NameObject("/Contents")] = writer._add_object(contents)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def submitted_claim(case_id, change="standard", *, prior_index=None):
    registration = REGISTRATION if prior_index is None else f"SYN-WP-CAB-{3100 + prior_index}"
    loss_date = None if change == "missing-date" else "2025-10-15"
    total = "96000.00" if prior_index is None else f"{90000 + prior_index * 2000}.00"
    narrative = "A van reversed into my parked car in Negombo, damaging the left rear bumper."
    claim_id = stable_id(case_id)
    docs = [{"type": "repair_estimate", "id": f"DOC-{claim_id}-1", "name": "Negombo quotation",
             "registration": registration, "date": "2025-10-16", "garage_name": "Pilot Auto Works",
             "garage_address": "Sample Road, Negombo", "total": total,
             "line_items": [{"description": "Rear bumper repair and fitting", "amount": "44000.00" if change == "cost-mismatch" else total}]}]
    if change in {"police-filed", "historic-reuse"}:
        docs.append({"type": "police_report", "id": f"DOC-{claim_id}-2", "name": "Traffic Branch extract", "registration": registration,
                     "station": "Negombo Traffic Branch", "date": "2025-10-16",
                     "summary": "The vehicle was stationary when struck by a reversing van. Left rear bumper damage was observed."})
    if change in {"historic-reuse", "prior-reuse"}:
        docs[0]["document_reference"] = REUSED_REFERENCE
    if change == "fictional-number":
        narrative += " I read a different reference number from an old, unrelated receipt during the call."
    if change == "notice-only":
        narrative += " I notified the Traffic Branch and filed the claim the next day."
    if change == "possible-conflict":
        narrative += " The van rolled a short distance before the driver stopped and remained at the scene."
    payload = {"claim_id": claim_id, "incident_id": stable_id(case_id + "/incident"), "revision": 1,
               "source": "Synthetic generator", "policy_id": f"PILOT-FICTIONAL-{claim_id[:8]}", "registration": registration,
               "currency": "LKR", "loss_type": "repair", "loss_date": loss_date, "reported_date": "2025-10-17",
               "gross_amount": total, "additional_costs": [], "narrative": narrative, "documents": docs}
    payload["unstructured_inputs"] = render_claim_text(payload, {"full_name": "Fictional Pilot Driver"}, {}, 0.2)
    payload.update(extract_claim_text(payload["unstructured_inputs"]))
    return payload


def fixtures():
    for policy in ("allianz", "fairfirst", "ipid"):
        for change in ("standard", "missing-date", "cost-mismatch", "notice-only", "fictional-number", "possible-conflict"):
            yield policy, change, INFO if change == "missing-date" else REVIEW, ["Claim form"] if change == "missing-date" else []
    yield "control", "standard", INFO, ["Police report"]
    yield "control", "missing-date", INFO, ["Claim form"]
    yield "control", "police-filed", REVIEW, []
    yield "control", "historic-reuse", FRAUD, []
    yield "control", "cost-mismatch", INFO, ["Police report"]
    yield "control", "fictional-number", INFO, ["Police report"]


def default_labels(cases):
    return {"suite": LABEL_VERSION, "reviewed_by": None, "cases": {
        f"{policy}/{change}": {"route": route, "documents": documents,
                                "pages": [1] if policy == "control" and "Police report" in documents else [],
                                "coverage_status": ("UNRESOLVED" if policy != "control" and change != "missing-date" else None),
                                "historical_lead": change == "historic-reuse"}
        for policy, change, route, documents in cases}}


def seed_history(store):
    reviewer_id = stable_id("reviewer")
    with store.session.begin() as session:
        session.add(Account(id=reviewer_id, email="fixture-reviewer@pilot.local", password_hash="disabled", role="staff"))
        for index in range(1, 6):
            claim = submitted_claim(f"history/{index}", "prior-reuse" if index < 3 else "standard", prior_index=index)
            identifier = stable_id(f"assessment/{index}")
            session.add(Claim(id=claim["claim_id"], sequence=index, batch_id="pilot-history", policy_id=claim["policy_id"],
                              payload=claim, payload_hash=digest(claim), received_at="2025-10-20T00:00:00+00:00"))
            session.flush()
            session.add(Assessment(id=identifier, claim_id=claim["claim_id"], status="complete", context={"findings": {}}, execution={"workflow": V3}))
            session.flush()
            decision = {"route": REVIEW, "justification": "Pilot fixture triage decision."}
            session.add(DecisionRecord(assessment_id=identifier, payload=decision, payload_hash=digest(decision),
                                       created_at="2025-10-21T00:00:00+00:00"))
    return reviewer_id


def index_policies(store, inputs):
    docs = {key: (path.name, path.read_bytes()) for key, path in inputs.items()}
    docs["control"] = ("completed-schedule-fixture.pdf", policy_pdf([
        "Motor policy contract and completed individual schedule",
        f"Insured registration: {REGISTRATION}. Cover type: Comprehensive.",
        "Period of insurance from 01/01/2025 to 31/12/2025 inclusive.",
        "Section I: Accidental collision damage to the insured vehicle is covered.",
        "For covered collision claims the insured must submit a police report.",
    ]))
    selected = {}
    for key, (filename, content) in docs.items():
        identifier = create_document(store, title=key, product="PILOT-MOTOR", version=LABEL_VERSION,
                                     filename=filename, media_type="application/pdf", content=base64.b64encode(content).decode())
        index_document(store, identifier)
        with store.session() as session:
            doc = session.get(PolicyDocument, identifier)
            if not selectable_policy_pdf(doc):
                raise RuntimeError(f"{filename}: {doc.error or 'Indexing did not finish'}")
            selected[key] = identifier
            print(f"Indexed {key}: {doc.embedding['page_count']} pages, {len(doc.embedding['ocr_pages'])} OCR pages", flush=True)
    return selected


def run_case(store, workflow, directory, doc_id, reviewer_id, case_id, claim, model, expected):
    batch = stable_id(case_id + "/batch")
    result = commit_batch(store, directory / "labels.sqlite3", {"batch_id": batch, "model": model, "fixtures": [
        {"scenario": case_id, "scenario_type": "architecture_pilot", "claim": claim, "ground_truth_labels": {}}]})
    identifier = result["assessment_ids"][0]
    generate_policy_schedule(store, identifier, doc_id, reviewer_id)
    queue_with_policy(store, identifier, doc_id, reviewer_id)
    started = time.monotonic()
    try:
        workflow.run(identifier)
        audit = export_audit(store, identifier)
    except (KeyError, ValueError, RuntimeError) as error:
        return {"complete": False, "route_correct": False, "correct": False, "safe": False, "error": type(error).__name__,
                "seconds": round(time.monotonic() - started, 3)}
    decision = audit["decision"] or {}
    route = decision.get("route")
    requests = sorted(entry["document"] for entry in decision.get("information_required", []))
    findings = audit["context"]["findings"]
    coverage = findings.get("coverage", {}).get("details", {})
    refs = set(coverage.get("policy_rag", {}).get("applicable_passages", []))
    pages = {passage["page"] for call in audit["retrievals"]
             for passage in call["response"].get("result", {}).get("passages", []) if passage["source"] in refs}
    lead = any(call["tool"] == "verify_historical_leads" and call["response"].get("verified") for call in audit["retrievals"])
    safe = (decision.get("payable_amount") is None and (route != FRAUD or expected["route"] == FRAUD)
            and set(requests).issubset(expected["documents"]))
    evidence_correct = (requests == sorted(expected["documents"]) and set(expected["pages"]).issubset(pages)
                        and (not expected.get("coverage_status") or coverage.get("coverage_status") == expected["coverage_status"])
                        and (not expected["historical_lead"] or lead))
    usage = [call["response"].get("usage", {}) for call in audit["model_calls"]]
    complete = bool(route) and verify_audit(audit)
    return {"complete": complete, "route": route, "route_correct": route == expected["route"],
            "correct": bool(complete and route == expected["route"] and evidence_correct and safe), "safe": safe,
            "documents": requests, "pages": sorted(pages), "coverage_status": coverage.get("coverage_status"), "calls": len(usage),
            "input_tokens": sum(item.get("input_tokens", 0) for item in usage),
            "output_tokens": sum(item.get("output_tokens", 0) for item in usage),
            "seconds": round(time.monotonic() - started, 3), "assessment_id": identifier}


def compare(results, labels, live, repeat):
    totals = {}
    for version, cases in results.items():
        rows = [row for runs in cases.values() for row in runs]
        totals[version] = {"consistently_correct": sum(all(run["correct"] for run in runs) for runs in cases.values()),
                           **{key: sum(int(row.get(key, 0)) for row in rows) for key in
                              ("complete", "route_correct", "safe", "calls", "input_tokens", "output_tokens")},
                           "attempts": len(rows), "seconds": round(sum(row["seconds"] for row in rows), 3)}
    before, after = totals["v3"], totals["v4"]
    eligible = (live and repeat >= 3 and len(labels["cases"]) >= 24 and bool(labels["reviewed_by"])
                and after["consistently_correct"] > before["consistently_correct"]
                and after["route_correct"] >= before["route_correct"] and after["complete"] >= before["complete"]
                and after["safe"] == after["attempts"])
    return {"metrics": totals, "release_eligible": eligible}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("allianz", "fairfirst", "ipid"):
        parser.add_argument(f"--{key}", type=Path)
    parser.add_argument("--repeat", default=3, type=int)
    parser.add_argument("--live", action="store_true", help="Use the configured Claude API in the isolated evaluation stores")
    parser.add_argument("--labels", type=Path, help="Load the reviewed answer key")
    parser.add_argument("--export-labels", type=Path, help="Write the draft answer key for human review and exit")
    args = parser.parse_args()
    cases = list(fixtures())
    labels = default_labels(cases)
    if args.export_labels:
        args.export_labels.write_text(json.dumps(labels, indent=2) + "\n", encoding="utf-8")
        return
    inputs = {key: getattr(args, key) for key in ("allianz", "fairfirst", "ipid")}
    if not all(inputs.values()) or args.repeat < 1:
        parser.error("Supply all three sample PDFs and a positive repetition count")
    if args.labels:
        labels = json.loads(args.labels.read_text(encoding="utf-8"))
        if labels["suite"] != LABEL_VERSION or labels["cases"].keys() != default_labels(cases)["cases"].keys():
            parser.error("Answer key does not match the fixed case suite")
    load_dotenv(ROOT / ".env", override=False)
    os.environ["CLAIMS_LLM_MODE"] = "anthropic" if args.live else "deterministic"
    os.environ["CLAIMS_ENABLE_LANGFUSE"] = "0"
    if args.live and not os.getenv("ANTHROPIC_API_KEY"):
        parser.error("Set ANTHROPIC_API_KEY for a live comparison")
    model = settings()[1] if args.live else "deterministic-tools"
    target = ROOT / ".local" / "selected-pdf-evaluation" / str(uuid.uuid4())
    seed_path = target / "seed.sqlite3"
    seed = Store(seed_path)
    reviewer_id = seed_history(seed)
    selected = index_policies(seed, inputs)
    seed.engine.dispose()
    snapshots = {f"{policy}/{change}": submitted_claim(f"{policy}/{change}", change) for policy, change, *_ in cases}
    results = {version: defaultdict(list) for version in ("v3", "v4")}
    for repetition in range(1, args.repeat + 1):
        for version, records in results.items():
            os.environ["CLAIMS_SELECTED_PDF_WORKFLOW"] = version
            directory = target / f"{repetition}-{version}"
            directory.mkdir()
            destination = directory / "claims.sqlite3"
            with closing(sqlite3.connect(seed_path)) as source, closing(sqlite3.connect(destination)) as copy:
                source.backup(copy)
            store = Store(destination)
            workflow = ClaimsWorkflow(store, directory / "checkpoints.sqlite3")
            try:
                for case_id, claim in snapshots.items():
                    policy = case_id.split("/")[0]
                    records[case_id].append(run_case(store, workflow, directory, selected[policy], reviewer_id,
                                                     case_id, claim, model, labels["cases"][case_id]))
                print(f"Completed {version} repetition {repetition}", flush=True)
            finally:
                workflow.close()
                store.engine.dispose()
    report = {"suite": LABEL_VERSION, "answer_key_hash": digest(labels), "model": model, "repeat": args.repeat,
              "frozen_claim_hashes": {case_id: digest(claim) for case_id, claim in snapshots.items()},
              **compare(results, labels, args.live, args.repeat), "runs": results}
    destination = target / "comparison.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(destination), "metrics": report["metrics"], "release_eligible": report["release_eligible"]}, indent=2))


if __name__ == "__main__":
    main()
