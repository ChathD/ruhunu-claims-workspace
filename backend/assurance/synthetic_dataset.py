"""Export committed synthetic text and its separately stored labels as JSON or CSV."""
import csv
import io
import json
import sqlite3
from contextlib import closing
from zipfile import ZIP_DEFLATED, ZipFile

from sqlalchemy import select

from backend.storage.sqlite_store import Assessment, Claim, EvaluationLabel, SyntheticClaimAttachment, SyntheticClaimSchedule


def generated_scenario_title(labels_path, assessment_id, *, store=None):
    if labels_path is None:
        with store.session() as session:
            return session.scalar(select(EvaluationLabel.scenario).where(EvaluationLabel.assessment_id == assessment_id))
    if not labels_path.exists():
        return None
    with closing(sqlite3.connect(labels_path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        if "scenario" not in {column[1] for column in connection.execute("PRAGMA table_info(labels)")}:
            return None
        row = connection.execute("SELECT scenario FROM labels WHERE assessment_id=?", (assessment_id,)).fetchone()
        return row[0] if row else None


def dataset_records(store, labels_path, batch_id):
    if labels_path is None:
        with store.session() as session:
            rows = session.execute(select(Claim, Assessment, EvaluationLabel).join(Assessment, Assessment.claim_id == Claim.id)
                .join(EvaluationLabel, EvaluationLabel.assessment_id == Assessment.id)
                .where(Claim.batch_id == batch_id, EvaluationLabel.batch_id == batch_id).order_by(Claim.sequence)).all()
        return [{"claim_id": claim.id, "scenario_type": assessment.execution["generation_scenario"],
                 "unstructured_inputs": claim.payload["unstructured_inputs"], "ground_truth_labels": label.ground_truth}
                for claim, assessment, label in rows if label.ground_truth is not None]
    if not labels_path.exists():
        return []
    with closing(sqlite3.connect(labels_path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        if "ground_truth" not in {column[1] for column in connection.execute("PRAGMA table_info(labels)")}:
            return []
        labels = dict(connection.execute(
            "SELECT assessment_id, ground_truth FROM labels WHERE batch_id=? AND ground_truth IS NOT NULL", (batch_id,)
        ))
    if not labels:
        return []
    with store.session() as session:
        claims = session.execute(select(Claim, Assessment).join(Assessment, Assessment.claim_id == Claim.id)
                                 .where(Claim.batch_id == batch_id, Assessment.id.in_(labels)).order_by(Claim.sequence)).all()
    return [{"claim_id": claim.id, "scenario_type": assessment.execution["generation_scenario"],
             "unstructured_inputs": claim.payload["unstructured_inputs"],
             "ground_truth_labels": json.loads(labels[assessment.id])} for claim, assessment in claims]


def dataset_csv(records):
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    label_fields = list(records[0]["ground_truth_labels"])
    writer.writerow(["claim_id", "scenario_type", "raw_combined_text", *label_fields])
    for record in records:
        raw = "\n\n".join(f"{source}:\n{value}" for source, value in record["unstructured_inputs"].items())
        fields = [record["ground_truth_labels"].get(field) for field in label_fields]
        writer.writerow([record["claim_id"], record["scenario_type"], raw, *(csv_label(value) for value in fields)])
    return output.getvalue()


def csv_label(value):
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return "|".join(value)
    return json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value


def dataset_zip(store, records):
    labels = {record["claim_id"]: record["ground_truth_labels"] for record in records}
    with store.session() as session:
        cases = session.execute(select(Claim, SyntheticClaimSchedule).join(
            SyntheticClaimSchedule, Claim.id == SyntheticClaimSchedule.claim_id,
        ).where(Claim.id.in_(labels)).order_by(Claim.sequence)).all()
        if not cases:
            return None
        attachments = session.scalars(select(SyntheticClaimAttachment).where(
            SyntheticClaimAttachment.claim_id.in_(row.id for row, _ in cases)
        ).order_by(SyntheticClaimAttachment.id)).all()

    schedules = {}
    submissions, links, truth = [], [], []
    for claim, schedule in cases:
        schedules[schedule.schedule_id] = schedule.payload
        submission = {key: value for key, value in claim.payload.items()
                      if key not in {"documents", "unstructured_inputs"} and value is not None}
        submissions.append({**submission, "fnol_transcript": claim.payload["unstructured_inputs"]["fnol_transcript"]})
        links.append({"claim_id": claim.id, "schedule_id": schedule.schedule_id})
        truth.append({"claim_id": claim.id, **labels[claim.id]})

    files = {
        "case_links.json": links,
        "policy_schedules.json": list(schedules.values()),
        "claim_submissions.json": submissions,
        "attachments.json": [{"id": row.id, "claim_id": row.claim_id, **row.payload} for row in attachments],
        "ground_truth.json": truth,
    }
    output = io.BytesIO()
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, json.dumps(content, indent=2, ensure_ascii=False))
    return output.getvalue()
