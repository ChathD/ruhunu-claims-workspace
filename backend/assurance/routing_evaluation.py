"""Compare published routes with separate synthetic labels and build the confusion matrix."""
import sqlite3
from contextlib import closing

from sqlalchemy import select

from backend.storage.sqlite_store import Adjudication, Assessment, DecisionRecord, FraudRoutingCorrection, InvestigationRevision, digest
from backend.claim_schemas import Route
from backend.workflow.investigation_retry import can_reprocess


def evaluate(store, labels_path, batch_id=None):
    if not labels_path.exists():
        return {"routes": [r.value for r in Route], "matrix": [[0] * 4 for _ in range(4)], "evaluated": 0, "pending": 0, "errors": []}
    with closing(sqlite3.connect(labels_path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute("SELECT * FROM labels" + (" WHERE batch_id=?" if batch_id else ""), (batch_id,) if batch_id else ()).fetchall()
    routes = [r.value for r in Route]
    matrix = [[0] * 4 for _ in range(4)]
    errors = []
    pending = 0
    with store.session() as session:
        predictions = {row.assessment_id: row.payload["route"] for row in session.scalars(select(DecisionRecord))}
        predictions.update({row.assessment_id: row.payload["route"] for row in session.scalars(select(FraudRoutingCorrection))})
        replacements = {row.source_id: row.target_id for row in session.scalars(select(InvestigationRevision))}
        adjudicated = set(session.scalars(select(Adjudication.assessment_id)))
        interrupted = {row.id for row in session.scalars(select(Assessment))
                       if can_reprocess(row, adjudicated=row.id in adjudicated)}
    for row in rows:
        assessment_id = row["assessment_id"]
        while assessment_id in replacements:
            assessment_id = replacements[assessment_id]
        prediction = predictions.get(assessment_id) if assessment_id not in interrupted else None
        if prediction is None:
            pending += 1
            continue
        matrix[routes.index(row["expected"])][routes.index(prediction)] += 1
        if prediction != row["expected"]:
            errors.append({"assessment_id": assessment_id, "expected": row["expected"], "predicted": prediction})
    return {"routes": routes, "matrix": matrix, "rows": "expected", "columns": "predicted",
            "evaluated": len(rows) - pending, "pending": pending, "complete": pending == 0,
            "label_hash": digest([dict(r) for r in rows]), "errors": errors,
            "scope": "Synthetic scenario validation. LLM preview labels assume no comparable history; live triage can use available approved claims."}
