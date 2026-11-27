"""Compare published routes with separate synthetic labels and build the confusion matrix."""
import sqlite3
from contextlib import closing

from sqlalchemy import select

from backend.storage.sqlite_store import Adjudication, Assessment, DecisionRecord, EvaluationLabel, FraudRoutingCorrection, InvestigationRevision, digest
from backend.claim_schemas import SCENARIO_ROUTES
from backend.workflow.investigation_retry import can_reprocess


def evaluate(store, labels_path, batch_id=None):
    if labels_path is None:
        query = select(EvaluationLabel.__table__).where(EvaluationLabel.expected.is_not(None)).order_by(EvaluationLabel.assessment_id)
        if batch_id is not None:
            query = query.where(EvaluationLabel.batch_id == batch_id)
        with store.session() as session:
            rows = session.execute(query).mappings().all()
    elif labels_path.exists():
        with closing(sqlite3.connect(labels_path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute("SELECT * FROM labels WHERE expected IS NOT NULL" + (" AND batch_id=?" if batch_id else ""),
                                      (batch_id,) if batch_id else ()).fetchall()
    else:
        rows = []
    routes = [route.value for route in SCENARIO_ROUTES.values()]
    matrix = [[0] * len(routes) for _ in routes]
    errors = []
    pending = 0
    with store.session() as session:
        assessments = session.scalars(select(Assessment)).all()
        existing = {row.id for row in assessments}
        rows = [row for row in rows if row["assessment_id"] in existing]
        predictions = {row.assessment_id: row.payload["route"] for row in session.scalars(select(DecisionRecord))}
        predictions.update({row.assessment_id: row.payload["route"] for row in session.scalars(select(FraudRoutingCorrection))})
        replacements = {row.source_id: row.target_id for row in session.scalars(select(InvestigationRevision))}
        adjudicated = set(session.scalars(select(Adjudication.assessment_id)))
        interrupted = {row.id for row in assessments if can_reprocess(row, adjudicated=row.id in adjudicated)}
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
    class_metrics = []
    for index, label in enumerate(SCENARIO_ROUTES):
        support = sum(matrix[index])
        predicted_count = sum(row[index] for row in matrix)
        true_positives = matrix[index][index]
        total = support + predicted_count
        class_metrics.append({
            "route": label,
            "support": support,
            "precision": true_positives / predicted_count if predicted_count else None,
            "recall": true_positives / support if support else None,
            "f1": 2 * true_positives / total if total else None,
        })
    return {"routes": routes, "matrix": matrix, "rows": "expected", "columns": "predicted",
            "class_metrics": class_metrics,
            "evaluated": len(rows) - pending, "pending": pending, "complete": pending == 0,
            "label_hash": digest([dict(r) for r in rows]), "errors": errors,
            "scope": "Synthetic scenario validation. LLM preview labels assume no comparable history. Live triage can use eligible prior claims."}
