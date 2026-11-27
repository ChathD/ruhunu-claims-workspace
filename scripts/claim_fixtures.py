"""Regression fixtures for isolated tests and scripted demonstrations.

No runtime specialist imports this module or receives a scenario name.
These fixtures model structured documents; scanned-image OCR is a later stage.
"""
import random
import sqlite3
import uuid
from contextlib import closing
from decimal import Decimal
from pathlib import Path

from sqlalchemy import func, select

from backend.storage.sqlite_store import Assessment, Claim, History, Policy, digest
from backend.assurance.execution_manifest import execution_manifest
from backend.evidence.claim_history import ClaimHistory


class FixtureHistory(ClaimHistory):
    """Fixed comparables for isolated test databases and scripted debugging."""

    def records(self, session, claim):
        loss_date = claim.payload.get("loss_date")
        rows = session.scalars(select(History).where(History.owner_claim_id == claim.id).order_by(History.id)).all()
        return [{**row.payload, "id": row.id} for row in rows
                if not row.payload.get("settled_on") or (loss_date and row.payload["settled_on"] < loss_date)]

    def exists(self, session, identifier):
        return session.get(History, identifier) is not None

SCENARIOS = [
    ("ordinary_glass", "auto-approve"),
    ("ordinary_repair", "auto-approve"),
    ("exact_threshold", "auto-approve"),
    ("missing_estimate", "request more information"),
    ("unreadable_loss_date", "request more information"),
    ("high_value_low_net", "adjuster review"),
    ("high_value_missing_estimate", "request more information"),
    ("inactive_policy", "adjuster review"),
    ("uncovered_loss", "adjuster review"),
    ("over_policy_limit", "adjuster review"),
    ("cost_outlier", "adjuster review"),
    ("insufficient_history", "adjuster review"),
    ("duplicate_submission", "adjuster review"),
    ("unexplained_additional_cost", "request more information"),
    ("historical_document_reuse", "fraud escalation"),
    ("missing_itemisation", "request more information"),
]


def make_fixture(seed: int, index: int, scenario: str):
    rng = random.Random(f"{seed}:{index}")
    claim_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"motor-starter:{seed}:{index}"))
    short = claim_id[:8].upper()
    gross = Decimal(rng.randrange(95000, 175000, 1000))
    if scenario == "exact_threshold":
        gross = Decimal("250000.00")
    if scenario in {"high_value_low_net", "high_value_missing_estimate"}:
        gross = Decimal("250000.01")
    if scenario == "cost_outlier":
        gross = Decimal("240000.00")
    loss_type = "repair" if scenario == "ordinary_repair" else "glass"
    first = (gross * Decimal("0.6")).quantize(Decimal("0.01"))
    policy = {"policy_id": f"POL-{short}", "version": "2026.1", "registration": f"SYN-{short}",
              "starts_on": "2026-01-01", "ends_before": "2027-01-01", "covered_losses": ["glass", "repair"],
              "limit": "500000.00", "deductible": "10000.00", "clause": "SYN-MOTOR-04"}
    estimate = {"id": f"EST-{short}", "type": "repair_estimate", "name": "Itemised repair estimate.json",
               "garage_name": "Fixture Auto Works", "garage_address": "25 Fixture Road, Colombo",
               "registration": policy["registration"], "date": "2026-10-16", "total": str(gross),
               "line_items": [{"description": "Replacement parts", "amount": str(first)},
                              {"description": "Labour and installation", "amount": str(gross - first)}]}
    if scenario == "missing_itemisation":
        estimate["line_items"] = []
    if scenario == "unexplained_additional_cost":
        estimate["total"] = str(gross - 1000)
        estimate["line_items"][0]["amount"] = str(first - 1000)
    documents = [estimate]
    if scenario in {"missing_estimate", "high_value_missing_estimate"}:
        documents = []
    if scenario == "high_value_low_net":
        policy["deductible"] = "160000.00"
    if scenario == "inactive_policy":
        policy["ends_before"] = "2026-10-15"
    if scenario == "uncovered_loss":
        policy["covered_losses"] = ["repair"]
    if scenario == "over_policy_limit":
        policy["limit"] = str(gross - 1)
    claim = {"claim_id": claim_id, "revision": 1, "policy_id": policy["policy_id"], "registration": policy["registration"],
             "incident_id": f"INC-{short}", "loss_type": loss_type,
             "loss_date": None if scenario == "unreadable_loss_date" else "2026-10-15",
             "reported_date": "2026-10-16", "gross_amount": str(gross), "currency": "LKR",
             "narrative": "Collision damage to the front bumper." if loss_type == "repair" else "A stone cracked the front windscreen glass.",
             "documents": documents, "source": "Synthetic generator"}
    median = 75000 if scenario == "cost_outlier" else 150000
    count = 2 if scenario == "insufficient_history" else 8
    history = [{"id": f"H-{short}-{i}", "incident_id": f"OLD-{short}-{i}", "loss_type": loss_type,
                "settled_on": f"2026-09-{i + 1:02d}", "gross_cost": str(median + rng.randrange(-9000, 9001, 1000)),
                "currency": "LKR", "decision": "human-adjudicated settlement",
                "document_reference": estimate["id"] if scenario == "historical_document_reuse" and i < 2 else f"OLD-EST-{short}-{i}"}
               for i in range(count)]
    for row in history:
        amount = Decimal(row["gross_cost"])
        row["narrative"] = claim["narrative"]
        row["documents"] = [{"type": "repair_estimate", "total": str(amount), "line_items": [
            {"description": "Replacement parts", "amount": str(amount * Decimal("0.6"))},
            {"description": "Labour and installation", "amount": str(amount * Decimal("0.4"))},
        ]}]
        if scenario == "insufficient_history":
            row["documents"][0]["line_items"] = []
    return claim, policy, history


def generate_batch(store, labels_path: Path, seed: int | None = None, count: int = 16, batch_id: str | None = None):
    store.history = FixtureHistory()
    if not 1 <= count <= 160:
        raise ValueError("Generate between 1 and 160 claims per batch.")
    if seed is None:
        seed = random.SystemRandom().randrange(2147483648)
    batch_id = batch_id or str(uuid.uuid4())
    manifest = execution_manifest()
    from backend.model_settings import settings
    agentic = settings()[0]
    manifest = {**manifest, "workflow": "claude-intake-first-v3", "mode": "anthropic-tools" if agentic else "deterministic-tools"}
    # Retain deterministic fixture variation for tests and evaluation labels.
    fixture_seed = f"{seed}:{batch_id}"
    labels = []
    ids = []
    first_claim = None
    with store.session.begin() as session:
        offset = session.scalar(select(func.max(Claim.sequence))) or 0
        for index in range(count):
            scenario, expected = SCENARIOS[index % len(SCENARIOS)]
            claim, policy, history = make_fixture(fixture_seed, index, scenario)
            if scenario == "duplicate_submission" and first_claim is not None:
                claim["incident_id"] = first_claim["incident_id"]
                claim["registration"] = first_claim["registration"]
            if first_claim is None:
                first_claim = claim
            assessment_id = str(uuid.uuid4())
            ids.append(assessment_id)
            session.add(Policy(id=policy["policy_id"], payload=policy))
            session.add(Claim(id=claim["claim_id"], sequence=offset + index + 1, batch_id=batch_id,
                              policy_id=policy["policy_id"], payload=claim, payload_hash=digest(claim)))
            session.flush()
            session.add(Assessment(id=assessment_id, claim_id=claim["claim_id"], execution=manifest,
                                   context={"findings": {}, "skipped": {}, "environment": "synthetic", "rules": "provisional-motor-v1",
                                            "workflow": manifest["workflow"], "policy_search_mode": "generated-policy", "version": 0}))
            for row in history:
                session.add(History(id=row["id"], owner_claim_id=claim["claim_id"], payload=row))
            store.add_event(session, assessment_id, "submitted", {"status": "queued"})
            labels.append((assessment_id, batch_id, expected, scenario, seed))
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(labels_path)) as connection, connection:
        connection.execute("CREATE TABLE IF NOT EXISTS labels(assessment_id TEXT PRIMARY KEY,batch_id TEXT,expected TEXT,scenario TEXT,seed INTEGER)")
        connection.executemany("INSERT INTO labels (assessment_id,batch_id,expected,scenario,seed) VALUES(?,?,?,?,?)", labels)
    return {"batch_id": batch_id, "count": count, "assessment_ids": ids}
