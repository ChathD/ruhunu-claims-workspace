"""Define the SQLAlchemy records and SQLite store for claims, evidence, decisions and audit data."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import JSON, ForeignKey, Integer, String, UniqueConstraint, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


class Base(DeclarativeBase):
    pass


class Claim(Base):
    __tablename__ = "claims"
    id: Mapped[str] = mapped_column(primary_key=True)
    sequence: Mapped[int] = mapped_column(Integer, unique=True)
    batch_id: Mapped[str]
    policy_id: Mapped[str]
    received_at: Mapped[str] = mapped_column(default=now)
    payload: Mapped[dict] = mapped_column(JSON)
    payload_hash: Mapped[str]


class Policy(Base):
    __tablename__ = "policies"
    id: Mapped[str] = mapped_column(primary_key=True)
    payload: Mapped[dict] = mapped_column(JSON)


class Person(Base):
    __tablename__ = "people"
    id: Mapped[str] = mapped_column(primary_key=True)
    full_name: Mapped[str]
    district: Mapped[str]


class SyntheticPolicyHolder(Base):
    __tablename__ = "synthetic_policy_holders"
    policy_id: Mapped[str] = mapped_column(ForeignKey("policies.id"), primary_key=True)
    person_id: Mapped[str] = mapped_column(ForeignKey("people.id"), index=True)


class History(Base):
    __tablename__ = "historical_settlements"
    id: Mapped[str] = mapped_column(primary_key=True)
    owner_claim_id: Mapped[str] = mapped_column(index=True)
    payload: Mapped[dict] = mapped_column(JSON)


class Assessment(Base):
    __tablename__ = "assessments"
    id: Mapped[str] = mapped_column(primary_key=True)
    claim_id: Mapped[str] = mapped_column(ForeignKey("claims.id"), unique=True)
    status: Mapped[str] = mapped_column(default="queued")
    context: Mapped[dict] = mapped_column(JSON, default=dict)
    version: Mapped[int] = mapped_column(default=0)
    error: Mapped[str | None]
    finished_at: Mapped[str | None]
    execution: Mapped[dict] = mapped_column(JSON)


class AgentRun(Base):
    __tablename__ = "agent_findings"
    __table_args__ = (UniqueConstraint("assessment_id", "stage"),)
    id: Mapped[str] = mapped_column(primary_key=True)
    assessment_id: Mapped[str] = mapped_column(ForeignKey("assessments.id"), index=True)
    stage: Mapped[str]
    instruction: Mapped[str]
    input_context: Mapped[dict] = mapped_column(JSON)
    input_version: Mapped[int]
    finding: Mapped[dict] = mapped_column(JSON)
    context_after: Mapped[dict] = mapped_column(JSON)
    duration_ms: Mapped[int]
    created_at: Mapped[str] = mapped_column(default=now)


class ToolCall(Base):
    __tablename__ = "tool_calls"
    id: Mapped[int] = mapped_column(primary_key=True)
    assessment_id: Mapped[str] = mapped_column(index=True)
    stage: Mapped[str]
    tool: Mapped[str]
    request: Mapped[dict] = mapped_column(JSON)
    response: Mapped[dict] = mapped_column(JSON)
    response_hash: Mapped[str]
    created_at: Mapped[str] = mapped_column(default=now)


class DecisionRecord(Base):
    __tablename__ = "decisions"
    assessment_id: Mapped[str] = mapped_column(ForeignKey("assessments.id"), primary_key=True)
    payload: Mapped[dict] = mapped_column(JSON)
    payload_hash: Mapped[str]
    created_at: Mapped[str] = mapped_column(default=now)


class FraudRoutingCorrection(Base):
    __tablename__ = "fraud_routing_corrections"
    assessment_id: Mapped[str] = mapped_column(ForeignKey("decisions.assessment_id"), primary_key=True)
    payload: Mapped[dict] = mapped_column(JSON)
    payload_hash: Mapped[str]
    original_decision_hash: Mapped[str]
    created_at: Mapped[str] = mapped_column(default=now)


class HumanTask(Base):
    __tablename__ = "human_tasks"
    id: Mapped[str] = mapped_column(primary_key=True)
    assessment_id: Mapped[str] = mapped_column(ForeignKey("decisions.assessment_id"), unique=True)
    queue: Mapped[str]
    status: Mapped[str] = mapped_column(default="pending")
    reviewer: Mapped[str | None]
    acknowledged_at: Mapped[str | None]


class HumanAction(Base):
    __tablename__ = "human_actions"
    id: Mapped[int] = mapped_column(primary_key=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("human_tasks.id"))
    reviewer: Mapped[str]
    action: Mapped[str]
    created_at: Mapped[str] = mapped_column(default=now)


class WorkflowEvent(Base):
    __tablename__ = "workflow_events"
    id: Mapped[int] = mapped_column(primary_key=True)
    assessment_id: Mapped[str] = mapped_column(index=True)
    kind: Mapped[str]
    payload: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[str] = mapped_column(default=now)


class TraceSpan(Base):
    __tablename__ = "investigation_trace_spans"
    id: Mapped[str] = mapped_column(primary_key=True)
    assessment_id: Mapped[str] = mapped_column(ForeignKey("assessments.id"), index=True)
    trace_id: Mapped[str]
    parent_id: Mapped[str | None]
    name: Mapped[str]
    kind: Mapped[str]
    payload: Mapped[dict] = mapped_column(JSON)
    started_at: Mapped[str] = mapped_column(default=now)
    ended_at: Mapped[str | None]
    duration_ms: Mapped[int | None]
    error_type: Mapped[str | None]


class Account(Base):
    __tablename__ = "accounts"
    id: Mapped[str] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(unique=True)
    password_hash: Mapped[str]
    role: Mapped[str]


class LoginSession(Base):
    __tablename__ = "login_sessions"
    token_hash: Mapped[str] = mapped_column(primary_key=True)
    account_id: Mapped[str] = mapped_column(ForeignKey("accounts.id"))
    expires_at: Mapped[str]


class Ownership(Base):
    __tablename__ = "claim_ownership"
    assessment_id: Mapped[str] = mapped_column(ForeignKey("assessments.id"), primary_key=True)
    account_id: Mapped[str] = mapped_column(ForeignKey("accounts.id"), index=True)


class Attachment(Base):
    __tablename__ = "manual_attachments"
    id: Mapped[str] = mapped_column(primary_key=True)
    assessment_id: Mapped[str] = mapped_column(ForeignKey("assessments.id"), index=True)
    name: Mapped[str]
    media_type: Mapped[str]
    content: Mapped[str]


class Adjudication(Base):
    __tablename__ = "adjudications"
    assessment_id: Mapped[str] = mapped_column(ForeignKey("assessments.id"), primary_key=True)
    reviewer_id: Mapped[str] = mapped_column(ForeignKey("accounts.id"))
    outcome: Mapped[str]
    reason: Mapped[str]
    created_at: Mapped[str] = mapped_column(default=now)


class Store:
    def __init__(self, path: Path):
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{self.path.as_posix()}", connect_args={"check_same_thread": False, "timeout": 30})

        @event.listens_for(self.engine, "connect")
        def sqlite_settings(connection, record):
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA journal_mode=WAL")

        Base.metadata.create_all(self.engine)
        self.session = sessionmaker(self.engine, expire_on_commit=False)
        from backend.evidence.approved_claim_history import ApprovedClaimHistory
        self.history = ApprovedClaimHistory()

    def add_event(self, session, assessment_id, kind, payload):
        session.add(WorkflowEvent(assessment_id=assessment_id, kind=kind, payload=payload))


class PolicyDocument(Base):
    __tablename__ = "policy_documents"
    id: Mapped[str] = mapped_column(primary_key=True)
    title: Mapped[str]
    product: Mapped[str]
    version: Mapped[str]
    filename: Mapped[str]
    media_type: Mapped[str]
    content: Mapped[str]
    file_hash: Mapped[str]
    status: Mapped[str] = mapped_column(default="queued")
    error: Mapped[str | None]
    embedding: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[str] = mapped_column(default=now)


class PolicyChunk(Base):
    __tablename__ = "policy_chunks"
    id: Mapped[str] = mapped_column(primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("policy_documents.id"), index=True)
    page: Mapped[int]
    text: Mapped[str]
    vector: Mapped[list] = mapped_column(JSON)


class PolicyBinding(Base):
    __tablename__ = "policy_wording_bindings"
    policy_id: Mapped[str] = mapped_column(ForeignKey("policies.id"), primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("policy_documents.id"))


class WorkspaceSetting(Base):
    __tablename__ = "workspace_settings"
    key: Mapped[str] = mapped_column(primary_key=True)
    value: Mapped[str]


class ModelInvocation(Base):
    __tablename__ = "model_invocations"
    id: Mapped[str] = mapped_column(primary_key=True)
    assessment_id: Mapped[str] = mapped_column(ForeignKey("assessments.id"), index=True)
    stage: Mapped[str]
    payload: Mapped[dict] = mapped_column(JSON)
    response: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(default="started")
    created_at: Mapped[str] = mapped_column(default=now)


class AttachmentExtraction(Base):
    __tablename__ = "attachment_extractions"
    attachment_id: Mapped[str] = mapped_column(ForeignKey("manual_attachments.id"), primary_key=True)
    payload: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[str] = mapped_column(default=now)


class InvestigationRevision(Base):
    __tablename__ = "investigation_revisions"
    source_id: Mapped[str] = mapped_column(ForeignKey("assessments.id"), primary_key=True)
    target_id: Mapped[str] = mapped_column(ForeignKey("assessments.id"), unique=True)
