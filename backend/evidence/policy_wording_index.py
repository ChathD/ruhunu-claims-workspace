"""Store versioned wording, index document passages and retrieve scoped citations for coverage."""
import base64
import hashlib
import io
import threading
import uuid
from functools import lru_cache
from pathlib import Path

from sqlalchemy import delete, select

from backend.storage.sqlite_store import PolicyBinding, PolicyChunk, PolicyDocument, WorkspaceSetting

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_LOCK = threading.Lock()
GENERATED_WORDING_PRODUCT = "SYN-MOTOR-GENERATED"
SYNTHETIC_TEXT = """SYNTHETIC DEMONSTRATION WORDING — SYN-MOTOR-04, version 2026.1.
This fixture is not a real insurance contract. The individual structured policy schedule
determines the insured vehicle, effective dates, covered loss types, limit and deductible.
Glass damage including a cracked windscreen and collision repair to a bumper are covered
only when the schedule lists that loss type and the incident falls within its effective dates.
The schedule end date is exclusive. Losses outside that period or unlisted loss types are excluded.
An itemised repair estimate with the insured registration, estimate date, garage name and
garage address is required. Itemised proposed work must reconcile with the estimate total.
The supported payable is min(max(gross minus deductible, zero), policy limit). Claims exceeding
the policy limit or with no positive payable require adjuster review. These demonstration terms
never authorize a payment. Suspected fraud requires verification and human assessment.
"""


@lru_cache(maxsize=1)
def encoder():
    from sentence_transformers import SentenceTransformer
    cache = Path(__file__).resolve().parents[2] / ".local" / "models"
    hub = cache / ("models--" + MODEL.replace("/", "--"))
    ref = hub / "refs" / "main"
    revision = ref.read_text().strip() if ref.is_file() else None
    snapshot = hub / "snapshots" / revision if revision else None
    local = snapshot is not None and snapshot.is_dir()
    model = SentenceTransformer(str(snapshot) if local else MODEL, device="cpu", trust_remote_code=False,
                                cache_folder=str(cache), local_files_only=local)
    model.claims_revision = revision if local else getattr(model[0].auto_model.config, "_commit_hash", None)
    return model


def document_record(*, title, product, version, filename, media_type, content):
    raw = base64.b64decode(content, validate=True)
    if not raw or len(raw) > 10 * 1024 * 1024:
        raise ValueError("Upload a nonempty file of at most 10 MB")
    if media_type not in {"application/pdf", "text/plain"}:
        raise ValueError("Only text-based PDF and UTF-8 TXT are supported")
    if media_type == "application/pdf" and not raw.startswith(b"%PDF-"):
        raise ValueError("File is not a PDF")
    if media_type == "text/plain":
        raw.decode("utf-8-sig")
    return PolicyDocument(id=str(uuid.uuid4()), title=title, product=product, version=version,
                          filename=filename, media_type=media_type, content=content,
                          file_hash=hashlib.sha256(raw).hexdigest(), status="queued", embedding={})


def create_document(store, **fields):
    doc = document_record(**fields)
    with store.session.begin() as session:
        session.add(doc)
    return doc.id


def generated_wording_matches(policy_id):
    return (PolicyDocument.product == GENERATED_WORDING_PRODUCT) & (PolicyDocument.filename == policy_id + ".txt")


def generated_wording_for_policy(session, policy_id):
    return session.scalar(select(PolicyDocument).join(PolicyBinding, PolicyBinding.document_id == PolicyDocument.id)
        .where(PolicyBinding.policy_id == policy_id, generated_wording_matches(policy_id)))


def ensure_synthetic_default(store):
    """Install the explicitly labelled demo configuration only on first setup."""
    with store.session() as session:
        if session.get(WorkspaceSetting, "generator_wording"):
            return
    identifier = create_document(store, title="Synthetic motor wording (demonstration only)",
        product="SYN-MOTOR", version="2026.1", filename="synthetic-motor.txt", media_type="text/plain",
        content=base64.b64encode(SYNTHETIC_TEXT.encode()).decode())
    with store.session.begin() as session:
        session.add(WorkspaceSetting(key="generator_wording", value=identifier))


def index_document(store, identifier):
    with store.session.begin() as session:
        doc = session.get(PolicyDocument, identifier)
        if doc is None or doc.status == "ready":
            return
        doc.status, doc.error = "indexing", None
        raw, media_type = base64.b64decode(doc.content), doc.media_type
    try:
        if media_type == "application/pdf":
            from pypdf import PdfReader
            pages = [(i + 1, page.extract_text() or "") for i, page in enumerate(PdfReader(io.BytesIO(raw)).pages)]
            if not pages or any(not text.strip() for _, text in pages):
                raise ValueError("OCR required: one or more PDF pages contain no extractable text")
        else:
            pages = [(1, raw.decode("utf-8-sig"))]
        if not any(text.strip() for _, text in pages):
            raise ValueError("Document contains no readable text")
        with MODEL_LOCK:
            model = encoder()
            chunks = []
            for page, text in pages:
                tokens = model.tokenizer.encode(text, add_special_tokens=False)
                for start in range(0, len(tokens), 150):
                    chunks.append((page, model.tokenizer.decode(tokens[start:start + 180], skip_special_tokens=True)))
            vectors = model.encode([text for _, text in chunks], normalize_embeddings=True).tolist()
            config = model[0].auto_model.config
            metadata = {"model": MODEL, "revision": getattr(model, "claims_revision", getattr(config, "_commit_hash", None)),
                        "dimension": len(vectors[0]), "normalized": True, "chunk_tokens": 180, "overlap": 30}
        with store.session.begin() as session:
            session.execute(delete(PolicyChunk).where(PolicyChunk.document_id == identifier))
            for index, ((page, text), vector) in enumerate(zip(chunks, vectors)):
                session.add(PolicyChunk(id=f"{identifier}:{index}", document_id=identifier, page=page, text=text, vector=vector))
            doc = session.get(PolicyDocument, identifier)
            doc.status, doc.embedding = "ready", metadata
    except Exception as error:
        with store.session.begin() as session:
            doc = session.get(PolicyDocument, identifier)
            doc.status = "failed"
            doc.error = str(error)[:300] if isinstance(error, (ValueError, UnicodeError)) else f"Indexing failed ({type(error).__name__}); retry after checking local model availability"


def search_document(store, document_id, query):
    with store.session() as session:
        doc = session.get(PolicyDocument, document_id) if document_id else None
        if doc is None or doc.status != "ready":
            return {"status": "unavailable", "passages": [], "reason": "Bound policy wording is absent or not indexed"}
        chunks = session.scalars(select(PolicyChunk).where(PolicyChunk.document_id == doc.id)).all()
    with MODEL_LOCK:
        model = encoder()
        if getattr(model, "claims_revision", getattr(model[0].auto_model.config, "_commit_hash", None)) != doc.embedding.get("revision"):
            raise ValueError("Embedding revision differs from indexed wording; indexing configuration must be restored")
        vector = model.encode(query, normalize_embeddings=True).tolist()
    ranked = sorted(((sum(a * b for a, b in zip(vector, chunk.vector)), chunk) for chunk in chunks),
                    key=lambda item: item[0], reverse=True)[:5]
    return {"status": "ready", "document_id": doc.id, "title": doc.title, "version": doc.version,
            "file_hash": doc.file_hash, "embedding": doc.embedding,
            "passages": [{"source": "chunk:" + chunk.id, "page": chunk.page, "text": chunk.text,
                          "score": round(score, 5)} for score, chunk in ranked]}


def uploaded_document_ids(store):
    """Snapshot uploaded wording; never mix the old demonstration fixture into it."""
    with store.session() as session:
        docs = session.scalars(select(PolicyDocument).order_by(PolicyDocument.created_at)).all()
    return [doc.id for doc in docs if doc.product != GENERATED_WORDING_PRODUCT
            and not (doc.filename == "synthetic-motor.txt" and doc.product == "SYN-MOTOR")]


def search_uploaded_documents(store, document_ids, query):
    passages, unavailable = [], []
    documents = []
    for identifier in document_ids:
        result = search_document(store, identifier, query)
        if result["status"] != "ready":
            unavailable.append(identifier)
            continue
        metadata = {key: result[key] for key in ("document_id", "title", "version", "file_hash")}
        documents.append(metadata)
        passages.extend({**passage, **metadata} for passage in result["passages"])
    passages.sort(key=lambda passage: passage["score"], reverse=True)
    return {"status": "ready" if passages else "unavailable", "documents": documents,
            "passages": passages[:5], "unavailable_document_ids": unavailable,
            "note": "Search similarity is not proof that wording applies. Check the document's subject, scope and exclusions against the claim."}


def index_pending(store):
    with store.session() as session:
        identifiers = session.scalars(select(PolicyDocument.id).where(PolicyDocument.status.in_(["queued", "indexing"]))).all()
    for identifier in identifiers:
        index_document(store, identifier)
