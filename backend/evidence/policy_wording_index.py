"""Store versioned wording, index document passages and retrieve scoped citations for coverage."""
import base64
import hashlib
import os
import re
import uuid
from bisect import bisect_right
from functools import cache
from pathlib import Path
from typing import Literal

import tiktoken
from sqlalchemy import delete, select

from backend.assurance.investigation_tracing import operation_trace, trace_span
from backend.evidence.policy_embeddings import DIMENSIONS, MODEL, PolicyEmbeddingError, embed_texts
from backend.evidence.policy_pdf import extract_pdf_pages
from backend.evidence.policy_reranker import MODEL as RERANKER_MODEL, REVISION as RERANKER_REVISION, rerank_scores
from backend.storage.policy_vector_store import CANDIDATE_LIMIT, POSTGRES_STORAGE, STORAGE, rank_policy_passages, replace_policy_vectors
from backend.storage.postgres import begin_write
from backend.storage.sqlite_store import PolicyChunk, PolicyDocument, PolicyDocumentRemoval, PolicyPage, WorkspaceSetting

MAX_POLICY_PDF_BYTES = 12 * 1024 * 1024
GENERATED_WORDING_PRODUCT = "SYN-MOTOR-GENERATED"
PASSAGE_LIMIT = 5
CHUNK_TOKENS, OVERLAP_TOKENS = 200, 40
LEGACY_CHUNK_FORMAT = "gpt2-200-40-v1"
LEGACY_OCR_CHUNK_FORMAT = "gpt2-200-40-ocr-rows-v2"
CHUNK_FORMAT = "gpt2-200-40-context-v2"
OCR_CHUNK_FORMAT = "gpt2-200-40-ocr-tables-v4"
LKR_TOKEN = re.compile(
    r"\b(?:lkr|rs\.?)\s*([0-9o?]+(?:[ \t]*,[ \t]*[0-9o?]+)*(?:\.[0-9o?]+)?"
    r"(?:[ \t]+(?:thousand|million|billion))?)(?![\w?])(?:[ \t]*/-)?", re.I,
)
PolicyTopic = Literal["insuring_clause", "exclusions", "extensions", "claim_conditions", "schedule"]
TOPIC_TERMS = {
    "insuring_clause": "indemnify cover section insured loss own vehicle third party liability",
    "exclusions": "exceptions exclusions excluded unless limitations use driver licence impairment wear tear",
    "extensions": "endorsement extension optional special conditions add-on peril excess exception",
    "claim_conditions": "claim conditions notice notify police theft accident proof documents consent survey repair estimate fraud",
    "schedule": "schedule certificate proposal insurance period insured registration excess deductible endorsements limit sum insured",
}
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


def searchable_index(document):
    return (document.status == "ready" and document.embedding.get("model") == MODEL
            and document.embedding.get("dimension") == DIMENSIONS and document.embedding.get("storage") in {STORAGE, POSTGRES_STORAGE})


def current_index(document):
    if document.embedding.get("ocr_pages"):
        formats = (OCR_CHUNK_FORMAT, LEGACY_OCR_CHUNK_FORMAT)
    else:
        formats = (CHUNK_FORMAT, LEGACY_CHUNK_FORMAT)
    return searchable_index(document) and document.embedding.get("chunk_format") in formats


def embed_query(text):
    return embed_texts([text])[0]


def document_record(*, title, product, version, filename, media_type, content):
    raw = base64.b64decode(content, validate=True)
    if not raw or len(raw) > MAX_POLICY_PDF_BYTES:
        raise ValueError("Upload a nonempty file of at most 12 MiB")
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


def policy_document_kind(filename, first_pages):
    headings = " ".join(first_pages[:1600].casefold().split())
    if re.search(r"\bipid\b", filename.casefold()) or re.search(r"\binsurance\s*product\s*information\s*document\b", headings):
        return "product_summary"
    return "wording"


def document_kind(session, document):
    if kind := document.embedding.get("document_kind"):
        return kind
    # Indexes created before document kinds were stored still contain the opening pages.
    first_pages = session.scalars(select(PolicyChunk.text).where(PolicyChunk.document_id == document.id, PolicyChunk.page <= 2))
    return policy_document_kind(document.filename, "\n".join(first_pages))


def page_passages(text):
    """Reproduce the 140-word baseline used by the local retrieval benchmark."""
    words = list(re.finditer(r"\S+", text))
    for start in range(0, len(words), 110):
        end = min(start + 140, len(words))
        yield text[words[start].start():words[end - 1].end()]
        if end == len(words):
            break


@cache
def passage_tokenizer():
    model_dir = Path(__file__).resolve().parents[2] / ".local" / "models"
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(model_dir / "tiktoken"))
    return tiktoken.get_encoding("gpt2")


def token_passages(text):
    if not text.strip():
        return
    tokenizer = passage_tokenizer()
    tokens = tokenizer.encode_ordinary(text)
    _, offsets = tokenizer.decode_with_offsets(tokens)
    offsets.append(len(text))
    words, clauses = passage_breaks(text, offsets)
    start = 0
    while start < len(tokens):
        target = min(start + CHUNK_TOKENS, len(tokens))
        if target == len(tokens):
            end = target
        else:
            clause = clauses[bisect_right(clauses, target) - 1]
            word = words[bisect_right(words, target) - 1]
            end = clause if clause >= target - OVERLAP_TOKENS else word
            if end <= start:
                # A single unbroken token sequence can exceed the target window.
                end = bisect_right(offsets, offsets[target]) - 1
        passage = text[offsets[start]:offsets[end]]
        if passage.strip():
            yield passage
        if end == len(tokens):
            break
        overlap_target = max(0, end - OVERLAP_TOKENS)
        clause = clauses[bisect_right(clauses, overlap_target) - 1]
        word = words[bisect_right(words, overlap_target) - 1]
        overlap = clause if clause > start and clause >= overlap_target - OVERLAP_TOKENS else word
        start = overlap if overlap > start else end


def passage_breaks(text, offsets):
    words, clauses = [0], [0]
    amounts = iter(LKR_TOKEN.finditer(text))
    amount = next(amounts, None)
    for index, offset in enumerate(offsets[1:-1], 1):
        if offset == offsets[index - 1] or not (text[offset].isspace() or text[offset - 1].isspace()):
            continue
        # Keep the currency and its complete value together, including spacing artifacts in OCR.
        while amount and offset >= amount.end():
            amount = next(amounts, None)
        if amount and amount.start() < offset < amount.end():
            continue
        words.append(index)
        if "\n" in text[offset - 1:offset + 1] or (text[offset].isspace() and text[offset - 1] in ".!?;"):
            clauses.append(index)
    return words, clauses


def index_document(store, identifier):
    with store.session.begin() as session:
        begin_write(session)
        doc = session.get(PolicyDocument, identifier)
        if doc is None or current_index(doc):
            return
        doc.status, doc.error = "indexing", None
        raw, media_type, filename = base64.b64decode(doc.content), doc.media_type, doc.filename
    with operation_trace("policy.index", identifier, content={"document_id": identifier, "media_type": media_type}) as trace:
        try:
            with trace_span("policy.extract", kind="span") as extraction:
                pages, ocr_pages = extract_pdf_pages(raw) if media_type == "application/pdf" else ([(1, raw.decode("utf-8-sig"))], [])
                extraction.summary = {"page_count": len(pages), "ocr_page_count": len(ocr_pages)}
            passages = [(page, excerpt) for page, text in pages for excerpt in token_passages(text)]
            if not passages:
                raise ValueError("Document contains no readable text")
            chunk_format = OCR_CHUNK_FORMAT if ocr_pages else CHUNK_FORMAT
            chunks = [PolicyChunk(id=f"{identifier}:{chunk_format}:{index}", document_id=identifier, page=page, text=text)
                      for index, (page, text) in enumerate(passages)]
            metadata = {"page_count": len(pages), "ocr_pages": ocr_pages,
                        "document_kind": policy_document_kind(filename, "\n".join(text for _, text in pages[:2])),
                        "provider": "openai", "model": MODEL, "dimension": DIMENSIONS,
                        "storage": POSTGRES_STORAGE if store.database_url else STORAGE,
                        "normalized": True, "chunk_format": chunk_format, "chunk_tokens": CHUNK_TOKENS,
                        "overlap_tokens": OVERLAP_TOKENS, "tokenizer": "gpt2"}
            with trace_span("policy.embed", kind="chain", metadata={"chunk_count": len(chunks)}) as embedding:
                vectors = embed_texts([chunk.text for chunk in chunks])
                embedding.summary = {"chunk_count": len(chunks)}
            with store.session.begin() as session:
                begin_write(session)
                doc = session.get(PolicyDocument, identifier)
                if not current_index(doc):
                    if media_type == "application/pdf":
                        scanned = set(ocr_pages)
                        session.execute(delete(PolicyPage).where(PolicyPage.document_id == identifier))
                        session.add_all(PolicyPage(document_id=identifier, page=page, text=text, ocr=page in scanned)
                                        for page, text in pages)
                    session.add_all(chunks)
                    session.flush()
                    replace_policy_vectors(session, identifier, chunks, vectors)
                    doc.status, doc.error, doc.embedding = "ready", None, metadata
            trace.summary = {"status": "ready", "page_count": metadata["page_count"], "ocr_page_count": len(metadata["ocr_pages"]),
                             "chunk_count": len(chunks), "document_kind": metadata["document_kind"]}
        except Exception as error:
            trace.error_type = type(error).__name__
            trace.summary = {"status": "failed"}
            with store.session.begin() as session:
                begin_write(session)
                doc = session.get(PolicyDocument, identifier)
                if not current_index(doc):
                    doc.status = "failed"
                    doc.error = str(error)[:300] if isinstance(error, (ValueError, PolicyEmbeddingError)) else f"Indexing failed ({type(error).__name__}). Check the document and OpenAI configuration before retrying."


def search_candidates(store, document_id, query, *, topic=None, query_embedding=embed_query):
    with store.session() as session:
        doc = session.get(PolicyDocument, document_id) if document_id else None
        if doc is None or doc.status != "ready":
            return {"status": "unavailable", "passages": [], "reason": "Bound policy wording is absent or not indexed"}
        if not searchable_index(doc):
            return {"status": "unavailable", "passages": [], "reason": "Policy wording needs reindexing with OpenAI embeddings"}
        kind = document_kind(session, doc)
    vector = query_embedding(f"{query} {TOPIC_TERMS[topic]}" if topic else query)
    with trace_span("policy.vector_search", kind="retriever", metadata={"topic": topic},
                    content={"document_id": document_id, "query": query}, retain_content=False) as span:
        with store.session() as session:
            ranked = rank_policy_passages(session, document_id, vector, query if topic else "")
        span.summary = {"candidate_count": len(ranked)}
        span.output = [{"chunk_id": chunk.id, "page": chunk.page, "score": chunk.score} for chunk in ranked]
    scanned = set(doc.embedding.get("ocr_pages", []))
    return {"status": "ready", "document_id": doc.id, "title": doc.title, "version": doc.version,
            "file_hash": doc.file_hash, "embedding": doc.embedding, "document_kind": kind,
            **({"topic": topic} if topic else {}),
            "passages": [{"source": "chunk:" + chunk.id, "page": chunk.page, "text": chunk.text,
                          "document_kind": kind, **({"extraction": "ocr"} if chunk.page in scanned else {}),
                          "score": chunk.score} for chunk in ranked]}


def rerank_passages(query, candidates):
    status = "unneeded"
    if len(candidates) > 1:
        with trace_span("policy.rerank", metadata={"batch_size": len(candidates)},
                        content={"query": query, "candidates": [passage["source"] for passage in candidates]},
                        retain_content=False) as span:
            try:
                scores = rerank_scores(query, [passage["text"] for passage in candidates])
                order = sorted(zip(candidates, scores, strict=True), key=lambda pair: pair[1], reverse=True)
                candidates = [{**passage, "rerank_score": round(score, 5)} for passage, score in order]
                status = "applied"
            except Exception as error:
                span.error_type = type(error).__name__
                status = "fallback"
            span.summary = {"status": status, "result_count": min(PASSAGE_LIMIT, len(candidates))}
            span.output = [{"source": passage["source"], "rerank_score": passage.get("rerank_score")}
                           for passage in candidates[:PASSAGE_LIMIT]]
    return ({"model": RERANKER_MODEL, "revision": RERANKER_REVISION, "status": status, "candidate_count": len(candidates)},
            [{**passage, "score": round(passage["score"], 5)} for passage in candidates[:PASSAGE_LIMIT]])


def search_document(store, document_id, query, *, topic=None, query_embedding=embed_query):
    result = search_candidates(store, document_id, query, topic=topic, query_embedding=query_embedding)
    if result["status"] == "ready":
        result["reranking"], result["passages"] = rerank_passages(query, result["passages"])
    return result


def read_document_page(store, document_id, page):
    with store.session() as session:
        doc = session.get(PolicyDocument, document_id)
        if doc is None or doc.status != "ready":
            return {"status": "unavailable", "passages": []}
        chunks = session.scalars(select(PolicyChunk).where(PolicyChunk.document_id == doc.id, PolicyChunk.page == page)).all()
        if chunk_format := doc.embedding.get("chunk_format"):
            chunks = [chunk for chunk in chunks if chunk.id.startswith(f"{doc.id}:{chunk_format}:")]
        chunks.sort(key=lambda chunk: int(chunk.id.rsplit(":", 1)[1]))
        return {"status": "ready" if chunks else "unavailable", "document_id": doc.id, "file_hash": doc.file_hash,
                "page": page, "truncated": len(chunks) > 12,
                "passages": [{"source": "chunk:" + chunk.id, "page": page, "text": chunk.text,
                              **({"extraction": "ocr"} if page in doc.embedding.get("ocr_pages", []) else {})} for chunk in chunks[:12]]}


def uploaded_document_ids(store):
    """Snapshot uploaded wording; never mix the old demonstration fixture into it."""
    with store.session() as session:
        docs = session.scalars(select(PolicyDocument).where(
            PolicyDocument.id.not_in(select(PolicyDocumentRemoval.document_id))
        ).order_by(PolicyDocument.created_at)).all()
    return [doc.id for doc in docs if doc.product != GENERATED_WORDING_PRODUCT
            and not (doc.filename == "synthetic-motor.txt" and doc.product == "SYN-MOTOR")]


def search_uploaded_documents(store, document_ids, query):
    passages, unavailable = [], []
    documents = []
    query_embedding = cache(embed_query)
    for identifier in document_ids:
        result = search_candidates(store, identifier, query, query_embedding=query_embedding)
        if result["status"] != "ready":
            unavailable.append(identifier)
            continue
        metadata = {key: result[key] for key in ("document_id", "title", "version", "file_hash", "document_kind")}
        documents.append(metadata)
        passages.extend({**passage, **metadata} for passage in result["passages"])
    passages.sort(key=lambda passage: passage["score"], reverse=True)
    reranking, passages = rerank_passages(query, passages[:CANDIDATE_LIMIT])
    return {"status": "ready" if passages else "unavailable", "documents": documents,
            "passages": passages, "reranking": reranking, "unavailable_document_ids": unavailable,
            "note": "Search similarity is not proof that wording applies. Check the document's subject, scope and exclusions against the claim."}


def index_pending(store):
    with store.session() as session:
        identifiers = session.scalars(select(PolicyDocument.id).where(PolicyDocument.status.in_(["queued", "indexing"]))).all()
    for identifier in identifiers:
        index_document(store, identifier)
