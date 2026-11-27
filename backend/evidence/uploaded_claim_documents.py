"""Extract facts from manual claim uploads and attach derived evidence to the investigation."""
import base64
import copy
import io
from datetime import date

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from backend.assurance.investigation_tracing import trace_span
from backend.storage.sqlite_store import Attachment, AttachmentExtraction, ToolCall, digest
from backend.claim_schemas import ChargeCategory, Money
from backend.evidence.required_claim_evidence import COST_DOCUMENT_KINDS


EXTRACTION_PROMPT = """## Task
Extract visible facts from one submitted document. Use null for missing or illegible fields and list the gaps. Cite page numbers in the summary where available.

## Classification and fields
- Use repair_estimate for a quotation for proposed work. Read the garage name, address and vehicle registration. Leave chassis_number null.
- Use vehicle_valuation for a valuation and third_party_demand for an itemised third-party loss. In a demand, identify the third party and their loss, and read the insured vehicle's registration if printed.
- Use additional_cost_receipt for a separate service receipt. Name the service and use a snake_case charge_category, such as towing or storage.
- For these cost documents, extract the printed vehicle registration, reference, date, total and up to 15 line items. If there are more, leave line_items empty and report the gap.
- Use police_report, incident_report or witness_statement for recognizable incident evidence. Summarize the incident date, cause, vehicle, circumstances and who reported them. Leave document_reference and total null and line_items empty.
- Use work_incapacity_evidence for a filed record that states the injury-related time off work. Use earnings_evidence for a filed record that substantiates earnings. Describe the recorded period or rate in the summary. A third-party demand or a diagnosis alone does not substantiate either fact.
- Use other when the type is unclear.

## Output
Call extract_document with the visible facts. Keep the summary under 100 words."""


class CostLine(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str = Field(max_length=300)
    amount: Money


class ExtractedDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str = Field(min_length=1, max_length=1500)
    document_type: str = Field(max_length=80)
    charge_category: ChargeCategory | None = None
    document_reference: str | None = Field(default=None, max_length=100)
    registration: str | None = Field(default=None, max_length=40)
    chassis_number: str | None = Field(default=None, max_length=40)
    garage_name: str | None = Field(default=None, max_length=150)
    garage_address: str | None = Field(default=None, max_length=300)
    document_date: date | None = None
    total: Money | None = None
    line_items: list[CostLine] = Field(default_factory=list, max_length=15)
    gaps: list[str] = Field(default_factory=list, max_length=15)


def prepare_attachments(store, assessment_id):
    from backend.agents.claude_investigation import model_with_timeout_retry
    from backend.model_settings import settings
    with store.session() as session:
        files = session.scalars(select(Attachment).where(Attachment.assessment_id == assessment_id)).all()
    results = []
    for file in files:
        with store.session() as session:
            saved = session.get(AttachmentExtraction, file.id)
        if saved:
            results.append(saved.payload)
            continue
        with trace_span("tool.extract_uploaded_document", kind="tool",
                        metadata={"media_type": file.media_type}, content={"attachment_id": file.id}) as span:
            result = {"attachment_id": file.id, "name": file.name, "source": "attachment:" + file.id,
                      "status": "unavailable", "provenance": "claimant upload; not independently verified"}
            model_started = False
            try:
                raw = base64.b64decode(file.content)
                pages = []
                if file.media_type == "application/pdf":
                    from pypdf import PdfReader
                    pdf = PdfReader(io.BytesIO(raw))
                    if len(pdf.pages) > 20:
                        raise ValueError("Upload PDFs with at most 20 pages")
                    pages = [{"page": i + 1, "text": (page.extract_text() or "")[:12000]} for i, page in enumerate(pdf.pages)]
                result["pages"] = pages
                if not settings()[0]:
                    result.update(status="text_only" if any(page["text"].strip() for page in pages) else "unavailable",
                                  reason="Claude extraction disabled; staff must verify document fields")
                else:
                    if pages and all(page["text"].strip() for page in pages):
                        content = [{"type": "text", "text": "Submitted PDF pages:\n" + str(pages)}]
                    else:
                        content = [{"type": "document" if file.media_type == "application/pdf" else "image",
                                    "source": {"type": "base64", "media_type": file.media_type, "data": file.content}}]
                    model_started = True
                    response = model_with_timeout_retry(store, assessment_id, "extract:" + file.id, EXTRACTION_PROMPT,
                        [{"role": "user", "content": content}], [{"name": "extract_document", "description": "Submit visible document facts",
                        "input_schema": ExtractedDocument.model_json_schema()}], {"type": "tool", "name": "extract_document"})
                    calls = [block for block in response["content"] if block["type"] == "tool_use"]
                    if len(calls) != 1 or calls[0]["name"] != "extract_document":
                        raise ValueError("Invalid document extraction")
                    result.update(status="extracted", **ExtractedDocument.model_validate(calls[0]["input"]).model_dump(mode="json"))
            except Exception as error:
                result.update(status="unavailable", error_type=type(error).__name__, reason="Document extraction could not complete; staff verification required")
                if model_started:
                    result.update(technical_failure=True, reason="Document extraction was interrupted; reprocessing required")
            with store.session.begin() as session:
                session.add(AttachmentExtraction(attachment_id=file.id, payload=result))
                session.add(ToolCall(assessment_id=assessment_id, stage="document_extraction", tool="extract_uploaded_document",
                            request={"attachment_id": file.id}, response=result, response_hash=digest(result)))
            span.output = result
            span.summary = {"status": result["status"]}
            span.error_type = result.get("error_type")
        results.append(result)
    return results


def effective_submission(store, assessment_id, original):
    """Derived cost evidence stays unverified and separate from the original submission."""
    with store.session() as session:
        files = session.scalars(select(Attachment).where(Attachment.assessment_id == assessment_id)).all()
        extracted = [session.get(AttachmentExtraction, file.id) for file in files]
    derived = copy.deepcopy(original)
    evidence = [row.payload for row in extracted if row]
    derived["documents"] = cost_documents_from_extractions(evidence)
    derived["documents"].extend(
        {"id": item["attachment_id"], "name": item["name"], "type": "additional_cost_receipt", "date": item["document_date"],
         "registration": item["registration"], "total": item["total"], "description": item.get("summary", ""),
         "category": item["charge_category"],
         "source": item["source"], "provenance": item["provenance"]}
        for item in evidence if item.get("status") == "extracted" and item.get("document_type") == "additional_cost_receipt"
        and all(item.get(field) not in (None, "") for field in ("document_date", "registration", "total", "charge_category"))
    )
    derived["uploaded_evidence"] = evidence
    return derived


def cost_documents_from_extractions(evidence):
    documents = []
    for item in evidence:
        document_type = item.get("document_type", "").strip().lower().replace("-", "_").replace(" ", "_")
        document_type = {"valuation": "vehicle_valuation", "estimate": "repair_estimate"}.get(document_type, document_type)
        if (item.get("status") == "extracted" and document_type in COST_DOCUMENT_KINDS
                and (document_type == "repair_estimate" or all(item.get(key) is not None for key in ("document_date", "total")))):
            documents.append({"id": item["attachment_id"], "document_reference": item.get("document_reference"), "name": item["name"],
                "type": document_type, "registration": item.get("registration"),
                "date": item.get("document_date"), "total": item.get("total"),
                "line_items": item.get("line_items", []), "provenance": item["provenance"], "source": item["source"]})
            if document_type == "repair_estimate":
                documents[-1].update(garage_name=item.get("garage_name"), garage_address=item.get("garage_address"))
            else:
                documents[-1]["chassis_number"] = item.get("chassis_number")
            if document_type == "third_party_demand":
                documents[-1]["summary"] = item.get("summary", "")
    return documents
