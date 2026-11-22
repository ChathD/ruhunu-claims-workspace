# Ruhunu Claims Workspace

**Single Intake-first workflow:** all new assessments run mandatory Intake,
then Claude assesses readiness separately for Coverage, Fraud and Historical.
Python enforces the coverage gate and validates dispatch. See
[the workflow walkthrough](docs/INTAKE_FIRST.md). Live Claude and offline adapters
use this same graph. Retired workflow code has been removed; completed audit
records remain available. Restart the backend to load source changes.

A standalone motor-claims triage project using **Python, FastAPI, LangChain tools, LangGraph, React, TypeScript and Tailwind**. The dashboard follows the agreed ivory-and-orange reference.

This first working slice processes synthetic structured claim forms and JSON repair estimates and other cost documents. It includes all four routing outcomes, evolving context, specialist-owned retrieval, human tasks, a confusion matrix and portable audit exports.

## Start locally

Requires Python 3.11+ and Node.js 22.12+ (validated with Python 3.11.9 and Node 24.11.1).

From the project root, install the backend:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[workspace]"
```

In terminal 1:

```powershell
.\.venv\Scripts\python.exe -m backend --reload
```

In terminal 2:

```powershell
cd frontend
npm.cmd ci
npm.cmd run dev -- --port 5180 --strictPort
```

Open **http://127.0.0.1:5180**. The backend listens on **http://127.0.0.1:8010** and API documentation is at **http://127.0.0.1:8010/docs**. Vite proxies `/api` to the backend.

Alternatively, after installing dependencies, run `powershell -ExecutionPolicy Bypass -File scripts/start-dev.ps1` from the root. It starts the backend in a hidden process, runs the frontend in your terminal and stops its own backend when you exit.

`--reload` is for local development: it restarts the server when backend Python
files change, avoiding mixed old/new modules after interface changes. It watches
the backend directory only. Omit the flag for a fixed-code deployment; restart
that server after deploying changes. Completed assessments retain their original
decisions and do not rerun automatically after a restart.

Start in **People & policies** and select **Generate and save**. Claude creates fictional people and one motor policy per person. In **Synthetic generator**, select a policy and preview up to its remaining capacity. Each policy allows ten saved synthetic claims, whether they are saved together or in successive batches. The generator checks each scenario type against the claim facts and rejects repeats across batches. Discarded previews leave capacity available. The background worker evaluates the submitted claims. Both generators need `ANTHROPIC_API_KEY` and use `ANTHROPIC_MODEL`.

For each scenario, Claude first plans a complete repair or theft claim against the selected policy, including the incident, covered costs and required evidence. A second request drafts the claim from that plan. For a deliberate omission, the backend validates the complete draft and then removes the selected item, such as a required police report. The preview shows the plan, the omission and the expected route. A complete claim is marked as an auto-approval candidate when its cost is supported by at least five comparable, staff-approved synthetic claims. Specialists determine the actual route after submission.

Expected routes and scenario conditions are stored in the evaluation database. The LLM labels assume an empty comparison history, so actual routing may differ as eligible history grows.

Newly generated policies store the vehicle make, model, year and fictional chassis number. Repair terms specify eligible incidents and work, and labour and painting limits. They explicitly list covered and excluded additional charges, including a decision for towing, storage, diagnostic inspections and temporary transport. Each eligible charge has its own conditions, sublimit and receipt rule. Theft cover specifies whether it pays only the vehicle valuation or also includes named fittings. The generator also records which covered loss types require a police report. An absent report blocks assessment for an active policy that requires it. A repair estimate includes the garage name and address, vehicle registration, date and itemised proposed work. Receipts for additional charges are kept separate from the estimate total.

Historical runs after coverage is confirmed and eligible claims exist. The `retrieve_comparable_claims` tool retrieves narratives and itemised cost documents from prior claims that staff approved before the current submission. Coverage checks that repair items fit the incident and assesses additional charges. The Historical specialist selects claims with similar damage and repairs and explains the comparison. Benchmarks use the selected itemised cost document totals. Claimants list each additional amount and its reason separately. With no eligible history, the workflow records the skip and routes otherwise approvable claims to an adjuster for amount validation.

When an earlier synthetic submission has a cost document for another vehicle, the generator can deliberately reuse that document in a new claim. Intake checks the exact copy against the frozen earlier submission, including the vehicle and incident identities. A verified match routes to Fraud escalation. The reviewer can assess whether the document was attached in error.

The backend loads the project-root `.env` when started with `python -m backend`. Set `ANTHROPIC_API_KEY` to enable the **Claude supervisor workflow** for new generated claims. `ANTHROPIC_MODEL` defaults to `claude-sonnet-4-6`; `CLAIMS_LLM_MODE=deterministic` selects offline checks on the same Intake-first graph. An assessment retains its stored execution mode; a missing API key never silently switches a model investigation to offline checks.

Intake runs and commits first. Claude then assesses Coverage, Fraud and Historical readiness, and selects ready specialists. Historical waits for confirmed coverage; new historical leads can trigger one Fraud follow-up. Each specialist chooses its permitted LangChain tools. Python validates check results, wording citations, financial limits and verified fraud indicators before publishing. There are at most four dispatch rounds after Intake, four requests per specialist invocation and 40 model requests per claim. Coverage has a 3,200-token output limit per request. Other model requests have a 1,800-token limit. A specialist can retry once for incomplete output or a schema validation error within its request count. Scanned-image OCR remains unconnected.

## Follow one claim

1. **Intake** retrieves the frozen claim and evidence, identifies gaps and checks duplicates. It records whether Coverage has enough information to check the policy period, insured vehicle, loss eligibility and claimed costs. Only a ready Intake can formulate questions for Coverage. In the Claude workflow, the supervisor can assign a Fraud screening task while Coverage waits for evidence.
2. **Supervisor** assesses readiness and dispatches ready Coverage, Fraud and Historical specialists against one committed context snapshot. Python validates required fields and dates, then gives Claude the available evidence. Claude interprets narratives and policy wording, reconciles records and evaluates comparables. Python validates precise arithmetic and enforces mandatory human routing.
3. **Collector** commits the round together, preserving evidence and context versions.
4. **Supervisor** reassesses after collection. Historical requires saved covered status; Fraud can follow up once on new historical leads. Coverage failure stops Historical.
5. **Judge** recommends one route with evidence references and an explanation.
6. **Publication guard** routes verified indicators and high-severity Fraud concerns based on conflicting, cited facts to the fraud team. It treats missing evidence as a completeness issue and requests it when executed checks identify a gap that blocks a required assessment. Each request names the missing item and the check it prevents. Gross LKR amounts above 250,000 create a human task even if the published route requests more information. Auto-approval is provisional synthetic triage.

Open a claim's **Investigation** tab to see the model's accepted specialist assignments that produced findings, grouped by round. A blocked claim with no recorded model assignment shows its screening status and points to the Intake findings. The audit export includes dispatch records, tool calls and retrieved passages. **Decision** shows the recommendation and any publication override.

Claude receives the exact policy, comparable and verification responses used by
its mandatory checks in `required_checks.supporting_evidence`. These remain in
the audit envelope rather than expanding downstream context. Its optional tools
are limited to claim details, manual-upload details and policy-wording search
for Coverage when wording is available. Python retains the full role-scoped check tools; Claude cannot repeat
duplicate searches, structured-policy retrieval, comparable queries or verification.

## Upload policy wording

Administrators can open **Policy documents** to upload English product wording as a text-based PDF or UTF-8 TXT, at most 10 MB. Every upload is an immutable version. Scanned/blank PDF pages fail with an OCR-required message. Indexing runs in the background; the page shows readiness, errors and retry controls.

Policy search uses local `sentence-transformers/all-MiniLM-L6-v2` CPU embeddings, downloaded into `.local/models` on first use. There is no embedding API key. Chunks use 180 model tokens with 30 overlap; normalized vectors persist in SQLite. Searches return up to five relevant passages across the assessment's uploaded document snapshot, with document, page and chunk citations.

Every newly generated synthetic policy includes a structured schedule and a separate wording document. The People & policies generator saves both together, links the wording to the policy, and queues it for indexing. A claim freezes that link at submission. Coverage uses the schedule for dates, vehicle identity, limits, deductibles and arithmetic. It searches the linked document for wording on eligibility, exclusions and evidence requirements. Findings show cited passages. Claude explains whether any difference could change this claim. Material or unresolved differences and unavailable applicable citations require adjuster review. Existing synthetic policies without a linked wording document continue to use their structured terms.

New manual assessments snapshot the uploaded document library. Claude searches that snapshot and assesses whether the cited wording applies. Generated wording stays out of that library. Completed assessments retain their original evidence references.

Run `.\.venv\Scripts\python.exe scripts/validate-agentic.py` for a generated-claim smoke test in an isolated `.local/agentic-validation` database. Use `scripts/validate-manual.py` to exercise uploaded wording. Both use the Anthropic API and export an audit JSON.

## Code map

The backend packages follow the [architecture diagram](docs/ruhunu-current-architecture.drawio).

```text
backend/
  api/                     FastAPI server, manual claims, staff review, SSE and policy uploads
  generation/              Synthetic claims, people, policies and wording
  workflow/                LangGraph, readiness, context, decision routing and the worker
  agents/                  Intake, Coverage, Fraud and Historical specialists
  evidence/                LangChain tools, document extraction, policy RAG and claim history
  storage/                 SQLAlchemy records and SQLite store
  assurance/               Audit export, investigation display, evaluation and Langfuse
  claim_schemas.py               Shared request, finding and decision types
  model_settings.py                Shared execution settings
frontend/src/
  pages/                  Overview, generator, evaluation and claim workspace
  components.tsx          Reusable cards, tables and controls
  App.tsx                 Navigation and remaining workspace screens
  api.ts                  HTTP client
  types.ts                Frontend contracts
  styles.css              Reference-inspired responsive design
```

Use the [backend component map](docs/ARCHITECTURE.md#backend-component-map) to find the exact files for each part of the workflow.

## Verify

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
cd frontend
npm.cmd run build
```

Tests exercise actual graph execution, boundary amounts, human-task creation, coverage exits, historical fraud follow-up, tool access, checkpoint recovery, publication guards, API acknowledgement and audit integrity. The matrix tests include an intentional off-diagonal mismatch.

## Local data and observability

Application data lives in `.local/workspace/` by default:

- `claims.sqlite3`: evidence, tool calls, findings, contexts, decisions and human actions.
- `checkpoints.sqlite3`: LangGraph checkpoints.
- `labels.sqlite3`: generator/evaluator labels, never exposed to specialist tools.

Copy `.env.example` to `.env` and set your key locally. Never commit `.env`. `python -m backend` loads it without overriding shell variables. Direct ASGI launches use their process environment.

### Investigation traces

The audit export includes `trace.id`, the local span tree, and submission and workflow events. Each assessment has a stable trace ID across worker restarts. An investigation reprocess gets a new trace ID and shares its claim session ID with the original. Context collection shows the before and after versions and changed keys. Each model turn links to its exact system instructions, messages, tools, parameters and response by `model_invocation_id`. Prompt hashes identify instruction versions without requiring a hosted prompt registry. Retrieval and tool spans include arguments, results, timing and failures.

Set `CLAIMS_ENABLE_LANGFUSE=1`, `LANGFUSE_BASE_URL`, `LANGFUSE_PUBLIC_KEY`, and `LANGFUSE_SECRET_KEY` for your Langfuse project, then restart the backend. Submit a claim and look up its audit `trace.id` in Langfuse. Metadata export includes the span hierarchy, context fingerprints, tool names, errors and token use. Keep `CLAIMS_LANGFUSE_CAPTURE_CONTENT=0` for real data. Set it to `1` only when your workspace, linked wording and claim history are wholly fictional. Content export applies to generated claims in synthetic-only workspaces.

Set `CLAIMS_STREAM_MODEL=1` to stream Anthropic investigation responses. A generation remains open until the response completes or fails. The local audit retains text and usage received before a streaming failure. Buffered mode records the complete response and its usage.

Audit exports include the exact policy wording and retrieved passages. Cost per claim displays **Not measured**. Queue acknowledgement uses a locally entered reviewer name.

## Delivery boundaries

This is a local, single-worker development slice, not the finished production deployment. Remaining work includes PDF/photo OCR and evidence coordinates, regional masking, PostgreSQL migrations and leases, authenticated human delivery, enforced deadlines, usage-based costing, 3,000-claim surge validation and immutable six-month retention.

Mandatory documents, fraud rules, auto-approval eligibility and contracted processing region still require client confirmation. Current scenario rules are explicitly provisional.

## Accounts, manual submission and staff decisions

The workspace now requires sign-in. Local development accounts are:

| Role | Email | Password |
| --- | --- | --- |
| User | user@ruhunu.local | UserDemo!2026 |
| Staff | staff@ruhunu.local | StaffDemo!2026 |
| Admin | admin@ruhunu.local | AdminDemo!2026 |

These deliberately seeded demonstration credentials are for loopback development only. Do not expose the service publicly. Passwords are scrypt-hashed; browser sessions use expiring HttpOnly, SameSite cookies. Production still requires HTTPS, rate limiting, password recovery and managed account provisioning.

Users can register their own account, submit claim details and up to five PDF/PNG/JPEG attachments (2 MB each), and track only their own claims. Additional costs have a separate amount and reason on the form. New manual submissions queue the same Claude supervisor/specialist/judge workflow as generated claims. PDF text is extracted locally. Claude extracts document facts from text and reads images or scanned PDFs using its native document and vision inputs. PDFs are limited to 20 pages each. Extracted document facts remain separate from the frozen submission. Every manual claim still needs a final authenticated staff decision.

Staff can inspect claims, download evidence, and record approval or rejection with a mandatory meaningful reason. Staff cannot review their own submissions. A final decision is append-only and cannot be overwritten through the review endpoint. This records adjudication only; no payment occurs. AI triage remains independently preserved.

Admins retain the original dashboard and synthetic generator, can use Claims & reviews, and can create accounts with user, staff or admin access. Ordinary registration cannot grant staff/admin access. The user's status page shows pending/processing, approved or rejected and the staff reason without exposing internal fraud findings or other users' records.

New tables are added automatically to the local SQLite database; existing claim records are preserved. Production migration tooling remains a later deployment task.

Staff and admins can use **Run investigation** on retired manual submissions without a final staff decision, including those with old findings. This creates an idempotent new investigation revision, copies ownership and attachments, preserves the original assessment and prevents adjudication of the superseded assessment. Claims & reviews separates model-assigned tasks from the audit evidence. Users see their claim status and requests for information. Run `python scripts/validate-manual.py` for an isolated synthetic live manual/PDF smoke test (incurs API usage).
