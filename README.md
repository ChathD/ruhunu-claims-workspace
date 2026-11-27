# Ruhunu Claims Workspace

**Intake-first investigations:** v3 is the default for newly processed claims. The
selected-PDF v4 pilot adds typed planning and independent evidence verification
before specialist findings enter routing. Both versions retain the investigation
audit. See [the workflow walkthrough](docs/INTAKE_FIRST.md).
Restart the backend after configuration or source changes.

A standalone Sri Lankan motor-claims triage project using **Python, FastAPI, LangChain tools, LangGraph, React and TypeScript**.

The workspace processes synthetic claim text and claimant-uploaded documents. It includes four routing outcomes, specialist-owned retrieval, human tasks, a confusion matrix and portable audit exports.

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

Sign in as staff and open **Policy schedule generator**. Select an indexed motor policy PDF, then generate one fictional Sri Lankan policy schedule at a time. The generator reads complete PDF pages in order and verifies cited lines before saving. It reads text-based PDFs and applies OCR to scanned wording, including ruled tables. A product summary (IPID) produces a provisional schedule with undisclosed terms left unspecified. Claims matched to a provisional schedule require the full wording and individual schedule to confirm coverage. Open **Claim generator** and select **Generate claim**. After saving, open the claim and choose **Process claim**. The system matches the reported policyholder and vehicle to the schedule and linked wording. Schedule and claim generation require `ANTHROPIC_API_KEY` and use `ANTHROPIC_MODEL`.

The backend draws a route and a scenario supported by a saved policy schedule and its linked PDF version. It checks the benefit citation and any scenario-specific wording, such as a police report threshold or excluded vehicle use, before drafting. Claude receives those clauses alongside the scheduled terms and assigned facts. The backend checks the rendered evidence against the expected route. The evaluation labels record the decision reason, PDF version, policy citations and claim evidence. Intake extracts the submitted text and document fields for triage.

- **Auto-approval** covers routine minor damage, outpatient care under motor medical cover, theft of one scheduled fitted item, towing at the scheduled cap, and glass-only repair under an excess waiver.
- **Adjuster review** covers shared liability, sublimits or depreciation, the boundary between prior and new damage, a severe loss near the policy limit, and conditional driver authorisation.
- **Request more information** covers an unitemised invoice, an omitted mandatory police report, a missing vehicle identifier, a small discrepancy between the form and invoice totals, and missing proof of payment.
- **Fraud escalation** covers an incident shown outside policy dates, a substituted vehicle registration, inflated severity and duplicate charges, concealed commercial use, and a cost document reused from a separate prior claim.

On the claim page, open the linked motor policy schedule. When investigation results are available, open the report and choose **Download PDF** if you need a copy.

Ground truth is stored in the evaluation database. Use **Download evaluation files** after saving a claim or request `/api/generator/batches/{batch_id}/dataset.zip`. The archive contains linked JSON files for the schedule snapshots, submissions, attachments, and ground truth. Use `case_links.json` to join the observable inputs and `ground_truth.json` to score predictions. The JSON and CSV batch endpoints also remain available. The target route is an evaluation label. Live triage applies its own evidence and routing rules.

Historical runs when eligible claims exist. The `retrieve_comparable_claims` tool retrieves narratives, itemised costs, published triage outcomes and any staff decisions recorded before the current submission. The Historical specialist selects similar incidents and explains how their decisions inform the current assessment. The report shows the selected cases and their outcomes. The benchmark uses any usable itemised costs from those cases. Coverage checks repair scope and separately claimed charges against the current policy and evidence. For a synthetic repair under a matched schedule with fixed terms, a fully reconciled itemised estimate and independent physical assessment can support provisional triage when no earlier claims are available.

The backend loads the project-root `.env` when started with `python -m backend`. Set `ANTHROPIC_API_KEY` to enable the **Claude supervisor workflow** for new generated claims. `ANTHROPIC_MODEL` defaults to `claude-sonnet-4-6`; `CLAIMS_LLM_MODE=deterministic` selects offline checks on the same Intake-first graph. An assessment retains its stored execution mode; a missing API key never silently switches a model investigation to offline checks.

Intake runs and commits first. Claude then assesses Coverage, Fraud and Historical readiness, and selects ready specialists. Historical waits for confirmed coverage; new historical leads can trigger one Fraud follow-up. Each specialist chooses its permitted LangChain tools. Python validates check results, wording citations, financial limits and verified fraud indicators before publishing. There are at most four dispatch rounds after Intake, four standard turns per specialist invocation and 40 model requests per claim. Coverage starts with a 3,200-token output budget. A truncated Coverage response can retry once at 8,192 tokens. A specialist can make one additional request to correct schema errors, including on the final turn.

## Follow one claim

1. **Intake** retrieves the frozen claim and evidence, identifies gaps and checks duplicates. It records whether Coverage has enough information to check the policy period, insured vehicle, loss eligibility and claimed costs. Only a ready Intake can formulate questions for Coverage. In the Claude workflow, the supervisor can assign a Fraud screening task while Coverage waits for evidence.
2. **Supervisor** assesses readiness and dispatches ready Coverage, Fraud and Historical specialists against one committed context snapshot. Python validates required fields and dates, then gives Claude the available evidence. Claude interprets narratives and policy wording, reconciles records and evaluates comparables. Python validates precise arithmetic and enforces mandatory human routing.
3. **Collector** commits the round together, preserving evidence and context versions.
4. **Supervisor** reassesses after collection. Historical requires saved covered status; Fraud can follow up once on new historical leads. Coverage failure stops Historical.
5. **Judge** recommends one route with evidence references and an explanation.
6. **Publication guard** routes verified indicators and high-severity Fraud concerns based on conflicting, cited facts to the fraud team. It treats missing evidence as a completeness issue and requests it when executed checks identify a gap that blocks a required assessment. Each request names the missing item and the check it prevents. Gross LKR amounts above 250,000 create a human task even if the published route requests more information. Auto-approval is provisional synthetic triage.

For Claude investigations of third-party claims against a staff-selected PDF, Coverage records policy applicability and amount support for each claimed line, alongside a separate liability finding. Coverage commits claimant and internal evidence requests before the Judge runs. Publication uses the claimant requests independently of the Judge's recommendation. Find these records in `context.findings.coverage.details.third_party_assessment` in the audit JSON.

Open a claim to see its documents, triage route and findings. **Audit JSON** includes specialist assignments, tool calls, retrieved passages and the published decision.

Claude receives the exact policy, comparable and verification responses used by
its mandatory checks in `required_checks.supporting_evidence`. These remain in
the audit envelope rather than expanding downstream context. Its optional tools
are limited to claim details, manual-upload details and policy-wording search
for Coverage when wording is available. Python retains the full role-scoped check tools; Claude cannot repeat
duplicate searches, structured-policy retrieval, comparable queries or verification.

## Upload policy wording

Staff and administrators can open **Policy wording** to upload a policy PDF up to 12 MiB and view uploaded versions. Each upload creates a version. The background indexer reads searchable text and runs local OCR on scanned pages, using Windows English OCR when available. The index records which passages came from OCR.

Set `OPENAI_API_KEY` in `.env` to index and search policy wording. The backend splits each page into 200 GPT-2 tokens with 40 tokens of overlap and embeds the passages using OpenAI `text-embedding-3-small`. The database returns 10 candidates using cosine similarity and a keyword boost for topic searches. A local MiniLM cross-encoder reranks them against the original query and returns the best five with page and passage citations. Selected-PDF v4 initially uses the best two per topic.

For a PDF indexed with older passage boundaries, staff and administrators can open **Policy wording** and choose **Reindex with OpenAI**. Reindexing reads the original PDF and creates a new set of passage IDs. Earlier passages remain available for existing citations. The tokenizer and reranker assets are cached in `.local/models`. If the reranker cannot run, the tool records the fallback and returns the top five from SQLite's first-stage ranking.

For a new synthetic claim, the matched policy schedule and its linked PDF version are pinned when staff starts processing. Intake inventories the claim evidence and records cost gaps for Coverage to check after applicability. Coverage checks the schedule period and elected benefits, applies the linked wording and cites relevant pages. Indexed OCR and native PDF text use the same citation and applicability checks. Fraud checks contradictory assertions in the claim evidence and verified historical leads.

Queued legacy manual assessments finish on their saved v3 workflow and document snapshot. Staff can review completed legacy cases and download their original evidence.

Run `.\.venv\Scripts\python.exe scripts/validate-agentic.py <policy-pdf>` for a generated-claim smoke test in an isolated `.local/agentic-validation` database. This command uses the Anthropic and OpenAI APIs and exports an audit JSON.

## Code map

The backend packages follow the [architecture diagram](docs/ruhunu-current-architecture.drawio).

```text
backend/
  api/                     FastAPI server, staff review, SSE and policy uploads
  generation/              Synthetic claims and Claude generation client
  workflow/                LangGraph, readiness, context, decision routing and the worker
  agents/                  Intake, Coverage, Fraud and Historical specialists
  evidence/                LangChain tools, document extraction, policy RAG and claim history
  storage/                 SQLAlchemy records and SQLite store
  assurance/               Audit export, investigation display, evaluation and Langfuse
  claim_schemas.py               Shared request, finding and decision types
  model_settings.py                Shared execution settings
frontend/src/
  pages/                  Staff generator and policy wording management
  components.tsx          Reusable cards, tables and controls
  Portal.tsx              Sign-in, role navigation, claims and accounts
  api.ts                  HTTP client
  types.ts                Frontend contracts
  styles.css              Shared portal styles
```

Use the [backend component map](docs/ARCHITECTURE.md#backend-component-map) to find the exact files for each part of the workflow.

## Deploy to Vercel

Import this repository as a Services project with `vercel.json` at the repository root.
The `frontend` service serves the React app; `/api` routes to the FastAPI `backend`
service. Services and Queues are Vercel beta features. Keep Vercel Authentication
enabled for the deployment while using the demonstration accounts.

Configure `CLAIMS_STORAGE=supabase`, `SUPABASE_DATABASE_URL`, `ANTHROPIC_API_KEY`
and `OPENAI_API_KEY` in the project's server environment. Copy the existing
Langfuse settings if tracing should continue. Set `CLAIMS_ALLOWED_ORIGINS` to any
custom HTTPS domain; Vercel's deployment and production hostnames are allowed
automatically. Credentials stay outside Git and the browser bundle.

Hosted claim investigations and policy indexing use the `ruhunu-claims-jobs`
Vercel queue. The subscriber resumes persisted LangGraph checkpoints and takes a
Postgres advisory lock so local and hosted workers do not process simultaneously.
Restart a running local backend after updating to this version so it also takes
that lock. Scratch models use `/tmp`; claims, documents and checkpoints use the
existing private Supabase schema. `/api/health` reports the deployed Git SHA.

## Verify

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
cd frontend
npm.cmd run build
```

Tests exercise actual graph execution, boundary amounts, human-task creation, coverage exits, historical fraud follow-up, tool access, checkpoint recovery, publication guards, API acknowledgement and audit integrity. The matrix tests include an intentional off-diagonal mismatch.

## Local data and observability

Application data lives in `.local/workspace/` by default:

Supabase mode stores submissions, evidence, evaluation labels, policy vectors, and LangGraph checkpoints in the private `ruhunu` Postgres schema. The backend uses `SUPABASE_DATABASE_URL`, a server-side Session pooler connection string with `sslmode=require`. It continues to handle browser authentication through the existing workspace API.

To move an existing SQLite workspace, stop the backend and run `python scripts/migrate-to-supabase.py`. The importer reads the three SQLite databases under `.local/workspace`, verifies row counts, and copies the policy index and checkpoint history. Use `--source` for a different workspace directory. Set `CLAIMS_STORAGE=supabase` and restart once the import succeeds. The local development setting `CLAIMS_STORAGE=sqlite` uses the original SQLite files.

Copy `.env.example` to `.env` and set your key locally. Never commit `.env`. `python -m backend` loads it without overriding shell variables. Direct ASGI launches use their process environment.

### Investigation traces

The audit export includes `trace.id`, the local span tree, and submission and workflow events. Each assessment has a stable trace ID across worker restarts. An investigation reprocess gets a new trace ID and shares its claim session ID with the original. Context collection shows the before and after versions and changed keys. Each model turn links to its exact system instructions, messages, tools, parameters and response by `model_invocation_id`. Prompt hashes identify instruction versions without requiring a hosted prompt registry. Retrieval and tool spans include arguments, results, timing and failures.

1. Set `CLAIMS_ENABLE_LANGFUSE=1`, `LANGFUSE_BASE_URL`, `LANGFUSE_PUBLIC_KEY`, and `LANGFUSE_SECRET_KEY` in `.env`. Copy the URL and keys from your Langfuse project's API Keys settings. `LANGFUSE_TRACING_ENVIRONMENT` and `LANGFUSE_RELEASE` let you filter by deployment.
2. By default, Langfuse shows input and output summaries. For full text and structured payloads, set `CLAIMS_LANGFUSE_CAPTURE_PAYLOADS=1` in a workspace approved to send claim and policy data to Langfuse. This includes model prompts and responses, tool arguments and results, and retrieved passages. Binary attachments and embedding vectors are omitted. Restart the backend after changing `.env`.
3. Run a new investigation, claim preview, policy schedule generation, or policy index job. In your Langfuse project's **Tracing** view, filter by `claim-investigation`, `generator-preview`, `generator-policy_schedules`, or `policy-index`. To find a specific investigation, use `trace.id` from **Export audit** on the staff claim.
4. Open the trace and expand observations. Select an agent or tool to see its input, output, duration, and errors. Policy search shows query embedding, vector retrieval, and reranking when used. Select a Claude generation or OpenAI embedding to see the model and provider-reported tokens. Langfuse estimates USD cost from its model pricing when the model and usage match a pricing definition.

### Final investigation correctness

**Ruhunu Final Investigation Correctness** is the project's sole LLM judge. It uses GPT-6.2 Astra to read the **Input** and **Output** of the root `claims.investigation` observation. Its rule matches the investigation name, **Is Root Observation** and `metadata.final_evaluation:ready`. The backend adds the marker to a completed investigation when full payload capture is enabled and the recorded submission and Intake finding are available. The input includes the submitted claim and document evidence, the policy terms and retrievals, and the committed findings. The output contains the published triage decision. [Langfuse observation-level evaluations](https://langfuse.com/docs/evaluation/evaluation-methods/llm-as-a-judge#observation-evaluator-context)

Open a completed investigation's root in **Tracing** and use **Preview** to inspect its input and output. **Scores** shows the evaluator's score from 0 to 1 and its explanation when the rule has run. A score of 1 means the disposition and rationale are correct and grounded. Fractional scores reflect partial correctness. To review individual model calls, expand their observations in the trace. The draft route appears in the `claims.route_recommendation` span and the local audit. The evaluator receives the published outcome.

The app sends observations in batches and flushes at shutdown. To verify an isolated live investigation, run `python scripts/validate-agentic.py path/to/sample-policy.pdf` and look up the trace ID it prints in Langfuse's `verification` environment. In **Scores**, filter by **Ruhunu Final Investigation Correctness** and exclude `verification` to review live workflow verdicts.

Open **Investigation → Specialist inputs** on a staff claim to inspect the saved model requests and recorded tool calls by round. On new Langfuse traces, the **claims.context.model_request** input lists the sections included in the request, the prior findings it received, and the number of policy passages included at the start.

Set `CLAIMS_STREAM_MODEL=1` to stream Anthropic investigation responses. A generation remains open until the response completes or fails. The local audit retains text and usage received before a streaming failure. Buffered mode records the complete response and its usage.

Audit exports include the exact policy wording and retrieved passages. Cost per claim displays **Not measured**. Queue acknowledgement uses a locally entered reviewer name.

## Delivery boundaries

This is a local, single-worker development slice, not the finished production deployment. Remaining work includes PDF/photo OCR and evidence coordinates, regional masking, PostgreSQL migrations and leases, authenticated human delivery, enforced deadlines, usage-based costing, 3,000-claim surge validation and immutable six-month retention.

Mandatory documents, fraud rules, auto-approval eligibility and contracted processing region still require client confirmation. Current scenario rules are explicitly provisional.

## Staff and admin access

Local development accounts are:

| Role | Email | Password |
| --- | --- | --- |
| Staff | staff@ruhunu.local | StaffDemo!2026 |
| Admin | admin@ruhunu.local | AdminDemo!2026 |

Use these demonstration credentials on loopback only. Staff generate synthetic
claims, upload, view, reindex and select policy PDFs, and review investigation findings and decisions.
Administrators create staff accounts and can open **Routing evaluation** in the admin portal.
Existing manual claim records, attachments and audits remain available to staff.
Legacy customer accounts are inactive and remain linked to their historical records.

## Selected-PDF v4 pilot

Set `CLAIMS_SELECTED_PDF_WORKFLOW=v4` to opt new selected-PDF assessments into the
pilot. The default is `v3`. Each queued run and subsequent retry uses the
recorded workflow and selected PDF. The v4 staff detail shows the typed plan,
verification results, and page links. The independent auditor runs only when
the specialists propose evidence-dependent conclusions.

Run `scripts/evaluate-selected-pdf.py` with `--allianz`, `--fairfirst` and
`--ipid` pointing to the three sample PDFs. The comparison runs 24 fixed pairs
three times per version in isolated SQLite stores. Default runs exercise
deterministic claim checks. Both modes use OpenAI for policy embeddings. Use
`--export-labels` to export the draft answer key for review.
Use `--live --labels <reviewed-answer-key.json>` to compare live Claude runs
with the reviewed expectations. Record the reviewer in `reviewed_by`.
The script reports release eligibility, route and evidence accuracy, safety
checks, completion, model requests, tokens and elapsed time. The pilot may only
become the default after it passes the live quality gate.
