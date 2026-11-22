# First-slice validation — 17 November 2026

## Manual submission investigation update

Manual submissions now queue investigation instead of publishing immediate review. Tests cover the shared Claude graph, native image-input construction, cached extraction, unreadable documents, immutable original submissions, staff adjudication after triage, internal-evidence privacy, and idempotent legacy investigation revisions.

A live isolated manual PDF submission completed all four specialists, uploaded-document extraction, policy RAG and judge publication with 16 successful model requests and no provider failures. It routed to adjuster review and its audit passed verification. The artifact is `.local/manual-validation/0551b14e-314f-4072-b46c-3186b54cf16c/audit.json`. `scripts/validate-manual.py` reproduces this check with synthetic evidence and incurs API usage. This does not establish independent document or policy authenticity.

## Claude supervisor and policy RAG update

- 43 backend tests passed, including native-tool dispatch, parallel round collection, second-round fraud follow-up, checkpoint resume at publication, immutable policy bindings, scoped RAG, PDF page citations, indexing retry, admin permissions, deadline/budget enforcement and routing overrides.
- React TypeScript checking and Vite production build passed.
- The actual local `all-MiniLM-L6-v2` model indexed demonstration wording and returned 384-dimensional embeddings. A separate check loaded its cached revision without network access.
- A real Claude Sonnet 4.6 claim investigation selected all four specialists, executed policy search and evidence checks, completed the follow-up supervisor and judge, and published adjuster review with exact source citations. Fifteen model requests completed with no provider failures. This validates integration, not recommendation accuracy or production calibration.
- Earlier live checks exposed overlong explanations and citation IDs with appended descriptions. Concise prompts and evidence-ID schema enums corrected those issues; invalid responses correctly required human review.
- The successful isolated smoke-test audit is under `.local/agentic-validation/0eb2cd66-932d-4a4f-8226-b65988ccc03c/audit.json` and passed integrity verification. `scripts/validate-agentic.py` reproduces the check using a fresh isolated database and incurs model usage.
- Windows sandbox socket restrictions prevented TestClient from creating its event loop. The full API suite passed outside that sandbox. Manual uploads remain staff-reviewed; scanned policy pages require OCR.

The sections below record the earlier deterministic implementation's validation, not Claude evaluation results.

## Automated verification

- 20 backend acceptance tests passed in 45.364 seconds.
- React TypeScript checking and Vite production build passed.
- Editable backend installation with `.[workspace]` succeeded.
- Python dependency consistency check passed.
- PowerShell startup script syntax check passed.

The tests execute real LangGraph workflows and LangChain tools without external model calls. They cover all 16 scenario types, the exact gross-amount boundary, fraud queue creation, coverage exits, document requests, duplicate handling, historical amount/fraud findings, scoped tools, publication guards, checkpoint recovery, audit integrity and API acknowledgement.

The first test run found a Windows-only open test connection during temporary-file cleanup. The connection was explicitly closed and the full suite passed on rerun.

## Live local demonstration

Two 16-claim batches were processed through the API/worker. One batch was generated using the React interface. Matrix rows are expected routes and columns are predictions:

| Expected / predicted | Auto-approve | Adjuster review | More information | Fraud escalation |
| --- | ---: | ---: | ---: | ---: |
| Auto-approve | 6 | 0 | 0 | 0 |
| Adjuster review | 0 | 16 | 0 | 0 |
| More information | 0 | 0 | 6 | 0 |
| Fraud escalation | 0 | 0 | 0 | 4 |

All 32 predictions matched the authored scenario labels. This is synthetic template validation, not independent evidence of production accuracy.

Browser checks confirmed the dashboard, generator, historical fraud claim workspace, structured document display and evaluation matrix. The claim workspace showed supported amount justification followed by independently verified historical invoice-reference reuse and fraud escalation. The downloaded audit package passed its integrity check.

Generated validation artifacts are in `.local/workspace/validation/` and are excluded from version control.

## Verification limits

No live model, OCR service or Langfuse endpoint was called. The runtime uses local SQLite and one worker. Authentication, real delivery receipts, signed immutable six-month retention, production-region enforcement, a preemptive processing deadline, sourced cost accounting and the 3,000-claim load test remain future delivery stages. See `ARCHITECTURE.md` for the staged roadmap.

## Intake-first v3 validation — 18 November 2026

The complete offline regression suite passed: **71 tests**, including 14 new Intake-first/readiness cases. The React TypeScript check and optimized Vite build passed. Tests use isolated databases, mocked Claude responses and fixture embeddings; no live provider evaluation was performed.

Observed v3 debug sequence: `r0:intake` (context 0 → 1), `r1:coverage` and `r1:fraud` (read v1; collector commits v2), `r2:historical` (reads v2; commits v3), then `r3:fraud` when historical leads require verification (reads v3; commits v4). The historical-reuse fixture publishes fraud escalation despite a scripted approval recommendation. A high-value fixture publishes adjuster review for gross LKR 250,000.01 even though payable is LKR 90,000.01. Audit exports verify successfully.

At this historical validation point a v2 debug run was also performed. That execution code has since been removed. See the cleanup validation below for the supported single graph.

See `INTAKE_FIRST.md` and `tests/test_intake_first.py` for dispatch contracts, stopping behavior and acceptance cases.

## Retired workflow removal — 18 November 2026

The sequential graph, v2 supervisor graph and additive model investigation loop
have been removed. Live and offline adapters now share the Intake-first graph.
The final backend suite passed **71 tests**; TypeScript checks and the optimized
Vite build also passed. These checks used mocked model responses and fixture
embeddings, without live Claude calls.

Regression coverage proves retired unfinished versions cannot execute or retry,
completed historical decisions remain readable, and a retired manual submission
with findings can receive an idempotent new revision without losing its original
findings or ownership. The existing routing matrix, gross-authority, fraud-task,
readiness, publication and checkpoint recovery cases all passed on the sole graph.

A covered-claim debug smoke test published auto-approve with a verified audit
export. Additional parallel debug invocations were stopped when checks slowed;
the final regression suite completed independently. Historical sections above
describe results at their original implementation stage, not additional supported
execution paths.

## Direct mandatory checks — 18 November 2026

Python now executes and audits each selected role's mandatory checks before its
first model request. No mandatory-check tool is advertised to Claude. The complete
backend suite passed **74 tests**, including check-before-model ordering, one
execution per role, first-request citations, failed-check/deadline prevention and
verified fraud preservation when the first model request fails.

The mocked covered-claim flow uses five specialist model requests: one each for
Intake, initial Fraud and Historical, plus two for Coverage (policy search then
analysis). The previous scripted tool-request flow used eight. This is a tested
fixture comparison, not a measured live-model cost or latency claim. Optional
evidence retrieval remains available.

The updated historical-fraud debug script completed with context v4, a fraud human
task and a verified audit export. The guard overrode its scripted auto-approve
recommendation to fraud escalation. No live model calls were made.

## Separate model-facing evidence tools — 18 November 2026

The complete backend suite passed **80 tests**, including six new model-tool and
evidence-audit cases. The focused six-test suite also passed with the updated
scripted provider. Model definitions and runtime dispatch expose only claim reads,
manual-upload reads where applicable, Coverage policy search and analysis submission.
Python retains its existing role-scoped deterministic retrieval tools.

Tests verify exact allowlists for every role and manual/synthetic claims; removed
and cross-role tools fail before retrieval. Check-time supporting responses match
their raw audit records, remain isolated between parallel specialists, preserve
successful verification details and missing-policy responses, and do not expand
downstream findings. Callback mutation cannot alter returned or persisted evidence.
Optional manual reads remain citable; later reads do not mutate the initial checks.

Three sequential isolated debug scenarios passed with valid audit exports:

| Scenario | Published route | Human queue |
| --- | --- | --- |
| Covered | auto-approve | None |
| Gross LKR 250,000.01; payable LKR 90,000.01 | adjuster review | adjuster |
| Verified historical invoice reuse | fraud escalation | fraud |

The last two deliberately override a scripted approval recommendation. No live
Claude calls or new cost/latency measurements were performed. Public endpoint
contracts and database schemas are unchanged; supporting evidence is an additive
field within the existing audited JSON response.

## Prior citations, manual invoice identity and verification snapshots — 18 November 2026

The complete backend suite passed **86 tests**, including six regression cases
for the issues found during the tool review. No live Claude calls were made.

Validated Coverage citations now work in Historical's initial model response
without another retrieval. References from hidden findings, other claims or
invented sources remain rejected; tampered prior audit responses stop before
model dispatch. The scripted judge deduplicates references and respects its
five-source prompt limit when findings inherit earlier validated references.

A manual invoice whose attachment ID differs from its printed invoice reference
now matches the two relevant settlements, produces a verified reuse indicator and
creates a fraud-queue task. A missing printed reference never falls back to the
attachment ID. Existing synthetic invoice-ID matching and old invoice_id-only
lead inputs remain supported.

Verification responses now include exact reread record snapshots, criteria,
distinct matching incident IDs and missing/out-of-scope record IDs for every lead.
Tests cover duplicate IDs/incidents, wrong invoice references, absent incident IDs,
missing records and isolation of other claims' contents. A changed settlement
cost is captured at verification time rather than reused from the earlier search.
Exported verification evidence remains unchanged after later database edits;
tampering with its exported snapshots fails audit integrity verification.

The new fields are additive JSON data; existing verified-lead consumers and
database schemas are preserved. Mandatory financial authority, fraud escalation,
checkpoint recovery and the authored routing confusion matrix all pass.

## Stale runtime and cascading operational errors — 18 November 2026

The reported synthetic claim had no Intake retrievals or Intake model invocation,
followed by a judge response whose citations were JSON field names rather than
available source identifiers. The running backend started at 11:00, before the
13:04 tool-interface changes, and returned an older health contract. The original
TypeError message was not retained, so its exact argument cannot be recovered.
An isolated old-signature regression reproduces the same pre-check failure when
the current caller passes `on_result` to the old `build_tools` interface.

The complete backend suite passed **91 tests**. New tests cover diagnostic error
messages and tracebacks, no judge call after Intake failure, preservation of the
primary error when judging also fails, empty-citation constraints and opt-in
development reload restricted to backend Python files. The reported claim's
exact payload passed Intake in an isolated temporary database with a scripted
model response, without modifying the original assessment or calling Claude.

The stale local backend was restarted with `--reload`; its health response now
exposes the current `execution_mode` contract. The development launcher enables
reload and stops its own process tree on exit. Fixed-code deployments omit the
flag and require a restart after deploying changes. Existing completed decisions
and audit history are preserved; they are not silently recomputed after restart.
