# Debugging a claim through the Intake-first graph

`backend/workflow/intake_first_graph.py` is the only graph. The sequential graph and
v2 supervisor graph have been removed. Live Claude and offline checks share
mandatory Intake, readiness, specialists, collection, judging and publication.

## Reproduce without external calls

```powershell
.\.venv\Scripts\python.exe scripts/debug_workflow.py --scenario historical-fraud
.\.venv\Scripts\python.exe scripts/debug_workflow.py --scenario high-value
.\.venv\Scripts\python.exe scripts/debug_workflow.py --scenario covered
```

Each command creates isolated databases under `.local/debug/`. The real graph,
tools, SQL queries, collection, routing and audit export execute. Scripted model
responses use the saved fixture policy. HTTP POSTs are blocked. The debugger uses
the model adapter with a scripted provider; normal deterministic assessments use
the offline adapter on the same graph. There is no workflow-selection flag.

Artifacts include `walkthrough.md`, `trace.json`, `audit.json` and application,
checkpoint and label databases. The walkthrough links exact function locations.
Profiling captures parallel threads; their chronological order can vary while
the committed context versions remain deterministic.

## Follow the code

1. `backend/api/portal_routes.py` or `backend/generation/synthetic_claim_generator.py` saves the submission, policy
   library snapshot, workflow version and execution mode. `backend/workflow/investigation_worker.py`
   loads queued assessments and calls `ClaimsWorkflow.run`.
2. `backend/workflow/claims_workflow.py` rejects unfinished retired versions. Completed records
   return without execution. Manual attachment extraction is prepared and cached.
3. `IntakeFirstWorkflow.run` starts/resumes `intake-first:<assessment_id>` with
   synchronous durability. `intake` runs `r0:intake` and commits facts, gaps and
   duplicates as context v1 before readiness runs.
4. `supervise` calls `backend/workflow/investigation_execution.py`. Live readiness uses
   `backend/workflow/specialist_readiness.py`. Python validates evidence references, gaps,
   completion claims and exact task/state consistency. Historical requires a
   committed COVERED finding.
5. `specialist` executes selected roles against one frozen round snapshot.
   `backend/agents/claude_investigation.py` runs and audits mandatory Python checks before
   calling Claude, then supplies them as `required_checks` in the initial input.
   Its `supporting_evidence` contains exact policy, comparable or verification
   responses captured after their retrieval audit commits. The enclosing reference
   cites this evidence. Python keeps its check tools; Claude can only request claim
   details, manual-upload details, and Coverage policy-wording searches.
   Claude may finish immediately or request additional evidence. Offline execution
   invokes the same checks from `backend/agents/specialist_preflight.py`. Specialists fetch
   records through `backend/evidence/specialist_evidence_tools.py`. `backend/workflow/specialist_context.py` filters
   the model context by role.
6. `collect` commits each AgentRun and before/after context in one transaction,
   advancing context once per round. Recovery reuses committed rounds.
   `after_collect` applies coverage, fraud, error and budget gates.
7. `_judge` persists the recommendation. `ClaimsWorkflow.publish` independently
   recomputes the route with `backend/workflow/routing_rules.py` and commits the decision
   and required human task together. Staff adjudication remains separate.

The covered fixture follows:

```text
r0:intake                  reads v0 -> commits v1
r1:coverage + r1:fraud      read  v1 -> commits v2
r2:historical              reads v2 -> commits v3
r3:fraud (new leads only)  reads v3 -> commits v4
judge -> publication guard -> exactly one route
```

Coverage failure prevents Historical retrieval. Fraud indicators require human
fraud review. Gross LKR 250,000.01 requires an adjuster even if payable is lower.

## Useful debugger stops

Run the script with `python -m pdb`; set function breakpoints after imports:

```text
b backend.workflow.intake_first_graph.IntakeFirstWorkflow.intake
b backend.workflow.intake_first_graph.IntakeFirstWorkflow.supervise
b backend.workflow.specialist_readiness.validate_selection
b backend.workflow.intake_first_graph.IntakeFirstWorkflow.collect
b backend.workflow.routing_rules.select_route
c
```

Inspect `state['snapshot']`, `state['tasks']`, `Assessment.context`,
`AgentRun.input_context`, `ToolCall.response` and the final decision. Readiness
history retains rejected selections and exact owned gaps. Audit exports contain
source snapshots, instructions, retrieved data, findings and recommendations.

`prior_evidence_sources` admits only audit-backed references visible in this
specialist's filtered context. For Fraud follow-up, inspect the
`verify_historical_leads` response's `criteria` and `checks` for the actual record
snapshots, rejected IDs and distinct incidents used to confirm or reject each lead.
Manual cost-document leads retain the printed `document_reference` and the file's
`document_id`. An unreadable printed reference cannot establish reuse.

## Operational failures and stale processes

Run `python -m backend --reload` during development; the development launcher also
enables backend-only reload. A server started before a Python interface change can
combine a lazily imported new module with already loaded old modules. For example,
passing `on_result` to an older `build_tools` signature raises TypeError before
any tool audit or Intake model request. Restart the server before investigating
claim contents when runtime behavior disagrees with the source on disk.

Specialist failures now log a traceback and retain a bounded `error_message` beside
`error_type` in internal findings. Failed Intake proceeds directly to a deterministic
human-review recommendation, without spending a judge request. A later judge
failure is recorded separately in `judge_error`; it does not overwrite an earlier
`investigation_error`. With no available citations, the judge schema requires an
empty evidence list; JSON field names are not valid evidence identifiers.

For the development launcher, see `.local/logs/backend.err.log`. Completed failed
investigations remain historical records: a restart repairs future execution,
not the original decision or its audit. Use a new submission or supported new
investigation revision when fresh processing is needed.

```powershell
$env:CLAIMS_LLM_MODE='deterministic'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
