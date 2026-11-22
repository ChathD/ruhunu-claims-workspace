import {
  ArrowLeft,
  Check,
  FileText,
  LoaderCircle,
  RefreshCw,
  ShieldCheck,
} from "lucide-react";
import { AuditLink, Badge, lossTypeLabel, money, stageNames } from "../components";
import { routeLabels } from "../types";
import type { Claim, Detail, Page } from "../types";
import { useState } from "react";
import InvestigationTasks from "../components/InvestigationTasks";
import AdditionalCosts from "../components/AdditionalCosts";
import ChargeReceipt from "../components/ChargeReceipt";
import { VehicleFacts } from "../components/PolicyTerms";
import SpecialistFinding from "../components/SpecialistFinding";
import SyntheticPoliceReport from "../components/SyntheticPoliceReport";
import SyntheticIncidentEvidence from "../components/SyntheticIncidentEvidence";

const costDocumentTitles = new Map([
  ["repair_estimate", "Repair estimate"],
  ["vehicle_valuation", "Vehicle valuation"],
  ["third_party_demand", "Third-party demand"],
]);

export default function ClaimWorkspace({
  detail,
  page,
  tab,
  setTab,
  onBack,
  onAcknowledge,
  onRetry,
  onInvestigate,
}: {
  detail: Detail | null;
  page: Page;
  tab: string;
  setTab: (tab: string) => void;
  onBack: () => void;
  onAcknowledge: (claim: Claim) => void;
  onRetry: () => Promise<void>;
  onInvestigate: () => Promise<void>;
}) {
  const [retrying, setRetrying] = useState(false);
  return (
    <>
      <button className="back-link" onClick={onBack}>
        <ArrowLeft size={15} />
        Back to {page.toLowerCase()}
      </button>
      {!detail ? (
        <div className="card loading">
          <LoaderCircle className="spin" />
          Loading claim evidence…
        </div>
      ) : (
        <>
          <div className="page-heading detail-heading">
            <div>
              <div className="eyebrow">CLAIM WORKSPACE / {detail.submission.source === "Manual submission" ? "MANUAL" : "SYNTHETIC"}</div>
              <h1>
                {detail.reference}
                <Badge route={detail.route} status={detail.status} />
              </h1>
              <p>
                {lossTypeLabel(detail.type)} ·{" "}
                {detail.registration} · Received{" "}
                {new Date(detail.received_at).toLocaleString()}
              </p>
            </div>
            {(detail.decision || detail.status === "failed") && (
              <div className="heading-actions">
                {detail.submission.source === "Manual submission" && !Object.keys(detail.context.findings).length && (
                  <button className="button primary" onClick={() => void onInvestigate()}>Run investigation</button>
                )}
                <AuditLink id={detail.id} />
              </div>
            )}
          </div>
          <div className="claim-summary card">
            <div>
              <span>Claimed amount</span>
              <strong>LKR {money(detail.gross_amount)}</strong>
            </div>
            <div>
              <span>Supported payable</span>
              <strong>
                {detail.decision?.payable_amount == null
                  ? "Undetermined"
                  : `LKR ${money(detail.decision.payable_amount)}`}
              </strong>
            </div>
            <div>
              <span>Loss date</span>
              <strong>{detail.submission.loss_date || "Unreadable"}</strong>
            </div>
            <div>
              <span>Context version</span>
              <strong>v{detail.version}</strong>
            </div>
          </div>
          <div className="claim-columns">
            <section className="card claim-content">
              <div className="tabs">
                {["Findings", "Documents", "Investigation", "Context", "Decision"].map((t) => (
                  <button
                    className={tab === t ? "active" : ""}
                    onClick={() => setTab(t)}
                    key={t}
                  >
                    {t}
                  </button>
                ))}
              </div>
              {tab === "Findings" && (
                <div className="findings-list">
                  {detail.runs.map((run) => (
                    <SpecialistFinding key={run.id} finding={run.finding} />
                  ))}
                  {detail.runs.length === 0 && (
                    <div className="empty-state">
                      {detail.status === "awaiting_review" ? "Review this synthetic claim, then start the AI Agent from the Synthetic generator page." : "The AI Agent has not started this claim yet."}
                    </div>
                  )}
                </div>
              )}
              {tab === "Documents" && (
                <div className="document-list">
                  <div className="document-preview">
                    <div className="document-title">
                      <FileText size={18} />
                      <h3>{detail.submission.source === "Manual submission" ? "Submitted claim form" : "Synthetic claim form"}</h3>
                      <span className="badge neutral">Structured JSON</span>
                    </div>
                    <dl>
                      {detail.submission.person_id && <><dt>Person ID</dt><dd>{detail.submission.person_id}</dd></>}
                      <dt>Policy ID</dt>
                      <dd>{detail.submission.policy_id}</dd>
                      <dt>Incident description</dt>
                      <dd>{detail.submission.narrative}</dd>
                      <dt>Vehicle registration</dt>
                      <dd>{detail.registration}</dd>
                      <dt>Loss date</dt>
                      <dd
                        className={
                          !detail.submission.loss_date ? "red-text" : ""
                        }
                      >
                        {detail.submission.loss_date ||
                          "Unreadable — a clear copy is required"}
                      </dd>
                      <dt>Declared amount</dt>
                      <dd>LKR {money(detail.gross_amount)}</dd>
                    </dl>
                    <VehicleFacts vehicle={detail.submission.vehicle} />
                  </div>
                  <AdditionalCosts costs={detail.submission.additional_costs} />
                  {(detail.submission.attachment_manifest || []).map((file: { id: string; name: string }) => <div className="document-preview" key={file.id}><a className="text-button" href={`/api/portal/attachments/${file.id}`}>{file.name} · Download original</a></div>)}
                  {(detail.context.attachment_extraction || []).map((file: any) => <div className="document-preview" key={file.attachment_id}><h3>{file.name}</h3><p>{file.summary || file.reason}</p><span className="badge">{file.status} · Unverified claimant evidence</span><details><summary>Extracted document facts</summary><pre>{JSON.stringify(file, null, 2)}</pre></details></div>)}
                  {detail.submission.documents.map((doc: any) => doc.type === "police_report" ? (
                    <SyntheticPoliceReport key={doc.id} report={doc} />
                  ) : doc.type === "additional_cost_receipt" ? (
                    <ChargeReceipt key={doc.id} receipt={doc} />
                  ) : doc.type === "incident_report" || doc.type === "witness_statement" ? (
                    <SyntheticIncidentEvidence key={doc.id} document={doc} />
                  ) : costDocumentTitles.has(doc.type) ? (
                    <div className="document-preview" key={doc.id}>
                      <div className="document-title">
                        <FileText size={18} />
                        <h3>{doc.name}</h3>
                      </div>
                      <div className="cost-document-header">
                        <span className="eyebrow">
                          {costDocumentTitles.get(doc.type)}
                        </span>
                        <h3>{doc.id}</h3>
                        {doc.document_reference && <p>Original reference {doc.document_reference}</p>}
                        {doc.type === "repair_estimate" && <p>{[doc.garage_name, doc.garage_address].filter(Boolean).join(" · ")}</p>}
                        <span>
                          {doc.registration || (doc.chassis_number ? `Chassis ${doc.chassis_number}` : "Vehicle identifier unavailable")} · {doc.date}
                          {doc.chassis_number && ` · Chassis ${doc.chassis_number}`}
                        </span>
                      </div>
                      {doc.line_items.length ? (
                        <table className="cost-document-table">
                          <thead>
                            <tr>
                              <th>Description</th>
                              <th>Amount (LKR)</th>
                            </tr>
                          </thead>
                          <tbody>
                            {doc.line_items.map((line: any, i: number) => (
                              <tr key={i}>
                                <td>{line.description}</td>
                                <td>{money(line.amount)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      ) : (
                        <div className="evidence-gap">
                          Itemised charges are missing.
                        </div>
                      )}
                      <div className="cost-document-total">
                        <span>{costDocumentTitles.get(doc.type)} total</span>
                        <strong>LKR {money(doc.total)}</strong>
                      </div>
                    </div>
                  ) : null)}
                  {!detail.submission.documents.some((doc: { type: string }) => costDocumentTitles.has(doc.type)) && !(detail.submission.attachment_manifest || []).length && (
                    <div className="empty-state">
                      No itemised cost evidence was attached to this claim.
                    </div>
                  )}
                </div>
              )}
              {tab === "Context" && (
                <div className="context-panel">
                  <h3>Committed assessment context · v{detail.version}</h3>
                  <p>
                    Each specialist reads earlier findings and submits a
                    validated update. Original evidence remains preserved.
                  </p>
                  <pre>{JSON.stringify(detail.context, null, 2)}</pre>
                </div>
              )}
              {tab === "Investigation" && <div className="context-panel">
                <InvestigationTasks rounds={detail.investigation_tasks} findings={detail.context.findings} />
              </div>}
              {tab === "Decision" && (
                <div className="decision-panel">
                  {detail.decision ? (
                    <>
                      <span className="eyebrow">
                        AUTHORITATIVE RULE-BASED JUSTIFICATION
                      </span>
                      <h2>{routeLabels[detail.decision.route]}</h2>
                      <div className="reason-code">
                        {detail.decision.reason_codes.join(" · ")}
                      </div>
                      <div className="justification">
                        {detail.decision.justification}
                      </div>
                      {detail.decision.recommendation && <div className="panel-note"><strong>AI Agent recommended: {routeLabels[detail.decision.recommendation.route]}</strong><p>{detail.decision.recommendation.explanation}</p>{detail.decision.override_reason && <p>Publication override: {detail.decision.override_reason}</p>}</div>}
                      <div className="panel-note">
                        Provisional synthetic rules. No settlement or payment is
                        authorized.
                      </div>
                    </>
                  ) : (
                    <p>There is no published decision yet.</p>
                  )}
                </div>
              )}
            </section>
            <aside className="claim-aside">
              <section className="card workflow-panel">
                <h2>Agent workflow</h2>
                <p>Evidence builds at every step.</p>
                <div className="workflow-steps">
                  {detail.runs.map(run => [run.stage, run.stage === "r0:intake" ? "Intake (mandatory)" : (stageNames[run.stage] || run.stage.replace(/^r(\d+):/, "Round $1 · "))]).map(([stage, name], i) => {
                    const run = detail.runs.find((r) => r.stage === stage);
                    const interrupted = Boolean(run?.finding.details.error_type);
                    const skipped = detail.context.skipped[stage];
                    return (
                      <div
                        className={`workflow-step ${run && !interrupted ? "done" : skipped ? "skipped" : ""}`}
                        key={stage}
                      >
                        <span className="step-dot">
                          {run && !interrupted ? <Check size={12} /> : i + 1}
                        </span>
                        <div>
                          <strong>{name}</strong>
                          <small>
                            {interrupted ? "Interrupted" : run
                              ? `Completed · context v${run.input_version + 1}`
                              : skipped
                                ? "Skipped"
                                : "Pending"}
                          </small>
                          {skipped && <p>{skipped}</p>}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </section>
              {detail.task && (
                <section className="card human-panel">
                  <ShieldCheck size={23} />
                  <h3>Human handoff</h3>
                  <p>
                    {detail.task.queue === "fraud"
                      ? "Fraud specialist"
                      : "Adjuster"}{" "}
                    queue · {detail.task.status}
                  </p>
                  {detail.task.status === "pending" ? (
                    <button
                      className="button primary"
                      onClick={() => onAcknowledge(detail)}
                    >
                      Acknowledge receipt
                    </button>
                  ) : (
                    <div className="green-text">
                      Acknowledged by {detail.task.reviewer}
                    </div>
                  )}
                </section>
              )}
              {detail.status === "failed" && (
                <section className="card human-panel">
                  <h3>Processing interrupted</h3>
                  <p>{detail.error || "The AI Agent could not complete this investigation."}</p>
                  {!!detail.processing_alerts?.length && <p>{detail.processing_alerts.join(" · ")}</p>}
                  {detail.can_reprocess && <button
                    className="button secondary"
                    disabled={retrying}
                    onClick={async () => {
                      setRetrying(true);
                      try { await onRetry(); } finally { setRetrying(false); }
                    }}
                  >
                    <RefreshCw size={14} />
                    {retrying ? "Starting…" : "Reprocess with AI Agent"}
                  </button>}
                </section>
              )}
            </aside>
          </div>
        </>
      )}
    </>
  );
}
