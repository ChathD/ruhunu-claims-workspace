import { FlaskConical, LoaderCircle, Plus } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "../api";
import { insuranceTypeLabel, lossTypeLabel, money } from "../components";
import type { ChargeReceiptDocument, PersonPolicyRecord, PoliceReportDocument, SupportingIncidentDocument } from "../types";
import AdditionalCosts from "../components/AdditionalCosts";
import ChargeReceipt from "../components/ChargeReceipt";
import { VehicleFacts } from "../components/PolicyTerms";
import SyntheticPoliceReport from "../components/SyntheticPoliceReport";
import SyntheticIncidentEvidence from "../components/SyntheticIncidentEvidence";

type ClaimPlan = {
  incident: string;
  gross_amount: string;
  policy_basis: string;
  required_evidence: string[];
  documents?: { name: string; requirement: "required" | "optional"; purpose: string; included: boolean }[];
  omission: string | null;
  expected_route: string;
  expectation: string;
};

export type SyntheticPreview = {
  batch_id: string;
  count: number;
  fixtures: {
    person: PersonPolicyRecord["person"];
    claim: Record<string, any> & { policy_id: string };
    scenario: string;
    conditions: string;
    scenario_type: string;
    plan: ClaimPlan;
  }[];
};
type ClaimPolicy = PersonPolicyRecord & { remaining_claims: number };

export default function GeneratorPage({
  generating,
  error,
  generate,
  preview,
  addToClaims,
  adding,
  discardPreview,
  openPeoplePolicies,
}: {
  generating: boolean;
  error: string;
  generate: (policyId: string, count: number) => Promise<void>;
  preview: SyntheticPreview | null;
  addToClaims: () => Promise<void>;
  adding: boolean;
  discardPreview: () => void;
  openPeoplePolicies: () => void;
}) {
  const [policies, setPolicies] = useState<ClaimPolicy[]>([]);
  const [policyId, setPolicyId] = useState(preview?.fixtures[0]?.claim.policy_id ?? "");
  const [count, setCount] = useState(preview?.count ?? 1);
  const [catalogError, setCatalogError] = useState("");
  useEffect(() => {
    let live = true;
    void api<{records: ClaimPolicy[]}>("/generator/claim-policies").then(({records}) => {
      if (live) {
        setPolicies(records);
        setCatalogError("");
      }
    }).catch((failure: Error) => {
      if (live) setCatalogError(failure.message);
    });
    return () => { live = false; };
  }, [preview]);
  const remaining = policies.find(record => record.policy.policy_id === policyId)?.remaining_claims ?? 0;
  const locked = generating || adding || !!preview;
  return (
    <div className="generator-layout">
      <section className="card generator-panel">
        <div className="feature-icon">
          <FlaskConical size={25} />
        </div>
        <h2>Generate claims for a policy</h2>
        <p>
          Select a policy. The generator can prepare claims that fit the policy, omit required evidence, or test a policy exception.
        </p>
        <button className="text-button" onClick={openPeoplePolicies}>Open people &amp; policies</button>
        <label className="form-field section-gap">
          Policy
          <select value={policyId} disabled={locked} onChange={event => {
            setPolicyId(event.target.value);
            setCount(1);
          }}>
            <option value="">Select a policy</option>
            {policies.map(({person, policy, remaining_claims}) => (
              <option key={policy.policy_id} value={policy.policy_id} disabled={!remaining_claims}>
                {person.full_name} · {policy.policy_id}{policy.insurance_type && ` · ${insuranceTypeLabel(policy.insurance_type)}`} · {remaining_claims} of 10 available
              </option>
            ))}
          </select>
        </label>
        <label className="form-field">
          Number of claims
          <input type="number" min={1} max={remaining || 1} value={Number.isNaN(count) ? "" : count}
            disabled={locked || !remaining} onChange={event => setCount(event.target.valueAsNumber)} />
        </label>
        <button
          className="button primary"
          disabled={locked || !Number.isInteger(count) || count < 1 || count > remaining}
          onClick={() => void generate(policyId, count)}
        >
          {generating ? (
            <LoaderCircle size={16} className="spin" />
          ) : (
            <Plus size={16} />
          )}
          {generating ? "Generating…" : "Generate preview"}
        </button>
        <small>{policyId ? `${remaining} of 10 claims available for this policy.` : "Choose a saved synthetic policy to see its remaining capacity."}</small>
        {catalogError && <div className="message error" role="alert">{catalogError}</div>}
        {error && <div className="message error" role="alert">{error}</div>}
      </section>
      {preview && <section className="card preview-panel">
        <div className="card-heading"><div><h2>Review synthetic data</h2><p>{preview.count} claims in this preview. These have not been added to Claims.</p></div></div>
        <div className="preview-claims">{preview.fixtures.map(({ claim, person, scenario, conditions, scenario_type, plan }, index) => {
          const costDocuments = claim.documents.filter((document: { type: string }) => ["repair_estimate", "vehicle_valuation", "third_party_demand"].includes(document.type));
          const receipts: ChargeReceiptDocument[] = claim.documents.filter((document: { type: string }) => document.type === "additional_cost_receipt");
          const policeReports: PoliceReportDocument[] = claim.documents.filter((document: { type: string }) => document.type === "police_report");
          const supportingEvidence: SupportingIncidentDocument[] = claim.documents.filter((document: { type: string }) =>
            document.type === "incident_report" || document.type === "witness_statement");
          return <details className="preview-claim" key={claim.claim_id} open={index === 0}>
          <summary><span>Claim {index + 1} · {lossTypeLabel(claim.loss_type)}<small>{claim.registration} · {claim.loss_date || "Loss date unavailable"}</small></span><strong>LKR {money(claim.gross_amount)}</strong></summary>
          <div className="preview-claim-body">
            <section><h3>Claim details</h3><dl className="preview-facts"><div><dt>Policyholder</dt><dd>{person.full_name} · {claim.person_id}</dd></div><div><dt>Policy</dt><dd>{claim.policy_id}</dd></div><div><dt>Reported</dt><dd>{claim.reported_date}</dd></div><div><dt>Incident</dt><dd>{claim.incident_id}</dd></div><div><dt>Description</dt><dd>{claim.narrative}</dd></div></dl></section>
            <section className="preview-wide">
              <h3>Cost evidence</h3>
              {costDocuments.length ? costDocuments.map((doc: Record<string, any>) => (
                <div className="preview-cost-document" key={doc.id}>
                  <p><strong>{doc.name}</strong> · {doc.id} · {doc.date}</p>
                  {doc.type === "repair_estimate" && <p>{doc.garage_name} · {doc.garage_address}</p>}
                  {doc.line_items.length ? (
                    <table>
                      <thead><tr><th>Item</th><th>Amount</th></tr></thead>
                      <tbody>{doc.line_items.map((item: Record<string, any>, i: number) => (
                        <tr key={i}><td>{item.description}</td><td>LKR {money(item.amount)}</td></tr>
                      ))}</tbody>
                      <tfoot><tr><th>Total</th><th>LKR {money(doc.total)}</th></tr></tfoot>
                    </table>
                  ) : <p>Itemised amounts are missing.</p>}
                </div>
              )) : <p>No cost document attached.</p>}
            </section>
            <section className="preview-wide">
              <h3>Police report</h3>
              {policeReports.length ? policeReports.map(report => <SyntheticPoliceReport key={report.id} report={report} />) : <p>No report attached.</p>}
            </section>
            <AdditionalCosts costs={claim.additional_costs} />
            {!!receipts.length && <section className="preview-wide">{receipts.map(receipt => <ChargeReceipt key={receipt.id} receipt={receipt} />)}</section>}
            {!!supportingEvidence.length && <section className="preview-wide">{supportingEvidence.map(document =>
              <SyntheticIncidentEvidence key={document.id} document={document} />)}</section>}
            {claim.vehicle && <section><h3>Reported vehicle</h3><VehicleFacts vehicle={claim.vehicle} /></section>}
            <section className="preview-wide"><h3>Generated scenario</h3><strong>{scenario}</strong><p>{scenario_type.replaceAll("_", " ")}</p><p>{conditions}</p></section>
            <section className="preview-wide">
              <h3>Claim plan</h3>
              <dl className="preview-facts">
                <div><dt>Planned incident</dt><dd>{plan.incident}</dd></div>
                <div><dt>Policy basis</dt><dd>{plan.policy_basis}</dd></div>
                <div><dt>Planned amount</dt><dd>LKR {money(plan.gross_amount)}</dd></div>
                <div><dt>Required evidence</dt><dd><ul>{plan.required_evidence.map((item, i) => <li key={i}>{item}</li>)}</ul></dd></div>
                {!!plan.documents?.length && <div><dt>Document plan</dt><dd><ul>{plan.documents.map(document => (
                  <li key={document.name}>
                    <strong>{document.name}</strong> · {document.requirement === "required" ? "Required" : "Optional"} · {document.included ? "Included" : "Omitted"}
                    <p>{document.purpose}</p>
                  </li>
                ))}</ul></dd></div>}
                {plan.omission && <div><dt>Deliberately omitted</dt><dd>{plan.omission}</dd></div>}
                <div><dt>Expected route</dt><dd><strong>{plan.expected_route}</strong><br />{plan.expectation}</dd></div>
              </dl>
            </section>
          </div>
        </details>;
        })}</div>
        <div className="preview-actions"><button className="button secondary" disabled={adding} onClick={discardPreview}>Discard preview</button><button className="button primary" disabled={adding} onClick={() => void addToClaims()}>{adding ? "Adding…" : `Add ${preview.count} to Claims and start AI Agent`}</button></div>
      </section>}
    </div>
  );
}
