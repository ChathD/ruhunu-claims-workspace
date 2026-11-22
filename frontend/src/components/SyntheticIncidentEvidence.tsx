import type { SupportingIncidentDocument } from "../types";

export const incidentDocumentNames = { incident_report: "Incident report", witness_statement: "Witness statement" };

export default function SyntheticIncidentEvidence({ document }: { document: SupportingIncidentDocument }) {
  return <div className="document-preview">
    <div className="document-title">
      <h3>{document.name}</h3>
      <span className="badge neutral">{incidentDocumentNames[document.type]}</span>
    </div>
    <dl>
      <dt>Prepared by</dt><dd>{document.author}</dd>
      <dt>Date</dt><dd>{document.date}</dd>
      <dt>Vehicle registration</dt><dd>{document.registration}</dd>
      <dt>Incident details</dt><dd>{document.summary}</dd>
    </dl>
  </div>;
}
