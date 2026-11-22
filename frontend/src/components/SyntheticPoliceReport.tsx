import { FileText } from "lucide-react";
import type { PoliceReportDocument } from "../types";

export default function SyntheticPoliceReport({ report }: { report: PoliceReportDocument }) {
  return (
    <div className="document-preview">
      <div className="document-title">
        <FileText size={18} />
        <h3>{report.name}</h3>
        <span className="badge neutral">Synthetic police report</span>
      </div>
      <dl>
        <dt>Report reference</dt><dd>{report.id}</dd>
        <dt>Police station</dt><dd>{report.station}</dd>
        <dt>Report date</dt><dd>{report.date}</dd>
        <dt>Vehicle registration</dt><dd>{report.registration}</dd>
        <dt>Incident details</dt><dd>{report.summary}</dd>
      </dl>
    </div>
  );
}
