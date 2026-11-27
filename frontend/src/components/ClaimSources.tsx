const labels: Record<string, string> = {
  fnol_transcript: "First notice of loss",
  police_report_ocr: "Police and incident evidence",
  repair_estimate_text: "Cost documents",
  independent_assessor_report: "Independent assessor report",
  medical_intake_note: "Medical intake",
  adjuster_notes: "Adjuster notes",
};
const absentSources = new Set(["", "No document submitted.", "No police report submitted."]);

export default function ClaimSources({ sources }: { sources: Record<string, string> }) {
  return <>{Object.entries(sources).filter(([, value]) => !absentSources.has(value.trim())).map(([key, value]) => (
    <details key={key}><summary>{labels[key] ?? key}</summary><pre>{value}</pre></details>
  ))}</>;
}
