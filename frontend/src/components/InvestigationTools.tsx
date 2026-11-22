import { groupInvestigationEvidence } from "../investigationEvidence";
import type { InvestigationToolCall } from "../types";

export default function InvestigationTools({
  calls = [],
  jsonClassName,
}: {
  calls?: InvestigationToolCall[];
  jsonClassName?: string;
}) {
  const evidence = groupInvestigationEvidence(calls);
  return (
    <section className="section-gap">
      <h3>Executed tools and evidence</h3>
      {evidence.map(({ id, title, payload }) => (
        <details key={id}>
          <summary>{title}</summary>
          <pre className={jsonClassName}>{JSON.stringify(payload, null, 2)}</pre>
        </details>
      ))}
      {!evidence.length && <p>No tool calls recorded yet.</p>}
    </section>
  );
}
