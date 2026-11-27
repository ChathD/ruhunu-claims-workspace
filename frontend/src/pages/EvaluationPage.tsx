import { useEffect, useState } from "react";
import { api } from "../api";
import type { Evaluation } from "../types";

const percent = new Intl.NumberFormat(undefined, { style: "percent", maximumFractionDigits: 1 });
const formatScore = (score: number | null) => score === null ? "—" : percent.format(score);

export default function EvaluationPage() {
  const [report, setReport] = useState<Evaluation | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [refresh, setRefresh] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    const { signal } = controller;
    void api<Evaluation>("/portal/evaluation", { signal })
      .then((result) => { if (!signal.aborted) setReport(result); })
      .catch((failure) => { if (!signal.aborted) setError(String(failure)); })
      .finally(() => { if (!signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [refresh]);

  function refreshReport() {
    setError("");
    setLoading(true);
    setRefresh((previous) => previous + 1);
  }

  return <main>
    <div className="actions page-heading">
      <h1>Routing evaluation</h1>
      <button type="button" onClick={refreshReport} disabled={loading}>
        {loading && report ? "Refreshing…" : "Refresh"}
      </button>
    </div>
    {error && <p role="alert">Could not load routing evaluation. {error}</p>}
    {!report && loading && <p role="status">Loading routing evaluation…</p>}
    {report && <EvaluationResults report={report} />}
  </main>;
}

function EvaluationResults({ report }: { report: Evaluation }) {
  const classes = report.class_metrics;

  return <>
    <dl className="evaluation-summary card">
      <div><dt>Evaluated claims</dt><dd>{report.evaluated}</dd></div>
      <div><dt>Pending predictions</dt><dd>{report.pending}</dd></div>
      <div><dt>Routing mismatches</dt><dd>{report.mismatches}</dd></div>
    </dl>
    {report.evaluated + report.pending === 0 && <p>No synthetic claims have a seeded route yet.</p>}
    <section className="card">
      <h2 id="routing-matrix-heading">Routing confusion matrix</h2>
      <p id="routing-matrix-description">
        Rows show seeded routes. Columns show published predictions. Support counts evaluated claims per seeded route.
      </p>
      <div className="table-scroll" role="region" aria-labelledby="routing-matrix-heading" tabIndex={0}>
        <table className="routing-matrix routing-numbers" aria-labelledby="routing-matrix-heading" aria-describedby="routing-matrix-description">
          <thead>
            <tr>
              <th scope="col">True ↓ / Predicted →</th>
              {classes.map(({ route }) => <th key={route} scope="col"><code>{route}</code></th>)}
              <th scope="col">Support</th>
            </tr>
          </thead>
          <tbody>
            {classes.map(({ route, support }, row) => <tr key={route}>
              <th scope="row"><code>{route}</code></th>
              {classes.map(({ route: predicted }, column) => {
                const count = report.matrix[row][column];
                const cellClass = row === column ? "diagonal" : count > 0 ? "off-diagonal" : undefined;
                return <td key={predicted} className={cellClass}>{count}</td>;
              })}
              <td className="support">{support}</td>
            </tr>)}
          </tbody>
        </table>
      </div>
    </section>
    <section className="card">
      <h2 id="routing-scores-heading">Per-route scores</h2>
      <p id="routing-scores-description">Scores use evaluated claims. A dash marks a score with no cases in its denominator.</p>
      <div className="table-scroll" role="region" aria-labelledby="routing-scores-heading" tabIndex={0}>
        <table className="routing-numbers" aria-labelledby="routing-scores-heading" aria-describedby="routing-scores-description">
          <thead>
            <tr><th scope="col">Route</th><th scope="col">Precision</th><th scope="col">Recall</th><th scope="col">F1</th></tr>
          </thead>
          <tbody>
            {classes.map(({ route, precision, recall, f1 }) => <tr key={route}>
              <th scope="row"><code>{route}</code></th>
              <td>{formatScore(precision)}</td>
              <td>{formatScore(recall)}</td>
              <td>{formatScore(f1)}</td>
            </tr>)}
          </tbody>
        </table>
      </div>
    </section>
  </>;
}
