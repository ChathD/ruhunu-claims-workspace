import { ArrowRight } from "lucide-react";
import { Metric } from "../components";
import { routes, routeLabels } from "../types";
import type { Evaluation } from "../types";

export default function EvaluationPage({
  evaluation,
  openClaim,
}: {
  evaluation: Evaluation | null;
  openClaim: (id: string) => Promise<void>;
}) {
  return (
    <>
      <div className="metrics-grid three">
        <Metric
          label="Labelled claims evaluated"
          value={String(evaluation?.evaluated ?? 0)}
          note="Predictions joined to separate labels"
        />
        <Metric
          label="Pending predictions"
          value={String(evaluation?.pending ?? 0)}
          note="Report is partial while processing"
        />
        <Metric
          label="Routing mismatches"
          value={String(evaluation?.errors.length ?? 0)}
          note="Inspect every off-diagonal result"
        />
      </div>
      <section className="card">
        <div className="card-heading">
          <div>
            <h2>Routing confusion matrix</h2>
            <p>Rows are expected routes. Columns are predicted routes.</p>
          </div>
          <span className="badge neutral">
            {evaluation?.complete ? "Complete set" : "Partial set"}
          </span>
        </div>
        <div className="table-scroll">
          <table className="matrix">
            <thead>
              <tr>
                <th>Expected ↓ / Predicted →</th>
                {routes.map((r) => (
                  <th key={r}>{routeLabels[r]}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {routes.map((r, i) => (
                <tr key={r}>
                  <th>{routeLabels[r]}</th>
                  {routes.map((col, j) => (
                    <td
                      className={
                        i === j
                          ? "diagonal"
                          : (evaluation?.matrix[i]?.[j] || 0) > 0
                            ? "off-diagonal"
                            : ""
                      }
                      key={col}
                    >
                      {evaluation?.matrix[i]?.[j] ?? 0}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="panel-note">
          Synthetic scenario labels validate provisional rules. New fixture
          seeds do not establish performance on independently adjudicated real
          claims.
        </div>
      </section>
      {!!evaluation?.errors.length && (
        <section className="card section-gap">
          <div className="card-heading">
            <h2>Routing mismatches</h2>
          </div>
          {evaluation.errors.map((e) => (
            <button
              className="activity-row"
              key={e.assessment_id}
              onClick={() => void openClaim(e.assessment_id)}
            >
              <span>
                Expected {routeLabels[e.expected]} · predicted{" "}
                {routeLabels[e.predicted]}
              </span>
              <ArrowRight size={15} />
            </button>
          ))}
        </section>
      )}
    </>
  );
}
