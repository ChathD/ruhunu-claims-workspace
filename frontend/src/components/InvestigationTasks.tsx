import type { Finding, InvestigationRound } from "../types";

const specialistLabels: Record<string, string> = {
  coverage: "Coverage",
  fraud_initial: "Fraud",
  fraud_followup: "Fraud follow-up",
  historical: "Historical comparison",
};

export default function InvestigationTasks({
  rounds = [],
  findings = {},
}: {
  rounds?: InvestigationRound[];
  findings?: Record<string, Finding>;
}) {
  const coverageBlocked = findings.intake?.details.coverage_readiness?.ready === false;
  const fraud = findings.fraud_initial?.details;
  const fraudCompleted = fraud?.checks_executed && !fraud.error_type;
  return (
    <section className="investigation-tasks">
      <h2>Investigation</h2>
      {!rounds.length && (
        <p>
          {coverageBlocked && fraudCompleted
            ? "Fraud screened the available evidence while Coverage was blocked. See the Intake findings for what Coverage needs."
            : coverageBlocked
              ? "Coverage is blocked. See the Intake findings for the outstanding requirements."
              : "No model-assigned tasks with completed findings."}
        </p>
      )}
      {rounds.map(({ round, assignments }) => (
        <section className="investigation-round" key={round}>
          <h3>Round {round}</h3>
          <ul>
            {assignments.map(({ specialist, task }) => (
              <li key={specialist}>
                <h4>{specialistLabels[specialist] ?? specialist}</h4>
                <p>{task}</p>
              </li>
            ))}
          </ul>
        </section>
      ))}
    </section>
  );
}
