import { lossTypeLabel, money, stageNames } from "../components";
import type { Finding } from "../types";

type Amount = string | number | null;
type FactRow = [label: string, value: string | number | null | undefined];
type CoverageOutcome = "covered" | "not_covered" | "undetermined";
type CoverageState = "confirmed" | "partially_confirmed" | "unconfirmed";
type CoverageItem = { item: string; outcome: CoverageOutcome; reason: string; policy_term?: string; passage_refs?: string[]; document_refs?: string[] };
type WordingRetrieval = { title?: string; version?: string; passages?: { source: string; page: number; text: string }[] };
type PolicyConflict = {
  item: string; structured_term: string; wording_term: string; impact: string;
  materiality: "material" | "immaterial" | "uncertain"; passage_refs: string[];
};
type FindingDetails = {
  facts?: { field: string; value: string | number | null }[];
  analysis?: { classified_loss?: string | null };
  coverage_readiness?: {
    ready: boolean;
    checks: { policy_period: boolean; vehicle_match: boolean; loss_and_costs: boolean };
    policy_available?: boolean | null;
    internal_issue?: string | null;
  };
  missing?: { document: string; location?: string | null; reason: string }[];
  review_reasons?: string[];
  duplicate_matches?: unknown[];
  error_type?: string;
  error_message?: string;
  coverage_status?: string;
  coverage_reason?: string;
  coverage_unconfirmed_reason?: string;
  coverage_outcome?: { state: CoverageState; reason: string };
  coverage_historical_context?: { reason: string; claim_references: string[] };
  coverage_items?: CoverageItem[];
  submitted_documents?: Record<string, string>;
  document_inconsistencies?: { reason: string; document_refs: string[] }[];
  policy_conflicts?: PolicyConflict[];
  policy_rag?: { retrievals?: WordingRetrieval[] };
  policy_omissions?: { item: string; impact: string }[];
  vehicle_mismatches?: string[];
  missing_policy_fields?: string[];
  policy?: { policy_id?: string; starts_on?: string; ends_before?: string; covered_losses?: string[] };
  financial?: { gross?: Amount; cost_documents_total?: Amount; additional_costs_total?: Amount; limit?: Amount; deductible?: Amount; payable?: Amount };
  amount_justification?: { conclusion: string; sample_size: number; median?: Amount; observed_range?: [Amount, Amount] | null; cost_basis?: string; limitation?: string };
  fraud_investigation?: { leads: { pattern: string; document_reference?: string; alternative?: string }[] };
  fraud_indicators?: { reason: string; alternative?: string }[];
  fraud_concerns?: { severity: "high" | "medium" | "low"; reason: string }[];
  plausibility?: { assessment: "consistent" | "inconclusive" | "contradicted"; reason: string };
};

const statusLabels: Record<string, string> = {
  pass: "Passed",
  fraud: "Indicator raised",
  information: "Needs information",
  review: "Review required",
};

const coverageLabels: Record<CoverageState, string> = {
  confirmed: "Confirmed",
  partially_confirmed: "Partially confirmed",
  unconfirmed: "Unconfirmed",
};

const resultLabel = (value?: string) => {
  const words = value?.toLowerCase().replaceAll("_", " ");
  return words ? words[0].toUpperCase() + words.slice(1) : undefined;
};

const lkr = (value: Amount | undefined) => value == null ? undefined : `LKR ${money(value)}`;
const fieldLabels: Record<string, string> = {
  policy_id: "policy number",
  person_id: "policyholder number",
  loss_date: "date of loss",
  loss_type: "type of loss",
  gross_amount: "claimed amount",
  reported_date: "date reported",
  additional_costs: "additional costs",
  "vehicle.chassis_number": "chassis number",
  chassis_number: "chassis number",
  document_date: "cost document date",
  document_reference: "cost document reference",
  garage_name: "garage name",
  garage_address: "garage address",
  line_items: "itemised charges",
  covered_losses: "covered loss types",
  starts_on: "policy start date",
  ends_before: "policy end date",
  repair_terms: "repair terms",
  theft_terms: "theft terms",
  covered_repairs: "covered repairs",
  covered_fittings: "covered fittings",
};

function readableText(value: string) {
  return value.replace(/\b(?:vehicle\.)?[a-z]+(?:_[a-z]+)+\b/gi,
    field => fieldLabels[field.toLowerCase()] ?? field.replaceAll("_", " "));
}

function FactSection({ title, rows }: { title: string; rows: FactRow[] }) {
  const populated = rows.filter(([, value]) => value != null && value !== "");
  if (!populated.length) return null;
  return (
    <section className="finding-section">
      <h4>{title}</h4>
      <dl className="finding-facts">
        {populated.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}
      </dl>
    </section>
  );
}

function IntakeFacts({ details }: { details: FindingDetails }) {
  if (!details.facts?.length) return null;
  const facts = Object.fromEntries(details.facts.map(({ field, value }) => [field, value]));
  return (
    <FactSection title="Claim information" rows={[
      ["Policy number", facts.policy_id],
      ["Loss date", facts.loss_date],
      ["Vehicle registration", facts.registration],
      ["Declared loss type", facts.loss_type ? lossTypeLabel(String(facts.loss_type)) : undefined],
      ["Claimed amount", facts.gross_amount != null ? `${facts.currency ?? ""} ${money(facts.gross_amount)}`.trim() : undefined],
    ]} />
  );
}

function IntakeReadiness({ details }: { details: FindingDetails }) {
  const readiness = details.coverage_readiness;
  if (!readiness) return null;
  const checkLabel = (ready: boolean) => ready ? "Enough information" : "Blocked";
  const status = details.error_type ? "Processing interrupted" : readiness.internal_issue ? "Needs internal review"
    : readiness.ready ? "Ready to start" : "Waiting for evidence";
  return <FactSection title="Readiness for Coverage" rows={[
    ["Dispatch status", status],
    ["Saved policy", readiness.policy_available === true ? "Found" : readiness.policy_available === false ? "Could not verify" : undefined],
    ["Evidence for policy period", checkLabel(readiness.checks.policy_period)],
    ["Evidence for vehicle identity", checkLabel(readiness.checks.vehicle_match)],
    ["Evidence for loss and cost eligibility", checkLabel(readiness.checks.loss_and_costs)],
    ["Blocking issue", readiness.internal_issue],
  ]} />;
}

function FraudSeverityGroups({ concerns }: { concerns: NonNullable<FindingDetails["fraud_concerns"]> }) {
  const groups = ["high", "medium", "low"] as const;
  return (
    <div className="finding-groups">
      {groups.map(severity => {
        const group = concerns.filter(concern => concern.severity === severity);
        if (!group.length) return null;
        return (
          <section className="finding-section" key={severity}>
            <h4>{resultLabel(severity)} severity</h4>
            <ul>{group.map((concern, index) => <li key={index}>{readableText(concern.reason)}</li>)}</ul>
          </section>
        );
      })}
    </div>
  );
}

function undeterminedReason(details: FindingDetails) {
  if (details.error_type && ["ReadTimeout", "ConnectTimeout", "WriteTimeout", "PoolTimeout", "TimeoutException"].includes(details.error_type)) {
    return "The Coverage check timed out before returning a result.";
  }
  if (details.error_type === "ModelOutputTruncated") return "The Coverage response reached its length limit before a decision was returned.";
  if (details.error_message === "Incomplete model response") return "Coverage started but the model did not return a complete decision.";
  if (details.error_type) return "The specialist could not complete the coverage investigation.";
  if (details.vehicle_mismatches?.length) {
    return `The reported vehicle differs from the insured vehicle in ${details.vehicle_mismatches.join(", ")}.`;
  }
  if (details.missing_policy_fields?.length) {
    return `The saved policy is missing terms needed for this claim, including ${details.missing_policy_fields.map(readableText).join(", ")}.`;
  }
  return details.missing?.[0]?.reason ?? "The available policy and claim information could not establish coverage.";
}

function WordingCitations({ references, retrievals = [] }: { references?: string[]; retrievals?: WordingRetrieval[] }) {
  return references?.map(reference => {
    const retrieval = retrievals.find(result => result.passages?.some(passage => passage.source === reference));
    const passage = retrieval?.passages?.find(item => item.source === reference);
    if (!passage) return null;
    const passageNumber = Number(reference.split(":").at(-1));
    return <details className="finding-note" key={reference}>
      <summary>{retrieval?.title ?? "Policy wording"}{retrieval?.version && ` · v${retrieval.version}`} · Page {passage.page}
        {Number.isInteger(passageNumber) && ` · Passage ${passageNumber + 1}`}</summary>
      <p>{passage.text}</p>
    </details>;
  });
}

function CoverageResults({ details }: { details: FindingDetails }) {
  const items = (details.coverage_items ?? []).filter(item => item.policy_term);
  if (!items.length && details.coverage_status === "INACTIVE" && details.policy?.starts_on && details.policy.ends_before) {
    items.push({item: "Reported loss", outcome: "not_covered",
      policy_term: `Cover starts on ${details.policy.starts_on} and ends before ${details.policy.ends_before}.`,
      reason: "The policy was outside its effective period on the date of loss."});
  }
  if (!items.length && details.coverage_status === "COVERED" && details.policy?.covered_losses?.length) {
    const coveredLosses = details.policy.covered_losses.map(lossTypeLabel).join(", ");
    items.push({item: "Reported loss", outcome: "covered", policy_term: `The policy lists ${coveredLosses} as covered losses.`,
      reason: "The reported loss type matched the saved policy's list of covered losses."});
  }
  const groups: [CoverageOutcome, string][] = [
    ["covered", "Covered"],
    ["not_covered", "Not covered"],
    ["undetermined", "Cannot be determined"],
  ];
  return (
    <div className="finding-groups">
      {groups.map(([outcome, title]) => {
        const group = items.filter(item => item.outcome === outcome);
        return (
          <section className="finding-section" key={outcome}>
            <h4>{title}</h4>
            {group.length ? (
              <ul>
                {group.map(({item, reason, policy_term, passage_refs, document_refs}, index) => {
                  const names = (document_refs ?? []).flatMap(ref => details.submitted_documents?.[ref] ?? []);
                  return (
                    <li key={index}>
                      <strong>{readableText(item)}</strong>
                      <p className="finding-note">Policy term · {readableText(policy_term ?? "")}</p>
                      <p>{readableText(reason)}</p>
                      {names.length > 0 && <p className="finding-note">Claim documents · {names.join(", ")}</p>}
                      <WordingCitations references={passage_refs} retrievals={details.policy_rag?.retrievals} />
                    </li>
                  );
                })}
              </ul>
            ) : <p className="finding-note">None identified.</p>}
          </section>
        );
      })}
    </div>
  );
}

export default function SpecialistFinding({ finding }: { finding: Finding }) {
  const details: FindingDetails = finding.details;
  const { policy, financial, amount_justification: comparison } = details;
  const leads = details.fraud_investigation?.leads ?? [];
  const fraudConcerns = details.fraud_concerns ?? [];
  const omissions = details.policy_omissions ?? [];
  const coverageIssue = details.error_type || details.coverage_status === "UNRESOLVED" || details.coverage_status === "NEEDS_INFORMATION";
  const coverageReason = finding.agent === "coverage"
    ? details.coverage_outcome?.reason ?? details.coverage_unconfirmed_reason ?? (coverageIssue ? details.coverage_reason ?? undeterminedReason(details) : null)
    : null;
  const coverageLabel = finding.agent === "coverage" && details.coverage_outcome ? coverageLabels[details.coverage_outcome.state] : null;
  const reasons = [...new Set([
    ...(details.review_reasons ?? []),
    ...(details.duplicate_matches?.length ? ["An earlier claim matches the vehicle and incident. Check whether this is a duplicate."] : []),
    ...(details.error_type && finding.agent !== "coverage" ? ["The specialist could not complete its investigation. Reprocess the claim to continue."] : []),
  ].map(readableText))].filter(reason => !coverageReason || reason !== readableText(coverageReason));
  return (
    <article className="finding specialist-finding">
      <header className="finding-heading">
        <h3>{stageNames[finding.agent] ?? resultLabel(finding.agent)}</h3>
        <span className={`finding-status ${coverageLabel ? details.coverage_outcome?.state === "confirmed" ? "pass" : "review" : finding.status}`}>
          {coverageLabel ?? (details.error_type ? "Processing interrupted" : statusLabels[finding.status] ?? resultLabel(finding.status))}
        </span>
      </header>
      {finding.agent === "intake" && <IntakeFacts details={details} />}
      {finding.agent === "intake" && <IntakeReadiness details={details} />}
      {finding.agent === "coverage" && (
        <>
          {coverageReason && <section className="finding-section">
            <h4>Coverage rationale</h4>
            <p>{readableText(coverageReason)}</p>
          </section>}
          {details.coverage_historical_context && <section className="finding-section">
            <h4>How earlier decisions informed coverage</h4>
            <p>{readableText(details.coverage_historical_context.reason)}</p>
          </section>}
          {details.analysis?.classified_loss && <FactSection title="Loss classification" rows={[["Assessed loss type", lossTypeLabel(details.analysis.classified_loss)]]} />}
          <CoverageResults details={details} />
        </>
      )}
      {!!details.document_inconsistencies?.length && (
        <section className="finding-section">
          <h4>Document inconsistencies</h4>
          <ul>{details.document_inconsistencies.map(({reason, document_refs}, index) => (
            <li key={index}>
              <p>{readableText(reason)}</p>
              <p className="finding-note">Claim documents · {document_refs.map(ref => details.submitted_documents?.[ref]).filter(Boolean).join(", ")}</p>
            </li>
          ))}</ul>
        </section>
      )}
      {!!details.policy_conflicts?.length && <section className="finding-section">
        <h4>Differences between policy wording and schedule</h4>
        <ul>{details.policy_conflicts.map(conflict => (
          <li key={conflict.item}>
            <strong>{conflict.item} · {resultLabel(conflict.materiality)}</strong>
            <p className="finding-note">Schedule · {conflict.structured_term}</p>
            <p className="finding-note">Wording · {conflict.wording_term}</p>
            <p>{conflict.impact}</p>
            <WordingCitations references={conflict.passage_refs} retrievals={details.policy_rag?.retrievals} />
          </li>
        ))}</ul>
      </section>}
      {finding.agent === "coverage" && omissions.length > 0 && (
        <section className="finding-section">
          <h4>Matters the policy does not address</h4>
          <ul>{omissions.map(({item, impact}, index) => (
            <li key={index}><strong>{readableText(item)}</strong><p>{readableText(impact)}</p></li>
          ))}</ul>
        </section>
      )}
      {policy && (
        <FactSection title="Policy checks" rows={[
          ["Policy number", policy.policy_id],
          ["Effective from", policy.starts_on],
          ["Ends before", policy.ends_before],
        ]} />
      )}
      {financial && (
        <FactSection title="Financial checks" rows={[
          ["Claimed amount", lkr(financial.gross)],
          ["Cost documents", lkr(financial.cost_documents_total)],
          ["Additional costs", lkr(financial.additional_costs_total)],
          ["Policy limit", lkr(financial.limit)],
          ["Deductible", lkr(financial.deductible)],
          ["Supported payable", financial.payable == null ? "Undetermined" : lkr(financial.payable)],
        ]} />
      )}
      {comparison && (
        <>
          <FactSection title="Cost comparison" rows={[
            ["Amount assessment", resultLabel(comparison.conclusion)],
            ["Comparable claims", comparison.sample_size],
            ["Median", lkr(comparison.median)],
            ["Observed range", comparison.observed_range?.every(value => value != null)
              ? comparison.observed_range.map(lkr).join(" to ") : undefined],
            ["Cost basis", comparison.cost_basis],
          ]} />
          {comparison.limitation && <p className="finding-note">{comparison.limitation}</p>}
        </>
      )}
      {leads.length > 0 && (
        <section className="finding-section">
          <h4>Fraud leads</h4>
          <ul>{leads.map((lead, index) => <li key={index}>{readableText(lead.pattern)}{lead.document_reference && ` · ${lead.document_reference}`}</li>)}</ul>
        </section>
      )}
      {!!details.fraud_indicators?.length && (
        <section className="finding-section">
          <h4>Verified fraud indicators</h4>
          {details.fraud_indicators.map((flag, index) => (
            <div className="indicator" key={index}>
              <p>{readableText(flag.reason)}</p>
              {flag.alternative && <small>Alternative explanation · {readableText(flag.alternative)}</small>}
            </div>
          ))}
        </section>
      )}
      {details.plausibility && (
        <section className="finding-section">
          <h4>Claim plausibility</h4>
          <strong>{details.plausibility.assessment === "consistent" ? "Consistent with submitted evidence"
            : details.plausibility.assessment === "contradicted" ? "Conflicting evidence" : "Inconclusive"}</strong>
          <p>{readableText(details.plausibility.reason)}</p>
        </section>
      )}
      {fraudConcerns.length > 0 && (
        <section className="finding-section">
          <h4>Fraud concerns</h4>
          <FraudSeverityGroups concerns={fraudConcerns} />
        </section>
      )}
      {!!details.missing?.length && (
        <section className="finding-section">
          <h4>Missing information</h4>
          {details.missing.map((gap, index) => (
            <div className="evidence-gap" key={index}>
              <strong>{gap.document}{gap.location && ` · ${readableText(gap.location)}`}</strong>
              <span>{readableText(gap.reason)}</span>
            </div>
          ))}
        </section>
      )}
      {finding.agent !== "intake" && !!reasons.length && (
        <section className="finding-section">
          <h4>Review concerns</h4>
          <ul>{reasons.map(reason => <li key={reason}>{reason}</li>)}</ul>
        </section>
      )}
    </article>
  );
}
