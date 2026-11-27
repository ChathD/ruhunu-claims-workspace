import { lossTypeLabel, money } from "../components";

export type DraftPolicySchedule = {
  generated_at: string;
  country: string;
  currency: string;
  wording: { id: string; title: string; product: string; version: string; document_kind: string };
  reported: {
    policy_number: string | null;
    insured_name: string | null;
    insured_address: string | null;
    registration: string | null;
    make: string | null;
    model: string | null;
    year: number | null;
    chassis_number: string | null;
    loss_type: string | null;
    loss_date: string | null;
    reported_date: string | null;
    claim_amount: string | null;
    claim_currency: string | null;
  };
  wording_references: { sections: string[]; page: number; passage_id: string; text: string; ocr: boolean }[];
};

const localDate = new Intl.DateTimeFormat("en-LK", { dateStyle: "medium", timeZone: "Asia/Colombo" });

export default function PolicySchedule({ schedule }: { schedule: DraftPolicySchedule }) {
  const { wording, reported } = schedule;
  const preview = `/api/policy-documents/${wording.id}/preview`;
  const vehicle = [reported.make, reported.model, reported.year].filter(Boolean).join(" ");

  return <section className="card policy-schedule" aria-labelledby="policy-schedule-heading">
    <h1 id="policy-schedule-heading">Draft motor policy schedule</h1>
    <p>{schedule.country} · {schedule.currency} · Generated {localDate.format(new Date(schedule.generated_at))}</p>
    <p>
      <a href={preview} target="_blank" rel="noreferrer">{wording.title} · {wording.product} · {wording.version}</a>
      {wording.document_kind === "product_summary" && " · Product summary (IPID)"}
    </p>
    <div className="form-grid">
      <div>
        <h3>Reported policy and vehicle</h3>
        <dl className="facts">
          <dt>Policy number</dt><dd>{reported.policy_number || "Not supplied"}</dd>
          <dt>Insured name</dt><dd>{reported.insured_name || "Not supplied"}</dd>
          <dt>Insured address</dt><dd>{reported.insured_address || "Not supplied"}</dd>
          <dt>Registration</dt><dd>{reported.registration || "Not supplied"}</dd>
          <dt>Vehicle</dt><dd>{vehicle || "Not supplied"}</dd>
          <dt>Chassis number</dt><dd>{reported.chassis_number || "Not supplied"}</dd>
        </dl>
      </div>
      <div>
        <h3>Reported loss</h3>
        <dl className="facts">
          <dt>Loss type</dt><dd>{reported.loss_type ? lossTypeLabel(reported.loss_type) : "Not supplied"}</dd>
          <dt>Loss date</dt><dd>{reported.loss_date ? localDate.format(new Date(reported.loss_date)) : "Not supplied"}</dd>
          <dt>Reported on</dt><dd>{reported.reported_date ? localDate.format(new Date(reported.reported_date)) : "Not supplied"}</dd>
          <dt>Claim amount</dt><dd>{reported.claim_amount == null ? "Not supplied" : `${reported.claim_currency ?? ""} ${money(reported.claim_amount)}`.trim()}</dd>
        </dl>
      </div>
    </div>
    <h3>Individual terms to confirm</h3>
    <p>Period of insurance, level of cover, sum insured, excess, premium and selected endorsements require verification against the issued policy schedule.</p>
  </section>;
}
