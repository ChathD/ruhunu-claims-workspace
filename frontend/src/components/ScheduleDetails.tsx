import { Fragment, type ReactNode } from "react";
import { lossTypeLabel, money } from "../components";

export type ScheduleSummary = {
  id: string;
  policy_number: string;
  insured_name: string;
  cover_type: "comprehensive" | "third_party" | "third_party_fire_theft";
  starts_on: string;
  ends_on: string;
  wording: { id: string; title: string; product: string; version: string; document_kind?: "wording" | "product_summary" };
};

type Citation = { reference: string; quote: string; page: number; ocr: boolean };

export type SavedPolicySchedule = ScheduleSummary & {
  insured_address: string;
  district: string;
  postal_code: string;
  vehicle: { registration: string; make: string; model: string; year: number; chassis_number: string; engine_number: string;
    vehicle_class: string; body_type: string; fuel: string; engine_capacity_cc: number; seats: number };
  issued_on: string;
  territorial_limits: string;
  permitted_use: string;
  authorised_drivers: string;
  named_drivers: string[];
  cover: { loss_type: string; benefit_name?: string;
    limit_basis: "wording_amount" | "wording_formula" | "scheduled_amount" | "market_value" | "statutory" | "unlimited" | "excluded" | "unspecified" | "unconfirmed";
    limit_lkr: string | null; compulsory_excess_lkr: string | null; voluntary_excess_lkr: string | null; conditions: string; wording: Citation }[];
  endorsements: string[];
  special_conditions: string[];
  claims_instructions: string;
  base_premium_lkr: string;
  no_claim_discount_lkr: string;
  charges: { label: string; category: string; amount_lkr: string }[];
  total_premium_lkr: string;
  wording_references: Citation[];
};

export const coverLabels = { comprehensive: "Comprehensive", third_party: "Third-party only", third_party_fire_theft: "Third-party, fire and theft" };
const limitLabels = { wording_formula: "Calculated from wording", market_value: "Market value", statutory: "Statutory limit", unlimited: "Unlimited", excluded: "Excluded",
  unspecified: "Limit unspecified", unconfirmed: "Cover unconfirmed" };
const formatter = new Intl.DateTimeFormat("en-GB", { dateStyle: "medium", timeZone: "Asia/Colombo" });
export const scheduleDate = (value: string) => formatter.format(new Date(value));
const lkr = (value: string) => `LKR ${money(value)}`;
const excess = (value: string | null) => value === null ? "Not stated" : lkr(value);

function Facts({ fields }: { fields: [string, ReactNode][] }) {
  return <dl className="facts">{fields.map(([label, value]) => <Fragment key={label}><dt>{label}</dt><dd>{value}</dd></Fragment>)}</dl>;
}

const passageNumber = String.raw`\bW\d{3,}\b`;
const passageList = String.raw`${passageNumber}(?:\s*(?:[,/&–-]|and|to)\s*${passageNumber})*`;
const sectionNumber = String.raw`(?:[IVXLCDM]+|\d+)\b(?:[-–.]\d+)*(?:\s*\((?:[a-z]|\d+|[ivxlcdm]+)\))*`;
const sectionList = String.raw`(?:sections?|clauses?)\s+${sectionNumber}(?:\s*(?:[,/&–-]|and|or)\s*${sectionNumber})*`;
const wordingReferences = new RegExp(
  String.raw`\(\s*(?:see\s+)?(?:${passageList}|${sectionList})\s*\)|\[\s*${passageList}\s*\]|\b(?:(?:under|see|per|refer to)\s+)?(?:${sectionList}\s*:\s*|${sectionList}|${passageList})`,
  "gi",
);

function conditionDetails(text: string) {
  return text.replace(wordingReferences, "").replace(/\s+([,.;:])/g, "$1").replace(/ {2,}/g, " ").trim()
    .replace(/^\p{Ll}/u, letter => letter.toUpperCase());
}

export default function ScheduleDetails({ schedule }: { schedule: SavedPolicySchedule }) {
  const { vehicle, wording } = schedule;
  const pdf = `/api/policy-documents/${wording.id}/preview`;
  return <section className="card policy-schedule">
    <h1>Motor policy schedule · {schedule.policy_number}</h1>
    <p>Synthetic Sri Lankan motor schedule · <a href={pdf} target="_blank" rel="noreferrer">{wording.title} · {wording.version}</a></p>
    {wording.document_kind === "product_summary" ? <p>Provisional schedule from a product summary. Confirm terms against the full policy wording.</p> : null}
    <div className="form-grid">
      <div><h3>Policyholder and period</h3><Facts fields={[
        ["Policyholder", schedule.insured_name], ["Postal address", `${schedule.insured_address}, ${schedule.postal_code}`],
        ["District", schedule.district], ["Issued on", scheduleDate(schedule.issued_on)],
        ["Period of insurance", `${scheduleDate(schedule.starts_on)} to ${scheduleDate(schedule.ends_on)}`],
        ["Cover type", coverLabels[schedule.cover_type]],
      ]} /></div>
      <div><h3>Insured vehicle</h3><Facts fields={[
        ["Registration", vehicle.registration], ["Make and model", `${vehicle.make} ${vehicle.model} (${vehicle.year})`],
        ["Class and body", `${vehicle.vehicle_class}, ${vehicle.body_type}`], ["Chassis number", vehicle.chassis_number],
        ["Engine number", vehicle.engine_number], ["Fuel and capacity", `${vehicle.fuel} · ${vehicle.engine_capacity_cc} cc`],
        ["Seats", vehicle.seats],
      ]} /></div>
    </div>
    <h3>Cover, limits and excesses</h3>
    <div className="table-scroll"><table>
      <thead><tr><th>Benefit</th><th>Limit</th><th>Compulsory excess</th><th>Voluntary excess</th><th>Conditions</th></tr></thead>
      <tbody>{schedule.cover.map(benefit => <tr key={benefit.benefit_name ? `${benefit.loss_type}:${benefit.benefit_name}` : benefit.loss_type}>
        <th scope="row">{benefit.benefit_name ?? lossTypeLabel(benefit.loss_type)}</th>
        <td>{benefit.limit_basis === "wording_amount" || benefit.limit_basis === "scheduled_amount"
          ? lkr(benefit.limit_lkr!) : limitLabels[benefit.limit_basis]}</td>
        <td>{benefit.limit_basis === "excluded" ? "—" : excess(benefit.compulsory_excess_lkr)}</td>
        <td>{benefit.limit_basis === "excluded" ? "—" : excess(benefit.voluntary_excess_lkr)}</td>
        <td>{conditionDetails(benefit.conditions)}</td>
      </tr>)}</tbody>
    </table></div>
    <div className="form-grid">
      <div><h3>Use, drivers and endorsements</h3><Facts fields={[
        ["Territorial limits", conditionDetails(schedule.territorial_limits)], ["Permitted use", conditionDetails(schedule.permitted_use)],
        ["Authorised drivers", conditionDetails(schedule.authorised_drivers)], ["Named drivers", schedule.named_drivers.join(", ")],
        ["Endorsements", schedule.endorsements.map(conditionDetails).filter(Boolean).join(" · ") || "None selected"],
        ["Special conditions", schedule.special_conditions.map(conditionDetails).filter(Boolean).join(" · ") || "Standard wording applies"],
      ]} /></div>
      <div><h3>Premium breakdown</h3><Facts fields={[
        ["Base premium", lkr(schedule.base_premium_lkr)], ["No-claim discount", lkr(schedule.no_claim_discount_lkr)],
        ...schedule.charges.map((charge): [string, string] => [charge.label, lkr(charge.amount_lkr)]),
        ["Total premium", <strong>{lkr(schedule.total_premium_lkr)}</strong>],
      ]} /></div>
    </div>
    <h3>Claims instructions</h3><p>{conditionDetails(schedule.claims_instructions)}</p>
  </section>;
}
