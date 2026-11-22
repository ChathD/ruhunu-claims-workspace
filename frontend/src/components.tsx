import {
  ArrowDownToLine,
  ArrowUpRight,
  ChevronRight,
  Search,
} from "lucide-react";
import type { Claim, Route } from "./types";
import { routeLabels } from "./types";

export const money = (amount: string | number | null | undefined) =>
  amount == null
    ? "Undetermined"
    : new Intl.NumberFormat("en-LK", { maximumFractionDigits: 2 }).format(
        Number(amount),
      );
export const time = (value: string) =>
  new Date(value).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
const lossTypeLabels: Record<string, string> = { glass: "Glass damage", repair: "Vehicle repair", theft: "Vehicle theft", injury: "Injury", other: "Other",
  third_party_property_damage: "Third-party property damage", third_party_injury: "Third-party injury" };
export const lossTypeLabel = (type: string) => lossTypeLabels[type] ?? type;
export const insuranceTypeLabel = (type: "comprehensive" | "third_party") => type === "comprehensive" ? "Comprehensive" : "Third-party";
export const stageNames: Record<string, string> = {
  intake: "Intake & duplicate check",
  fraud_initial: "Initial fraud investigation",
  coverage: "Coverage & financial checks",
  historical: "Historical claim comparison",
  fraud_followup: "Fraud follow-up",
};
export const routeClass = (route: Route) =>
  ({
    "auto-approve": "green",
    "adjuster review": "amber",
    "request more information": "blue",
    "fraud escalation": "red",
  })[route];

export function Badge({
  route,
  status,
}: {
  route?: Route | null;
  status?: string;
}) {
  return (
    <span
      className={`badge ${route ? routeClass(route) : status === "failed" ? "red" : "neutral"}`}
    >
      <i />
      {route
        ? routeLabels[route]
        : status === "processing"
          ? "Processing"
          : status === "awaiting_review"
            ? "Ready for review"
          : status === "failed"
            ? "Needs retry"
            : "Queued"}
    </span>
  );
}

export function ClaimTable({
  claims,
  open,
  empty = "No claims match these filters.",
}: {
  claims: Claim[];
  open: (id: string) => void;
  empty?: string;
}) {
  return (
    <div className="table-scroll">
      <table className="claims-table">
        <thead>
          <tr>
            <th>Claim</th>
            <th>Loss type</th>
            <th>Claimed amount</th>
            <th>Routing outcome</th>
            <th>Received</th>
            <th>
              <span className="sr-only">Open</span>
            </th>
          </tr>
        </thead>
        <tbody>
          {claims.map((claim) => (
            <tr key={claim.id} onClick={() => open(claim.id)}>
              <td>
                <button
                  className="text-button claim-link"
                  onClick={(e) => {
                    e.stopPropagation();
                    open(claim.id);
                  }}
                >
                  {claim.reference}
                </button>
                <small>{claim.registration}</small>
              </td>
              <td>
                <span className="loss-type">
                  {lossTypeLabel(claim.type)}
                </span>
              </td>
              <td className="tabular">LKR {money(claim.gross_amount)}</td>
              <td>
                <Badge route={claim.route} status={claim.status} />
              </td>
              <td className="muted tabular">{time(claim.received_at)}</td>
              <td>
                <ChevronRight size={16} className="muted" />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {claims.length === 0 && <div className="empty-state">{empty}</div>}
    </div>
  );
}

export function SearchField({
  value,
  onChange,
}: {
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <label className="search-field">
      <Search size={16} />
      <input
        aria-label="Search claims"
        placeholder="Search claim ID or vehicle…"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      />
    </label>
  );
}

export function Metric({
  label,
  value,
  note,
  accent = false,
}: {
  label: string;
  value: string;
  note: string;
  accent?: boolean;
}) {
  return (
    <div className="metric card">
      <div className="metric-label">
        {label}
        <ArrowUpRight size={14} />
      </div>
      <div className="metric-value">{value}</div>
      <div className={accent ? "metric-note accent" : "metric-note"}>
        {note}
      </div>
    </div>
  );
}

export function AuditLink({
  id,
  children = "Export audit",
}: {
  id: string;
  children?: React.ReactNode;
}) {
  return (
    <a className="button secondary" href={`/api/claims/${id}/audit`} download>
      <ArrowDownToLine size={15} />
      {children}
    </a>
  );
}
