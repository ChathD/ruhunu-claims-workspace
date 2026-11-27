export type Route =
  | "auto-approve"
  | "adjuster review"
  | "request more information"
  | "fraud escalation";
export const routes: Route[] = [
  "auto-approve",
  "adjuster review",
  "request more information",
  "fraud escalation",
];
export const routeLabels: Record<Route, string> = {
  "auto-approve": "Auto-approve",
  "adjuster review": "Adjuster review",
  "request more information": "Request more information",
  "fraud escalation": "Fraud escalation",
};
export type InformationRequest = { document: string; location: string | null; reason: string };
export type PoliceReportDocument = {
  id: string;
  type: "police_report";
  name: string;
  registration: string;
  date: string;
  station: string;
  summary: string;
};
export type ChargeReceiptDocument = {
  id: string;
  type: "additional_cost_receipt";
  name: string;
  registration: string;
  date: string;
  total: string;
  category: string;
  description: string;
};
export interface Claim {
  can_reprocess?: boolean;
  replacement_id?: string | null;
  processing_alerts?: string[];
  id: string;
  claim_id: string;
  batch_id: string;
  reference: string;
  type: string;
  gross_amount: string;
  registration: string;
  received_at: string;
  finished_at: string | null;
  status: string;
  route: Route | null;
  task: {
    id: string;
    queue: string;
    status: string;
    reviewer: string | null;
  } | null;
}
export interface Finding {
  agent: string;
  status: string;
  summary: string;
  evidence: string[];
  details: Record<string, any>;
}
export interface InvestigationToolCall {
  id: number;
  stage: string;
  tool: string;
  request: Record<string, unknown>;
  response: Record<string, unknown>;
}
export type InvestigationRound = {
  round: number;
  assignments: { specialist: string; task: string }[];
};
export type VehicleDetails = {
  make?: string | null;
  model?: string | null;
  year?: number | null;
  chassis_number?: string | null;
};
export type SupportingIncidentDocument = {
  id: string; type: "incident_report" | "witness_statement"; name: string;
  author: string; registration: string; date: string; summary: string;
};

export type PersonPolicyRecord = {
  person: { person_id: string; full_name: string; district: string; locality?: string | null };
  wording: {
    id: string; title: string; filename: string; version: string;
    status: "queued" | "indexing" | "ready" | "failed"; error: string | null;
  } | null;
  policy: {
    policy_id: string;
    insurance_type?: "comprehensive" | "third_party";
    vehicle_type?: string;
    permitted_use?: string;
    use_conditions?: string[];
    registration: string;
    vehicle?: VehicleDetails | null;
    repair_terms?: {
      description: string;
      covered_incidents?: string[];
      flood_damage?: "covered" | "excluded";
      covered_repairs?: string[];
      repair_benefits?: { cause: string; description: string; covered_repairs: string[]; conditions: string[]; exclusions: string[] }[];
      exclusions: string[];
      labour?: { basis: "no_separate_limit" | "per_panel" | "per_claim"; limit: string | null };
      painting?: { scope: "excluded" | "affected_panels_only" | "full_respray_if_required"; limit_per_panel: string | null; max_panels: number | null };
      additional_charges?: {
        covered: boolean;
        eligible_charges: { category: string; name: string; conditions: string; limit_per_incident: string | null; receipt_required: boolean }[];
        exclusions: { category: string; name: string; reason: string }[];
      };
    } | null;
    theft_terms?: { description: string; basis: "vehicle_valuation_only" | "vehicle_and_listed_fittings"; covered_fittings: string[] } | null;
    liability_terms?: { description: string; conditions: string[]; exclusions: string[] };
    starts_on: string;
    ends_before: string;
    covered_losses: string[];
    police_report_required_for?: string[];
    supporting_documents?: {
      loss_type: string; document_type: SupportingIncidentDocument["type"];
      requirement: "required" | "optional"; purpose: string;
    }[];
    limit: string;
    deductible: string;
    clause: string;
    version: string;
  };
};
export interface Detail extends Claim {
  submission: Record<string, any>;
  synthetic_evidence?: { historical_costs: Record<string, any>[] } | null;
  context: {
    findings: Record<string, Finding>;
    skipped: Record<string, string>;
    [key: string]: any;
  };
  version: number;
  error: string | null;
  decision: {
    route: Route;
    justification: string;
    reason_codes: string[];
    payable_amount: string | null;
    recommendation?: { route: Route; explanation: string; evidence: string[]; outstanding_gaps: string[] } | null;
    override_reason?: string | null;
  } | null;
  runs: {
    id: string;
    stage: string;
    finding: Finding;
    instruction: string;
    input_version: number;
    duration_ms: number;
  }[];
  investigation_tasks: InvestigationRound[];
  tool_calls?: InvestigationToolCall[];
}
export interface Overview {
  total: number;
  completed: number;
  processing: number;
  failed: number;
  human_pending: number;
  within_15_minutes: number | null;
  average_seconds: number | null;
  cost_per_claim: number | null;
  model_calls: number;
  breakdown: Record<Route, number>;
  volume: { date: string; label: string; claims: number }[];
  recent_claims: Claim[];
  rules: string;
}
export interface Evaluation {
  matrix: number[][];
  class_metrics: {
    route: "auto_approval" | "adjuster_review" | "need_information" | "fraud_escalation";
    support: number;
    precision: number | null;
    recall: number | null;
    f1: number | null;
  }[];
  evaluated: number;
  pending: number;
  mismatches: number;
}
export interface ActivityEvent {
  id: number;
  assessment_id: string;
  kind: string;
  payload: Record<string, any>;
  created_at: string;
}

export type Page =
  | "Overview"
  | "Claims"
  | "Human review"
  | "Synthetic generator"
  | "People & policies"
  | "Agent activity"
  | "Evaluation"
  | "Audit trail"
  | "Policy documents"
  | "Settings";
