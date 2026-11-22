import { useEffect, useState } from "react";
import App from "./App";
import ClaimDetailDialog from "./components/ClaimDetailDialog";
import InvestigationTasks from "./components/InvestigationTasks";
import InvestigationTools from "./components/InvestigationTools";
import AdditionalCosts from "./components/AdditionalCosts";
import { VehicleFacts } from "./components/PolicyTerms";
import SpecialistFinding from "./components/SpecialistFinding";
import SyntheticPoliceReport from "./components/SyntheticPoliceReport";
import SyntheticIncidentEvidence from "./components/SyntheticIncidentEvidence";
import type { Finding, InformationRequest, InvestigationRound, InvestigationToolCall } from "./types";
import { api } from "./api";
import { lossTypeLabel, money } from "./components";

type Actor = { id: string; email: string; role: "user" | "staff" | "admin" };
type PortalClaim = {
  id: string;
  reference: string;
  source: string;
  status: string;
  reason: string | null;
  received_at: string;
  reviewed_at: string | null;
  submission: Record<string, any>;
  triage: Record<string, any> | null;
  can_reprocess?: boolean;
  processing_error?: string | null;
  processing_alerts?: string[];
  information_required: InformationRequest[];
  findings: Record<string, Finding>;
  attachments: { id: string; name: string }[];
  investigation?: { tasks: InvestigationRound[]; extraction: Record<string, any>[]; tools: InvestigationToolCall[]; replacement_id: string | null } | null;
};

export default function Portal() {
  const [actor, setActor] = useState<Actor | null>(null);
  const [ready, setReady] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [register, setRegister] = useState(false);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [screen, setScreen] = useState<"dashboard" | "claims" | "accounts" | "people-policies">("dashboard");
  const [dashboardVisit, setDashboardVisit] = useState(0);
  function openDashboard(target: "dashboard" | "people-policies") {
    setScreen(target);
    setDashboardVisit(visit => visit + 1);
  }
  useEffect(() => {
    api<Actor>("/auth/me")
      .then(setActor)
      .catch(() => {})
      .finally(() => setReady(true));
  }, []);
  async function signIn(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      if (register)
        await api("/auth/register", {
          method: "POST",
          body: JSON.stringify({ email, password }),
        });
      const account = await api<Actor>("/auth/login", {
        method: "POST",
        body: JSON.stringify({ email, password }),
      });
      setActor(account);
      setPassword("");
      setScreen(account.role === "admin" ? "dashboard" : "claims");
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }
  if (!ready) return <div className="auth-shell">Loading workspace…</div>;
  if (!actor)
    return (
      <div className="auth-shell">
        <form className="card login-panel" onSubmit={signIn}>
          <div className="brand-mark">r·</div>
          <div className="eyebrow">RUHUNU CLAIMS WORKSPACE</div>
          <h1>{register ? "Create your user account" : "Welcome back"}</h1>
          <p>
            {register
              ? "Submit claims and follow their progress."
              : "Sign in to your claims workspace."}
          </p>
          {error && (
            <div className="message error" role="alert">
              {error}
            </div>
          )}
          <label className="form-field">
            Email
            <input
              type="email"
              required
              autoComplete="username"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </label>
          <label className="form-field">
            Password
            <input
              type="password"
              minLength={10}
              maxLength={128}
              required
              autoComplete={register ? "new-password" : "current-password"}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </label>
          <button className="button primary" disabled={busy}>
            {busy
              ? "Please wait…"
              : register
                ? "Create account & sign in"
                : "Sign in"}
          </button>
          <button
            type="button"
            className="text-button"
            onClick={() => {
              setRegister(!register);
              setError("");
            }}
          >
            {register
              ? "Already registered? Sign in"
              : "New user? Create an account"}
          </button>
          <details>
            <summary>Local demonstration accounts</summary>
            <p>
              user@ruhunu.local / UserDemo!2026
              <br />
              staff@ruhunu.local / StaffDemo!2026
              <br />
              admin@ruhunu.local / AdminDemo!2026
            </p>
            <small>
              Development accounts only. Do not expose this local app publicly.
            </small>
          </details>
        </form>
      </div>
    );
  return (
    <div className="authenticated">
      <header className="account-bar">
        <strong>Ruhunu</strong>
        <span>
          {actor.email} · {actor.role}
        </span>
        <div>
          {actor.role === "admin" && (
            <button
              onClick={() => openDashboard("dashboard")}
              className="text-button"
            >
              Dashboard
            </button>
          )}
          <button className="text-button" onClick={() => setScreen("claims")}>
            {actor.role === "user" ? "My claims" : "Claims & reviews"}
          </button>
          {actor.role === "admin" && (
            <>
              <button className="text-button" onClick={() => openDashboard("people-policies")}>
                People &amp; policies
              </button>
              <button className="text-button" onClick={() => setScreen("accounts")}>
                Accounts
              </button>
            </>
          )}
          <button
            className="text-button"
            onClick={async () => {
              await api("/auth/logout", { method: "POST" });
              setActor(null);
            }}
          >
            Sign out
          </button>
        </div>
      </header>
      {actor.role === "admin" && (screen === "dashboard" || screen === "people-policies") ? (
        <App key={dashboardVisit} initialPage={screen === "people-policies" ? "People & policies" : "Overview"} />
      ) : screen === "accounts" && actor.role === "admin" ? (
        <Accounts />
      ) : (
        <ClaimsPortal actor={actor} />
      )}
    </div>
  );
}

function ClaimsPortal({ actor }: { actor: Actor }) {
  const [claims, setClaims] = useState<PortalClaim[]>([]);
  const [selected, setSelected] = useState<PortalClaim | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [error, setError] = useState("");
  const [outcome, setOutcome] = useState("approved");
  const [reason, setReason] = useState("");
  const [busy, setBusy] = useState(false);
  const [filter, setFilter] = useState("all");
  async function refresh() {
    try {
      setClaims(await api<PortalClaim[]>("/portal/claims"));
    } catch (e) {
      setError(String(e));
    }
  }
  useEffect(() => {
    void refresh();
    const timer = setInterval(() => void refresh(), 5000);
    return () => clearInterval(timer);
  }, []);
  const current = claims.find((c) => c.id === selected?.id) || selected;
  async function review(e: React.FormEvent) {
    e.preventDefault();
    if (!current) return;
    setBusy(true);
    setError("");
    try {
      await api(`/portal/claims/${current.id}/review`, {
        method: "POST",
        body: JSON.stringify({ outcome, reason }),
      });
      setReason("");
      await refresh();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <main className="portal-main">
      <div className="page-heading">
        <div>
          <div className="eyebrow">
            {actor.role === "user" ? "YOUR CLAIMS" : "STAFF WORKSPACE"}
          </div>
          <h1>{actor.role === "user" ? "My claims" : "Claim review"}</h1>
          <p>
            {actor.role === "user"
              ? "Submit a claim and track the staff decision here."
              : "Review the evidence and record an approval or rejection with a reason."}
          </p>
        </div>
        <button
          className="button primary"
          onClick={() => setShowForm(!showForm)}
        >
          {showForm ? "Close submission form" : "Submit a claim"}
        </button>
      </div>
      {error && (
        <div role="alert" className="message error">
          {error}
        </div>
      )}
      {showForm && (
        <ManualForm
          onDone={async () => {
            setShowForm(false);
            await refresh();
          }}
        />
      )}
      <section className="card section-gap">
        <div className="filter-bar">
          <strong>{claims.length} claims</strong>
          <select
            aria-label="Filter status"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          >
            {[
              "all",
              "awaiting review",
              "more information requested",
              "processing",
              "investigation failed",
              "approved",
              "rejected",
            ].map((v) => (
              <option key={v}>{v}</option>
            ))}
          </select>
        </div>
        <div className="table-scroll">
          <table className="claims-table">
            <thead>
              <tr>
                <th>Claim</th>
                <th>Type</th>
                <th>Amount</th>
                <th>Status</th>
                <th>Source</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {claims
                .filter((c) => filter === "all" || c.status === filter)
                .map((c) => (
                  <tr
                    key={c.id}
                    onClick={() => {
                      setSelected(c);
                      setReason("");
                      setOutcome("approved");
                      setError("");
                    }}
                  >
                    <td>{c.reference}</td>
                    <td>{c.submission.loss_type}</td>
                    <td>LKR {money(c.submission.gross_amount)}</td>
                    <td>
                      <span
                        className={`badge ${c.status === "approved" ? "green" : c.status === "rejected" ? "red" : "amber"}`}
                      >
                        {c.status}
                      </span>
                    </td>
                    <td>{c.source}</td>
                    <td>
                      <button
                        className="text-button"
                        aria-label={`View details for ${c.reference}`}
                      >
                        View details →
                      </button>
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
          {claims.length === 0 && (
            <div className="empty-state">
              No claims yet. Use Submit a claim to get started.
            </div>
          )}
        </div>
      </section>
      {current && (
        <ClaimDetailDialog
          reference={current.reference}
          status={current.status}
          onClose={() => setSelected(null)}
        >
          {error && (
            <div role="alert" className="message error">
              {error}
            </div>
          )}
          <div className="claim-summary">
            {current.submission.person_id && <div><span>Person ID</span><strong>{current.submission.person_id}</strong></div>}
            <div>
              <span>Policy</span>
              <strong>{current.submission.policy_id}</strong>
            </div>
            <div>
              <span>Vehicle</span>
              <strong>{current.submission.registration}</strong>
            </div>
            <div>
              <span>Loss date</span>
              <strong>{current.submission.loss_date || "Unreadable"}</strong>
            </div>
            <div>
              <span>Submitted</span>
              <strong>
                {new Date(current.received_at).toLocaleDateString()}
              </strong>
            </div>
          </div>
          <p>{current.submission.narrative}</p>
          <VehicleFacts vehicle={current.submission.vehicle} />
          <AdditionalCosts costs={current.submission.additional_costs} />
          <h3 className="section-gap">Submitted documents</h3>
          {current.attachments.map((f) => (
            <p key={f.id}>
              <a
                className="text-button"
                href={`/api/portal/attachments/${f.id}`}
              >
                {f.name} ↓
              </a>
            </p>
          ))}
          {!current.attachments.length && <p>No uploaded files.</p>}
          {current.submission.documents?.map((d: any) => d.type === "police_report" ? (
            <SyntheticPoliceReport key={d.id} report={d} />
          ) : d.type === "incident_report" || d.type === "witness_statement" ? (
            <SyntheticIncidentEvidence key={d.id} document={d} />
          ) : (
            <pre key={d.id} className="portal-json">
              {JSON.stringify(d, null, 2)}
            </pre>
          ))}
          {actor.role === "user" && !!current.information_required?.length && (
            <section className="section-gap">
              <h3>Information requested</h3>
              <ul>
                {current.information_required.map(({ document, location, reason }) => (
                  <li key={JSON.stringify([document, location, reason])}>
                    <strong>{location ? `${document} · ${location}` : document}</strong>
                    <p>{reason}</p>
                  </li>
                ))}
              </ul>
            </section>
          )}
          {current.reason && (
            <div className="info-strip section-gap">
              <div>
                <strong>{current.status.toUpperCase()}</strong>
                <p>{current.reason}</p>
                <small>
                  Reviewed {new Date(current.reviewed_at!).toLocaleString()}
                </small>
              </div>
            </div>
          )}
          {actor.role !== "user" && (
            <>
              <div className="section-gap">
                <InvestigationTasks rounds={current.investigation?.tasks} findings={current.findings} />
              </div>
              {current.triage?.justification ? (
                <details className="section-gap">
                  <summary>Published triage justification</summary>
                  <div className="justification">{current.triage.justification}</div>
                </details>
              ) : current.status === "investigation failed" ? (
                <section className="section-gap">
                  <h3>Processing interrupted</h3>
                  <p>{current.processing_error || "The AI Agent could not complete this investigation."}</p>
                  {!!current.processing_alerts?.length && <p>{current.processing_alerts.join(" · ")}</p>}
                  {current.can_reprocess && <button className="button primary" disabled={busy} onClick={async () => {
                    setBusy(true); setError("");
                    try {
                      const result = await api<{ id: string }>(`/portal/claims/${current.id}/retry`, { method: "POST" });
                      const list = await api<PortalClaim[]>("/portal/claims");
                      setClaims(list);
                      setSelected(list.find(c => c.id === result.id) || null);
                    } catch (e) { setError(String(e)); } finally { setBusy(false); }
                  }}>{busy ? "Starting…" : "Reprocess with AI Agent"}</button>}
                </section>
              ) : <p>Triage in progress</p>}
              {current.source === "Manual submission" && !current.reason && !Object.keys(current.findings).length && current.triage && <div className="policy-document-actions"><button className="button primary" disabled={busy} onClick={async () => {
                setBusy(true); setError("");
                try {
                  const result = await api<{ id: string }>(`/portal/claims/${current.id}/investigate`, { method: "POST" });
                  const list = await api<PortalClaim[]>("/portal/claims");
                  setClaims(list); setSelected(list.find(c => c.id === result.id) || null);
                } catch (e) { setError(String(e)); } finally { setBusy(false); }
              }}>{busy ? "Starting…" : current.investigation?.replacement_id ? "Open investigation" : "Run investigation"}</button></div>}
              {!!Object.keys(current.findings).length && (
                <section className="section-gap">
                  <h3>Specialist findings</h3>
                  {Object.values(current.findings).map(finding => <SpecialistFinding key={finding.agent} finding={finding} />)}
                </section>
              )}
              {current.investigation && (current.investigation.extraction.length > 0 || current.investigation.tools.length > 0) && (
                <details className="section-gap">
                  <summary>Audit evidence</summary>
                  {!!current.investigation.extraction.length && (
                    <details className="section-gap">
                      <summary>Uploaded document extraction</summary>
                      <pre className="portal-json">{JSON.stringify(current.investigation.extraction, null, 2)}</pre>
                    </details>
                  )}
                  <InvestigationTools calls={current.investigation.tools} jsonClassName="portal-json" />
                </details>
              )}
              {!current.reason && current.triage && !current.investigation?.replacement_id && (
                <form onSubmit={review} className="review-form">
                  <h3>Final staff decision</h3>
                  <label className="form-field">
                    Outcome
                    <select
                      value={outcome}
                      onChange={(e) => setOutcome(e.target.value)}
                    >
                      <option value="approved">Approve claim</option>
                      <option value="rejected">Reject claim</option>
                    </select>
                  </label>
                  <label className="form-field">
                    Reason (required)
                    <textarea
                      minLength={10}
                      maxLength={3000}
                      required
                      value={reason}
                      onChange={(e) => setReason(e.target.value)}
                      placeholder="Explain the evidence and reason for your decision."
                    />
                  </label>
                  <p>
                    This records a final staff decision. It does not issue
                    payment.
                  </p>
                  <button
                    className="button primary"
                    disabled={busy || !current.triage}
                  >
                    {busy ? "Saving…" : "Record final decision"}
                  </button>
                </form>
              )}
            </>
          )}
        </ClaimDetailDialog>
      )}
    </main>
  );
}

function ManualForm({ onDone }: { onDone: () => Promise<void> }) {
  const [files, setFiles] = useState<File[]>([]);
  const [additionalRows, setAdditionalRows] = useState<string[]>([]);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  async function submit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setBusy(true);
    setError("");
    const form = new FormData(e.currentTarget);
    const vehicle: Record<string, string> = {};
    for (const field of ["make", "model", "year", "chassis_number"]) {
      const value = String(form.get(`vehicle_${field}`) ?? "").trim();
      if (value) vehicle[field] = value;
      form.delete(`vehicle_${field}`);
    }
    const amounts = form.getAll("additional_amount");
    const additionalCosts = form.getAll("additional_reason").map((reason, index) => ({ reason, amount: amounts[index] }));
    form.delete("additional_amount");
    form.delete("additional_reason");
    try {
      if (files.length > 5 || files.some((f) => f.size > 2 * 1024 * 1024))
        throw new Error("Use at most five files, each no larger than 2 MB.");
      const attachments = await Promise.all(
        files.map(
          (file) =>
            new Promise<object>((resolve, reject) => {
              const reader = new FileReader();
              reader.onerror = () =>
                reject(new Error("File could not be read"));
              reader.onload = () =>
                resolve({
                  name: file.name,
                  media_type: file.type,
                  content: String(reader.result).split(",")[1],
                });
              reader.readAsDataURL(file);
            }),
        ),
      );
      await api("/portal/claims", {
        method: "POST",
        body: JSON.stringify({ ...Object.fromEntries(form), vehicle: Object.keys(vehicle).length ? vehicle : null,
          additional_costs: additionalCosts, attachments }),
      });
      await onDone();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <form className="card manual-form" onSubmit={submit}>
      <h2>Manual claim submission</h2>
      <p>
        Enter your claim details. The AI Agent investigates your form and uploaded PDF or image evidence against the uploaded policy documents. Staff records the final decision.
      </p>
      {error && <div className="message error">{error}</div>}
      <div className="form-grid">
        <label className="form-field">
          Policy number
          <input name="policy_id" required maxLength={100} />
        </label>
        <label className="form-field">
          Vehicle registration
          <input name="registration" required maxLength={40} />
        </label>
        <label className="form-field">
          Claim type
          <select name="loss_type">
            {["glass", "repair", "theft", "third_party_property_damage", "third_party_injury", "injury", "other"].map((t) => (
              <option key={t} value={t}>{lossTypeLabel(t)}</option>
            ))}
          </select>
        </label>
        <label className="form-field">
          Date of loss
          <input
            type="date"
            name="loss_date"
            required
            max={new Date().toLocaleDateString("en-CA")}
          />
        </label>
        <label className="form-field">
          Claimed amount (LKR)
          <input
            type="number"
            name="gross_amount"
            min="0.01"
            step="0.01"
            required
          />
        </label>
      </div>
      <h3 className="section-gap">Vehicle details</h3>
      <div className="form-grid">
        <label className="form-field">Chassis number or VIN<input name="vehicle_chassis_number" maxLength={40} /></label>
        <label className="form-field">Make<input name="vehicle_make" maxLength={60} /></label>
        <label className="form-field">Model<input name="vehicle_model" maxLength={80} /></label>
        <label className="form-field">Manufacture year<input name="vehicle_year" type="number" min="1886" max="2100" /></label>
      </div>
      <label className="form-field">
        Incident description
        <textarea name="narrative" minLength={10} maxLength={5000} required />
      </label>
      <section className="section-gap">
        <h3>Additional costs</h3>
        <p>List any costs included in the claimed amount that are outside the repair estimate, valuation or third-party demand.</p>
        {additionalRows.map((id, index) => (
          <div key={id} className="form-grid section-gap">
            <label className="form-field">
              Cost {index + 1} (LKR)
              <input name="additional_amount" type="number" min="0.01" step="0.01" required />
            </label>
            <label className="form-field">
              Reason for cost {index + 1}
              <input name="additional_reason" maxLength={500} required />
            </label>
            <button type="button" className="text-button" onClick={() => setAdditionalRows(rows => rows.filter(row => row !== id))}>
              Remove cost {index + 1}
            </button>
          </div>
        ))}
        {additionalRows.length < 10 && (
          <button type="button" className="button secondary" onClick={() => setAdditionalRows(rows => [...rows, crypto.randomUUID()])}>Add cost</button>
        )}
      </section>
      <label className="form-field section-gap">
        Supporting documents (PDF, PNG or JPEG; 2 MB each)
        <small>For repair or glass damage, upload an itemised estimate with the garage name and address.</small>
        <input
          type="file"
          accept="application/pdf,image/png,image/jpeg"
          multiple
          onChange={(e) => setFiles(Array.from(e.target.files || []))}
        />
      </label>
      <button className="button primary section-gap" disabled={busy}>
        {busy ? "Submitting…" : "Submit claim for review"}
      </button>
    </form>
  );
}

function Accounts() {
  const [accounts, setAccounts] = useState<Actor[]>([]);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const refresh = () =>
    api<Actor[]>("/portal/accounts")
      .then(setAccounts)
      .catch((e) => setError(String(e)));
  useEffect(() => {
    void refresh();
  }, []);
  return (
    <main className="portal-main">
      <h1>Account management</h1>
      <p>
        Administrators create user, staff and admin accounts. Public
        registration always creates a user.
      </p>
      {error && <div className="message error">{error}</div>}
      <form
        className="card manual-form section-gap"
        onSubmit={async (e) => {
          e.preventDefault();
          const form = e.currentTarget;
          setBusy(true);
          setError("");
          try {
            await api("/portal/accounts", {
              method: "POST",
              body: JSON.stringify(Object.fromEntries(new FormData(form))),
            });
            form.reset();
            await refresh();
          } catch (e) {
            setError(String(e));
          } finally {
            setBusy(false);
          }
        }}
      >
        <h2>Create an account</h2>
        <div className="form-grid">
          <label className="form-field">
            Email
            <input type="email" name="email" required />
          </label>
          <label className="form-field">
            Initial password
            <input
              type="password"
              name="password"
              minLength={10}
              maxLength={128}
              required
            />
          </label>
          <label className="form-field">
            Access role
            <select name="role">
              <option>user</option>
              <option>staff</option>
              <option>admin</option>
            </select>
          </label>
        </div>
        <button className="button primary" disabled={busy}>
          Create account
        </button>
      </form>
      <section className="card section-gap">
        <table className="claims-table">
          <thead>
            <tr>
              <th>Email</th>
              <th>Role</th>
            </tr>
          </thead>
          <tbody>
            {accounts.map((a) => (
              <tr key={a.id}>
                <td>{a.email}</td>
                <td>{a.role}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </main>
  );
}
