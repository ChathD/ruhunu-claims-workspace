import { useCallback, useEffect, useRef, useState } from "react";
import { ArrowLeft } from "lucide-react";
import { api } from "./api";
import { dayMonthYear, lossTypeLabel, money, vehicleMakeAndModel } from "./components";
import ClaimSources from "./components/ClaimSources";
import InvestigationReport from "./components/InvestigationReport";
import EvaluationPage from "./pages/EvaluationPage";
import PolicyDocumentsPage from "./pages/PolicyDocumentsPage";
import PolicyScheduleView, { PolicyScheduleLink, schedulePreviewPath } from "./pages/PolicyScheduleView";
import PolicySchedulesPage from "./pages/PolicySchedulesPage";
import StaffGenerator from "./pages/StaffGenerator";
import type { Route, VehicleDetails } from "./types";

type Actor = { id: string; email: string; role: "staff" | "admin" };
type Screen = "claims" | "generator" | "accounts" | "evaluation" | "wording" | "schedules";
type WorkspaceView = { screen: Screen; claimId: string | null; depth: number };
type ClaimStatus = Route | "awaiting processing" | "investigation failed" | "processing";
const claimStatusLabels: Partial<Record<ClaimStatus, string>> = {
  "awaiting processing": "Awaiting processing",
  "investigation failed": "Investigation failed",
  processing: "Processing",
};
type ClaimRow = {
  id: string;
  reference: string;
  source: string;
  status: ClaimStatus;
  submission: { loss_type: string; gross_amount: string };
};
type ClaimDetail = ClaimRow & {
  scenario_title: string | null;
  submission: {
    policy_id?: string;
    insured_name?: string;
    registration: string;
    vehicle?: VehicleDetails | null;
    loss_date: string | null;
    narrative: string;
    unstructured_inputs?: Record<string, string>;
    documents: { id: string; name: string }[];
  };
  attachments: { id: string; name: string }[];
  archived: boolean;
  can_reprocess: boolean;
  report_available: boolean;
  replacement_id: string | null;
  selected_policy: { id: string; title: string; version: string } | null;
  has_policy_schedule: boolean;
};

function scheduleFromLocation() {
  const params = new URLSearchParams(window.location.search);
  const claim = params.get("claim-schedule");
  const schedule = params.get("schedule");
  if (claim) return { path: schedulePreviewPath(claim, true), claimId: claim };
  return schedule ? { path: schedulePreviewPath(schedule, false), claimId: null } : null;
}

function workspaceView(): WorkspaceView {
  return window.history.state?.claimsWorkspace ?? { screen: "claims", claimId: null, depth: 0 };
}

export default function Portal() {
  const [actor, setActor] = useState<Actor | null | undefined>(undefined);
  const [view, setView] = useState(workspaceView);
  const [generatorVisited, setGeneratorVisited] = useState(() => workspaceView().screen === "generator");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const backPending = useRef(false);
  const schedule = scheduleFromLocation();
  const admin = actor?.role === "admin";
  const staff = actor?.role === "staff";
  const activeScreen = admin
    ? (view.screen === "evaluation" ? "evaluation" : "accounts")
    : staff && schedule ? "schedule" : view.screen;

  useEffect(() => { void api<Actor>("/auth/me").then(setActor, () => setActor(null)); }, []);
  useEffect(() => {
    function restoreView() {
      backPending.current = false;
      const previous = workspaceView();
      if (previous.screen === "generator") setGeneratorVisited(true);
      setView(previous);
    }
    window.addEventListener("popstate", restoreView);
    return () => window.removeEventListener("popstate", restoreView);
  }, []);

  function openScreen(screen: Screen, claimId: string | null = null, replace = false) {
    const current = workspaceView();
    if (current.depth !== view.depth || current.screen !== view.screen || current.claimId !== view.claimId) return;
    if (screen === activeScreen && claimId === view.claimId) return;
    const next = { screen, claimId, depth: replace ? view.depth : view.depth + 1 };
    const url = new URL(window.location.href);
    url.searchParams.delete("claim-schedule");
    url.searchParams.delete("schedule");
    const state = { ...window.history.state, claimsWorkspace: next };
    if (replace) window.history.replaceState(state, "", url);
    else window.history.pushState(state, "", url);
    if (screen === "generator") setGeneratorVisited(true);
    setView(next);
  }

  function goBack() {
    if (view.depth > 0) {
      if (backPending.current) return;
      backPending.current = true;
      window.history.back();
    } else if (schedule) openScreen(schedule.claimId ? "claims" : "schedules", schedule.claimId, true);
    else if (view.claimId) openScreen("claims", null, true);
  }

  function openClaim(id: string) {
    openScreen("claims", id);
  }

  async function signIn(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const body = JSON.stringify(Object.fromEntries(new FormData(event.currentTarget)));
    setError("");
    setBusy(true);
    try {
      const account = await api<Actor>("/auth/login", { method: "POST", body });
      setActor(account);
      const home: WorkspaceView = { screen: account.role === "admin" ? "accounts" : "claims", claimId: null, depth: 0 };
      window.history.replaceState({ ...window.history.state, claimsWorkspace: home }, "");
      setView(home);
    } catch (failure) {
      setError(String(failure));
    } finally {
      setBusy(false);
    }
  }

  async function signOut() {
    try {
      await api("/auth/logout", { method: "POST" });
      setActor(null);
      setGeneratorVisited(false);
      setError("");
    } catch (failure) {
      setError(String(failure));
    }
  }

  if (actor === undefined) return <main>Loading…</main>;
  if (!actor) return (
    <main className="login-panel">
      <h1>Ruhunu Claims</h1>
      <form className="card" onSubmit={signIn}>
        {error && <p role="alert">{error}</p>}
        <label>Email<input name="email" type="email" autoComplete="username" required /></label>
        <label>Password<input name="password" type="password" minLength={10} maxLength={128}
          autoComplete="current-password" required /></label>
        <div className="actions">
          <button className="primary" disabled={busy}>Sign in</button>
        </div>
      </form>
    </main>
  );

  return <>
    <header className="account-bar">
      <button type="button" className="workspace-back" aria-label="Back"
        disabled={!view.depth && !(staff && (schedule || view.claimId))} onClick={goBack}>
        <ArrowLeft size={18} aria-hidden="true" />
      </button>
      <strong>Ruhunu Claims</strong>
      <nav aria-label="Workspace" className="actions">
        {admin ? <>
          <button aria-current={activeScreen === "accounts" ? "page" : undefined} onClick={() => openScreen("accounts")}>Accounts</button>
          <button aria-current={activeScreen === "evaluation" ? "page" : undefined} onClick={() => openScreen("evaluation")}>Routing evaluation</button>
        </>
        : <>
          <button aria-current={activeScreen === "claims" ? "page" : undefined} onClick={() => openScreen("claims")}>Claims</button>
          <button aria-current={activeScreen === "schedules" ? "page" : undefined} onClick={() => openScreen("schedules")}>Policy schedule generator</button>
          <button aria-current={activeScreen === "generator" ? "page" : undefined} onClick={() => openScreen("generator")}>Claim generator</button>
          <button aria-current={activeScreen === "wording" ? "page" : undefined} onClick={() => openScreen("wording")}>Upload policy wording</button>
        </>}
      </nav>
      <div className="account-actions">
        <span>{actor.email}</span>
        <button type="button" onClick={() => void signOut()}>Sign out</button>
      </div>
    </header>
    {error && <p role="alert">{error}</p>}
    {activeScreen === "wording" && <main><h1>Policy wording</h1><PolicyDocumentsPage /></main>}
    {activeScreen === "accounts" && <Accounts />}
    {admin && activeScreen === "evaluation" && <EvaluationPage />}
    {activeScreen === "claims" && <ClaimsPortal selected={view.claimId} onOpen={openClaim} />}
    {activeScreen === "schedules" && <PolicySchedulesPage />}
    {activeScreen === "schedule" && schedule && <main><PolicyScheduleView path={schedule.path} /></main>}
    {staff && generatorVisited && <StaffGenerator hidden={activeScreen !== "generator"} openClaim={openClaim} openSchedules={() => openScreen("schedules")} />}
  </>;
}

function ClaimsPortal({ selected, onOpen }: { selected: string | null; onOpen: (id: string) => void }) {
  const [claims, setClaims] = useState<ClaimRow[] | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    if (selected) return;
    const controller = new AbortController();
    let timer: number | undefined;
    async function refresh() {
      try {
        const rows = await api<ClaimRow[]>("/portal/claims?summary=true", { signal: controller.signal });
        if (controller.signal.aborted) return;
        setClaims(rows);
        setError("");
      } catch (failure) {
        if (!controller.signal.aborted) setError(String(failure));
      } finally {
        if (!controller.signal.aborted) timer = window.setTimeout(refresh, 5000);
      }
    }
    void refresh();
    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [selected]);

  if (selected) return <SingleClaim key={selected} id={selected} onOpen={onOpen} />;
  return <main>
    <div className="actions page-heading">
      <h1>Claims</h1>
    </div>
    {error && <p role="alert">{error}</p>}
    {claims === null ? (!error && <p role="status">Loading claims…</p>) : claims.length ? <div className="table-scroll">
      <table>
        <thead><tr><th>Claim</th><th>Type</th><th>Amount (LKR)</th><th>Status</th><th>Source</th></tr></thead>
        <tbody>{claims.map(claim => <tr key={claim.id}>
          <td><button className="text-button" onClick={() => onOpen(claim.id)}>{claim.reference}</button></td>
          <td>{lossTypeLabel(claim.submission.loss_type)}</td>
          <td>{money(claim.submission.gross_amount)}</td><td>{claimStatusLabels[claim.status] ?? "Investigation complete"}</td><td>{claim.source}</td>
        </tr>)}</tbody>
      </table>
    </div> : <p>No claims yet.</p>}
  </main>;
}

function SingleClaim({ id, onOpen }: { id: string; onOpen: (id: string) => void }) {
  const [claim, setClaim] = useState<ClaimDetail | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const path = `/portal/claims/${id}`;
  const refresh = useCallback(async () => {
    try { setClaim(await api<ClaimDetail>(`${path}?overview=true`)); }
    catch (failure) { setError(String(failure)); }
  }, [path]);
  useEffect(() => {
    void refresh();
    const timer = setInterval(() => void refresh(), 5000);
    return () => clearInterval(timer);
  }, [refresh]);

  async function run(action: "retry" | "process") {
    setBusy(true);
    setError("");
    try {
      const endpoint = action === "process" ? `/claims/${id}/process` : `${path}/${action}`;
      const result = await api<{ id?: string }>(endpoint, { method: "POST" });
      if (result.id && result.id !== id) onOpen(result.id);
      else await refresh();
    } catch (failure) {
      setError(String(failure));
    } finally {
      setBusy(false);
    }
  }

  const vehicleName = vehicleMakeAndModel(claim?.submission.vehicle);

  return <main>
    {error && <p role="alert">{error}</p>}
    {!claim ? (!error && <p>Loading…</p>) : <>
      <h1>{claim.reference} · {claim.scenario_title || lossTypeLabel(claim.submission.loss_type)}</h1>
      <p>{claim.archived ? "Archived" : (claimStatusLabels[claim.status] ?? "Investigation complete")}
        {claim.scenario_title && ` · ${lossTypeLabel(claim.submission.loss_type)}`} · LKR {money(claim.submission.gross_amount)}</p>
      <dl className="facts">
        {claim.submission.insured_name && <><dt>Reported policyholder</dt><dd>{claim.submission.insured_name}</dd></>}
        {claim.submission.policy_id && <><dt>Submitted policy number</dt><dd>{claim.submission.policy_id}</dd></>}
        {claim.has_policy_schedule && <><dt>Motor policy schedule</dt><dd>
          <PolicyScheduleLink identifier={id} fromClaim>Open motor policy schedule</PolicyScheduleLink>
        </dd></>}
        {claim.selected_policy && !claim.has_policy_schedule && <><dt>Selected policy</dt><dd><a href={`/api/policy-documents/${claim.selected_policy.id}/preview`} target="_blank" rel="noreferrer">
          {claim.selected_policy.title} · {claim.selected_policy.version}</a></dd></>}
        <dt>Vehicle</dt><dd>{vehicleName || "Unavailable"}</dd><dt>Loss date</dt>
        <dd>{claim.submission.loss_date ? dayMonthYear(claim.submission.loss_date) : "Unavailable"}</dd></dl>
      {!claim.archived && claim.status === "awaiting processing" && <div className="actions section-gap">
        <button type="button" className="primary" disabled={busy} onClick={() => void run("process")}>Process claim</button>
      </div>}
      {!claim.submission.unstructured_inputs && <p>{claim.submission.narrative}</p>}
      <h2>Documents</h2>
      {!!claim.attachments.length && <ul>{claim.attachments.map(file => <li key={file.id}><a href={`/api/portal/attachments/${file.id}`}>{file.name}</a></li>)}</ul>}
      {claim.submission.unstructured_inputs ? <ClaimSources sources={claim.submission.unstructured_inputs} />
        : (claim.submission.documents ?? []).map(document => <details key={document.id}><summary>{document.name}</summary><pre>{JSON.stringify(document, null, 2)}</pre></details>)}
      {!claim.attachments.length && !claim.submission.unstructured_inputs && !claim.submission.documents?.length && <p>No documents.</p>}
      {(claim.report_available || claim.can_reprocess || claim.replacement_id) && <section>
        <h2>Investigation</h2>
        <div className="actions">
          {claim.report_available && <InvestigationReport path={`${path}/investigation.pdf`} />}
          {claim.can_reprocess && <button disabled={busy} onClick={() => void run("retry")}>Retry investigation</button>}
          {claim.replacement_id && <button onClick={() => onOpen(claim.replacement_id!)}>Open latest investigation</button>}
        </div>
      </section>}
    </>}
  </main>;
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
      <h1>Accounts</h1>
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
        <h2>Create a staff account</h2>
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
              autoComplete="new-password"
              minLength={10}
              maxLength={128}
              required
            />
          </label>
        </div>
        <button className="button primary" disabled={busy}>
          Create staff account
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
