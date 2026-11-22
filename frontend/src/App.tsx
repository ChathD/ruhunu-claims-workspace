import ClaimWorkspace from "./pages/ClaimWorkspace";
import EvaluationPage from "./pages/EvaluationPage";
import GeneratorPage from "./pages/GeneratorPage";
import type { SyntheticPreview } from "./pages/GeneratorPage";
import OverviewPage from "./pages/OverviewPage";
import PeoplePoliciesPage from "./pages/PeoplePoliciesPage";
import PolicyDocumentsPage from "./pages/PolicyDocumentsPage";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  Activity,
  ArrowDownToLine,
  ArrowRight,
  Bell,
  Check,
  CheckCheck,
  CircleHelp,
  Clock3,
  FileCheck2,
  FileText,
  FlaskConical,
  Gauge,
  LayoutGrid,
  LoaderCircle,
  Menu,
  Plus,
  Settings2,
  ShieldCheck,
  ShieldQuestion,
  Users,
  X,
} from "lucide-react";
import { api } from "./api";
import {
  AuditLink,
  Badge,
  ClaimTable,
  SearchField,
  lossTypeLabel,
  money,
  stageNames,
  time,
} from "./components";
import type {
  ActivityEvent,
  Claim,
  Detail,
  Evaluation,
  Overview,
  Page,
} from "./types";
import { routeLabels, routes } from "./types";

const navigation = [
  {
    section: "WORKSPACE",
    items: [
      { name: "Overview", icon: LayoutGrid },
      { name: "Claims", icon: FileText },
      { name: "Human review", icon: ShieldCheck },
    ],
  },
  {
    section: "OPERATIONS",
    items: [
      { name: "People & policies", icon: Users },
      { name: "Synthetic generator", icon: FlaskConical },
      { name: "Agent activity", icon: Activity },
    ],
  },
  {
    section: "ASSURANCE",
    items: [
      { name: "Evaluation", icon: Gauge },
      { name: "Audit trail", icon: FileCheck2 },
    ],
  },
  { section: "SYSTEM", items: [{ name: "Policy documents", icon: FileText }, { name: "Settings", icon: Settings2 }] },
];

export default function App({ initialPage = "Overview" }: { initialPage?: Page }) {
  const [page, setPage] = useState<Page>(initialPage);
  const [overview, setOverview] = useState<Overview | null>(null);
  const [claims, setClaims] = useState<Claim[]>([]);
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null);
  const [events, setEvents] = useState<ActivityEvent[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<Detail | null>(null);
  const [tab, setTab] = useState("Findings");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [search, setSearch] = useState("");
  const [routeFilter, setRouteFilter] = useState("all");
  const [chartMode, setChartMode] = useState("outcomes");
  const [generating, setGenerating] = useState(false);
  const [generationError, setGenerationError] = useState("");
  const [preview, setPreview] = useState<SyntheticPreview | null>(null);
  const [adding, setAdding] = useState(false);
  const [ackClaim, setAckClaim] = useState<Claim | null>(null);
  const [reviewer, setReviewer] = useState("");
  const [ackBusy, setAckBusy] = useState(false);
  const [sidebar, setSidebar] = useState(false);
  const selectedRef = useRef<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [summary, list, matrix, activity] = await Promise.all([
        api<Overview>("/overview"),
        api<{ claims: Claim[] }>("/claims"),
        api<Evaluation>("/evaluation"),
        api<{ events: ActivityEvent[] }>("/activity"),
      ]);
      setOverview(summary);
      setClaims(list.claims);
      setEvaluation(matrix);
      setEvents(activity.events);
      setError("");
      if (selectedRef.current)
        setDetail(await api<Detail>(`/claims/${selectedRef.current}`));
    } catch (e) {
      setError(
        e instanceof Error ? e.message : "Cannot connect to the backend.",
      );
    }
  }, []);

  useEffect(() => {
    void refresh();
    let timer: ReturnType<typeof setTimeout> | undefined;
    const stream = new EventSource("/api/events");
    stream.onmessage = () => {
      if (!timer)
        timer = setTimeout(() => {
          timer = undefined;
          void refresh();
        }, 350);
    };
    const poll = setInterval(() => {
      void refresh();
    }, 10000);
    return () => {
      stream.close();
      clearInterval(poll);
      if (timer) clearTimeout(timer);
    };
  }, [refresh]);

  useEffect(() => {
    if (notice) {
      const timer = setTimeout(() => setNotice(""), 6000);
      return () => clearTimeout(timer);
    }
  }, [notice]);

  const navigate = (name: Page) => {
    setPage(name);
    setSelected(null);
    selectedRef.current = null;
    setDetail(null);
    setSidebar(false);
    setSearch("");
    setRouteFilter("all");
  };
  const openClaim = async (id: string) => {
    setSelected(id);
    selectedRef.current = id;
    setDetail(null);
    setTab("Findings");
    try {
      setDetail(await api<Detail>(`/claims/${id}`));
    } catch (e) {
      setError(String(e));
    }
  };
  const generate = async (policyId: string, count: number) => {
    setGenerationError("");
    setGenerating(true);
    try {
      const result = await api<SyntheticPreview>("/generator/preview", {
        method: "POST",
        body: JSON.stringify({ policy_id: policyId, count }),
      });
      setPreview(result);
      setPage("Synthetic generator");
      setNotice(`${count} synthetic claims are ready to review. Nothing has been added to Claims yet.`);
    } catch (e) {
      setGenerationError(e instanceof Error ? e.message : String(e));
    } finally {
      setGenerating(false);
    }
  };
  const addToClaims = async () => {
    if (!preview) return;
    setGenerationError("");
    setAdding(true);
    try {
      await api(`/generator/preview/${preview.batch_id}/commit`, { method: "POST" });
      setNotice(`${preview.count} claims added. The AI Agent is starting.`);
      setPreview(null);
      await refresh();
    } catch (e) {
      setGenerationError(e instanceof Error ? e.message : String(e));
    } finally {
      setAdding(false);
    }
  };
  const discardPreview = async () => {
    if (!preview) return;
    setGenerationError("");
    setAdding(true);
    try {
      await api<void>(`/generator/preview/${preview.batch_id}`, { method: "DELETE" });
      setPreview(null);
    } catch (e) {
      setGenerationError(e instanceof Error ? e.message : String(e));
    } finally {
      setAdding(false);
    }
  };
  const acknowledge = async () => {
    if (!ackClaim?.task || !reviewer.trim()) return;
    setAckBusy(true);
    try {
      await api(`/tasks/${ackClaim.task.id}/acknowledge`, {
        method: "POST",
        body: JSON.stringify({ reviewer }),
      });
      setAckClaim(null);
      setReviewer("");
      setNotice(
        "Acknowledgement recorded. The claim has not been adjudicated or paid.",
      );
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setAckBusy(false);
    }
  };
  const exportCsv = () => {
    const content = [
      "Reference,Type,Gross LKR,Route,Status",
      ...claims.map(
        (c) =>
          `${c.reference},${c.type},${c.gross_amount},${c.route || ""},${c.status}`,
      ),
    ].join("\n");
    const url = URL.createObjectURL(new Blob([content], { type: "text/csv" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = "claims-register.csv";
    a.click();
    URL.revokeObjectURL(url);
  };
  const matchingClaims = claims.filter(
    (c) => !search || `${c.reference} ${c.registration}`.toLowerCase().includes(search.toLowerCase()),
  );
  const filtered = matchingClaims.filter((c) => routeFilter === "all" || c.route === routeFilter);
  const humanClaims = matchingClaims.filter((c) => c.task || c.processing_alerts?.length);
  const displayName = selected ? detail?.reference || "Claim workspace" : page;

  return (
    <div className="app-shell">
      {sidebar && (
        <div className="sidebar-scrim" onClick={() => setSidebar(false)} />
      )}
      <aside className={`sidebar ${sidebar ? "is-open" : ""}`}>
        <button className="brand" onClick={() => navigate("Overview")}>
          <span className="brand-mark">
            r<span>·</span>
          </span>
          <span>
            Ruhunu<small>Claims workspace</small>
          </span>
        </button>
        <nav>
          {navigation.map((group) => (
            <div className="nav-group" key={group.section}>
              <div className="nav-label">{group.section}</div>
              {group.items.map((item) => (
                <button
                  key={item.name}
                  className={`nav-item ${page === item.name ? "active" : ""}`}
                  onClick={() => navigate(item.name as Page)}
                >
                  <item.icon size={17} strokeWidth={1.7} />
                  <span>{item.name}</span>
                  {item.name === "Human review" &&
                    !!overview?.human_pending && (
                      <span className="nav-count">
                        {overview.human_pending}
                      </span>
                    )}
                </button>
              ))}
            </div>
          ))}
        </nav>
        <div className="sidebar-footer">
          <div className="environment">
            <i />
            Synthetic workspace
          </div>
          <p>Provisional rules · No payments</p>
          <div className="workspace-user">
            <span className="avatar">RC</span>
            <div>
              Ruhunu Claims<small>Local development</small>
            </div>
          </div>
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <div className="breadcrumb">
            <button
              className="icon-button mobile-menu"
              aria-label="Open navigation"
              onClick={() => setSidebar(true)}
            >
              <Menu size={19} />
            </button>
            <span>Ruhunu</span>
            <span className="slash">/</span>
            <strong>{displayName}</strong>
          </div>
          <div className="topbar-right">
            <span className="workspace-pill">
              <i /> Sandbox
            </span>
            <button
              className="icon-button"
              title="Agent activity"
              aria-label="View agent activity"
              onClick={() => navigate("Agent activity")}
            >
              <Bell size={17} />
            </button>
            <button
              className="icon-button"
              title="Workspace information"
              aria-label="Workspace information"
              onClick={() => navigate("Settings")}
            >
              <CircleHelp size={17} />
            </button>
            <span className="avatar small">RC</span>
          </div>
        </header>
        <main>
          {error && (
            <div className="message error" role="alert">
              {error}{" "}
              <button className="text-button" onClick={() => void refresh()}>
                Try again
              </button>
            </div>
          )}
          {notice && (
            <div className="toast" role="status">
              <Check size={17} />
              {notice}
              <button
                className="icon-button"
                onClick={() => setNotice("")}
                aria-label="Dismiss notification"
              >
                <X size={15} />
              </button>
            </div>
          )}
          {!selected && (
            <div className="page-heading">
              <div>
                <div className="eyebrow">WORKSPACE / {page.toUpperCase()}</div>
                <h1>
                  {page === "Overview" ? "Claims dashboard overview" : page}
                </h1>
                <p>
                  {
                    (
                      {
                        Overview:
                          "A focused view of your claims, decisions, and the agents behind them.",
                        Claims:
                          "Every claim, its evidence, and the reasoning behind its route.",
                        "Human review":
                          "Claims waiting for an adjuster or fraud specialist to take a look.",
                        "Synthetic generator":
                          "Generate fresh motor claims using saved people and policies.",
                        "People & policies": "Create the fictional people and motor policies used by the claim generator.",
                        "Agent activity":
                          "Follow the work from submission to an evidence-backed decision.",
                        Evaluation:
                          "Expected routes against predicted routes, with every class visible.",
                        "Audit trail":
                          "Reconstruct what the system saw, which rules applied, and what it decided.",
                        Settings:
                          "The configuration and boundaries of this local workspace.",
                        "Policy documents": "Upload product wording and link the exact version used for policy investigation.",
                      } as Record<Page, string>
                    )[page]
                  }
                </p>
              </div>
              <div className="heading-actions">
                {page === "Claims" && (
                  <button className="button secondary" onClick={exportCsv}>
                    <ArrowDownToLine size={15} />
                    Export CSV
                  </button>
                )}
                {page !== "People & policies" && <button
                  className="button primary"
                  onClick={() => navigate("Synthetic generator")}
                >
                  <Plus size={16} />
                  Generate claims
                </button>}
              </div>
            </div>
          )}

          {!selected && page === "Overview" && (
            <OverviewPage
              overview={overview}
              chartMode={chartMode}
              setChartMode={setChartMode}
              navigate={navigate}
              openClaim={openClaim}
            />
          )}

          {!selected && page === "Claims" && (
            <section className="card">
              <div className="filter-bar">
                <SearchField value={search} onChange={setSearch} />
                <select
                  aria-label="Filter by route"
                  value={routeFilter}
                  onChange={(e) => setRouteFilter(e.target.value)}
                >
                  <option value="all">All outcomes</option>
                  {routes.map((r) => (
                    <option key={r} value={r}>
                      {routeLabels[r]}
                    </option>
                  ))}
                </select>
                <span className="muted">{filtered.length} claims</span>
              </div>
              <ClaimTable claims={filtered} open={(id) => void openClaim(id)} />
            </section>
          )}

          {!selected && page === "Human review" && (
            <>
              <div className="info-strip">
                <ShieldQuestion size={18} />
                <span>
                  Every fraud indicator and claim above LKR 250,000 reaches a
                  human. Acknowledgement records receipt; it does not settle the
                  claim.
                </span>
              </div>
              <div className="filter-bar standalone">
                <SearchField value={search} onChange={setSearch} />
                <span className="muted">
                  {
                    humanClaims.filter((c) => c.task?.status === "pending")
                      .length
                  }{" "}
                  awaiting acknowledgement
                </span>
              </div>
              <div className="review-grid">
                {humanClaims.map((c) => (
                  <section className="card review-card" key={c.id}>
                    <div className="review-top">
                      <span
                        className={`queue-icon ${c.task?.queue === "fraud" ? "red" : "amber"}`}
                      >
                        <ShieldQuestion size={21} />
                      </span>
                      <Badge route={c.route} status={c.status} />
                    </div>
                    <h2>{c.reference}</h2>
                    <p>
                      {lossTypeLabel(c.type)} ·{" "}
                      {c.registration}
                    </p>
                    <div className="review-amount">
                      LKR {money(c.gross_amount)}
                    </div>
                    <div className="review-status">
                      {c.processing_alerts?.length ? (
                        <>
                          <Clock3 size={15} />
                          <span>{c.processing_alerts.join(" · ")} · Reprocessing required</span>
                        </>
                      ) : c.task?.status === "acknowledged" ? (
                        <>
                          <CheckCheck size={15} />
                          <span>Acknowledged by {c.task.reviewer}</span>
                        </>
                      ) : (
                        <>
                          <Clock3 size={15} />
                          <span>
                            Waiting for{" "}
                            {c.task?.queue === "fraud"
                              ? "fraud specialist"
                              : "adjuster"}
                          </span>
                        </>
                      )}
                    </div>
                    <div className="review-actions">
                      <button
                        className="button secondary"
                        onClick={() => void openClaim(c.id)}
                      >
                        Review evidence
                      </button>
                      {c.task?.status === "pending" && (
                        <button
                          className="button primary"
                          onClick={() => {
                            setAckClaim(c);
                            setReviewer("");
                          }}
                        >
                          Acknowledge
                        </button>
                      )}
                    </div>
                  </section>
                ))}
              </div>
              {humanClaims.length === 0 && (
                <div className="card empty-state">
                  No human-review tasks match your search.
                </div>
              )}
            </>
          )}

          {!selected && page === "People & policies" && <PeoplePoliciesPage openClaims={() => setPage("Synthetic generator")} />}

          {!selected && page === "Synthetic generator" && (
            <GeneratorPage
              generating={generating}
              error={generationError}
              generate={generate}
              preview={preview}
              addToClaims={addToClaims}
              adding={adding}
              openPeoplePolicies={() => setPage("People & policies")}
              discardPreview={discardPreview}
            />
          )}

          {!selected && page === "Agent activity" && (
            <section className="card">
              <div className="card-heading">
                <div>
                  <h2>Workflow event stream</h2>
                  <p>Persisted events, updated as the agents work.</p>
                </div>
                <span className="live-label">
                  <i />
                  Live
                </span>
              </div>
              <div className="activity-list">
                {events.map((e) => (
                  <button
                    className="activity-row"
                    key={e.id}
                    onClick={() => void openClaim(e.assessment_id)}
                  >
                    <span
                      className={`activity-icon ${e.kind === "decision_published" ? "green" : ""}`}
                    >
                      {e.kind === "decision_published" ? (
                        <Check size={16} />
                      ) : (
                        <Activity size={16} />
                      )}
                    </span>
                    <span>
                      <strong>{e.kind.replaceAll("_", " ")}</strong>
                      <small>
                        {e.payload.stage
                          ? stageNames[e.payload.stage]
                          : e.payload.route ||
                            `Assessment ${e.assessment_id.slice(0, 8)}`}
                      </small>
                    </span>
                    <span className="activity-time">{time(e.created_at)}</span>
                    <ArrowRight size={14} />
                  </button>
                ))}
              </div>
            </section>
          )}

          {!selected && page === "Evaluation" && (
            <EvaluationPage evaluation={evaluation} openClaim={openClaim} />
          )}

          {!selected && page === "Audit trail" && (
            <>
              <div className="info-strip">
                <FileCheck2 size={18} />
                <span>
                  Export original evidence, retrieval responses, context
                  versions, findings and the exact backend source used for an
                  assessment.
                </span>
              </div>
              <section className="card">
                <div className="filter-bar">
                  <SearchField value={search} onChange={setSearch} />
                  <span className="muted">Completed assessments</span>
                </div>
                <div className="audit-list">
                  {filtered
                    .filter((c) => c.route)
                    .map((c) => (
                      <div className="audit-row" key={c.id}>
                        <span className="file-icon">
                          <FileCheck2 size={19} />
                        </span>
                        <div>
                          <button
                            className="text-button claim-link"
                            onClick={() => void openClaim(c.id)}
                          >
                            {c.reference}
                          </button>
                          <small>Evidence + findings + source manifest</small>
                        </div>
                        <Badge route={c.route} />
                        <AuditLink id={c.id}>Download JSON</AuditLink>
                      </div>
                    ))}
                </div>
              </section>
              <p className="page-note">
                Local hashes detect inconsistent edits. Enforced six-month
                retention, signed manifests and immutable storage remain
                deployment work.
              </p>
            </>
          )}

          {!selected && page === "Settings" && (
            <>
              <section className="card">
                <div className="card-heading">
                  <div>
                    <h2>Workspace configuration</h2>
                    <p>
                      Current development settings and the production decisions
                      still to confirm.
                    </p>
                  </div>
                </div>
                <div className="settings-list">
                  {[
                    ["Environment", "Synthetic motor claims only"],
                    [
                      "Execution",
                      "Python checks through LangChain tools · LangGraph orchestration",
                    ],
                    [
                      "Automated authority gate",
                      "Gross amount greater than LKR 250,000 requires a human",
                    ],
                    [
                      "Business rules",
                      "Provisional · client manual and authority rules pending",
                    ],
                    [
                      "Persistence",
                      "Local SQLite + durable LangGraph checkpoints",
                    ],
                    ["Model calls", `${overview?.model_calls ?? 0} recorded · AI Agent when configured`],
                    [
                      "Langfuse",
                      "Opt-in metadata spans · disabled unless explicitly configured",
                    ],
                    [
                      "Identity",
                      "Signed-in user, staff and administrator roles",
                    ],
                    [
                      "Documents",
                      "Policy wording PDF/TXT search · scanned-image OCR pending",
                    ],
                    ["Deployment region", "Client confirmation required"],
                  ].map(([label, value]) => (
                    <div key={label}>
                      <span>{label}</span>
                      <strong>{value}</strong>
                    </div>
                  ))}
                </div>
              </section>
              <div className="info-strip section-gap">
                <CircleHelp size={18} />
                <span>
                  This is the first working local slice. PostgreSQL deployment,
                  real OCR and surge validation
                  follow after the workflow is verified.
                </span>
              </div>
            </>
          )}

          {!selected && page === "Policy documents" && <PolicyDocumentsPage />}
          {selected && (
            <ClaimWorkspace
              detail={detail}
              page={page}
              tab={tab}
              setTab={setTab}
              onBack={() => {
                setSelected(null);
                selectedRef.current = null;
                setDetail(null);
              }}
              onAcknowledge={(claim) => {
                setAckClaim(claim);
                setReviewer("");
              }}
              onRetry={async () => {
                if (!detail) return;
                try {
                  const result = await api<{ id: string }>(`/claims/${detail.id}/retry`, { method: "POST" });
                  await refresh();
                  await openClaim(result.id);
                  setTab("Investigation");
                } catch (e) {
                  setError(String(e));
                }
              }}
              onInvestigate={async () => {
                if (!detail) return;
                try {
                  const result = await api<{ id: string }>(`/portal/claims/${detail.id}/investigate`, { method: "POST" });
                  await refresh();
                  await openClaim(result.id);
                  setTab("Investigation");
                } catch (e) { setError(String(e)); }
              }}
            />
          )}
        </main>
      </div>
      {ackClaim && (
        <div className="modal-backdrop">
          <section
            role="dialog"
            aria-modal="true"
            aria-labelledby="ack-title"
            className="modal"
          >
            <div className="modal-top">
              <ShieldCheck size={27} />
              <button
                className="icon-button"
                aria-label="Close acknowledgement"
                onClick={() => setAckClaim(null)}
              >
                <X size={19} />
              </button>
            </div>
            <h2 id="ack-title">Acknowledge {ackClaim.reference}</h2>
            <p>
              Record that you received this task. This local demonstration uses
              an entered name, not authenticated production identity.
            </p>
            <label className="form-field">
              Reviewer name
              <input
                autoFocus
                value={reviewer}
                maxLength={100}
                onChange={(e) => setReviewer(e.target.value)}
                placeholder="Enter your name"
              />
            </label>
            <div className="modal-actions">
              <button
                className="button secondary"
                onClick={() => setAckClaim(null)}
              >
                Cancel
              </button>
              <button
                className="button primary"
                disabled={!reviewer.trim() || ackBusy}
                onClick={() => void acknowledge()}
              >
                {ackBusy && <LoaderCircle size={16} className="spin" />}
                Acknowledge receipt
              </button>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}
