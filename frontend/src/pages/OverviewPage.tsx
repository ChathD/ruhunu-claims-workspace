import { ArrowRight, ChevronDown, ShieldCheck } from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { ClaimTable, Metric, routeClass } from "../components";
import { routes, routeLabels } from "../types";
import type { Overview, Page } from "../types";

export default function OverviewPage({
  overview,
  chartMode,
  setChartMode,
  navigate,
  openClaim,
}: {
  overview: Overview | null;
  chartMode: string;
  setChartMode: (mode: string) => void;
  navigate: (page: Page) => void;
  openClaim: (id: string) => Promise<void>;
}) {
  return (
    <>
      <div className="metrics-grid">
        <Metric
          label="Claims received"
          value={String(overview?.total ?? "—")}
          note={`${overview?.completed ?? 0} triaged in this workspace`}
          accent
        />
        <Metric
          label="Awaiting human review"
          value={String(overview?.human_pending ?? "—")}
          note="Adjuster & fraud queues"
          accent
        />
        <Metric
          label="Triaged within 15 min"
          value={
            overview?.within_15_minutes == null
              ? "—"
              : `${overview.within_15_minutes}%`
          }
          note="Measured from submission"
          accent
        />
        <Metric
          label="Cost per claim"
          value="Not measured"
          note="No live model calls in this mode"
        />
      </div>
      <div className="dashboard-columns">
        <section className="card chart-card">
          <div className="card-heading">
            <div>
              <h2>Routing overview</h2>
              <p>
                {chartMode === "outcomes"
                  ? "How your claims move through triage."
                  : "Claims received over the last seven days."}
              </p>
            </div>
            <label className="select-wrap">
              <select
                aria-label="Chart view"
                value={chartMode}
                onChange={(e) => setChartMode(e.target.value)}
              >
                <option value="outcomes">By outcome</option>
                <option value="volume">Last 7 days</option>
              </select>
              <ChevronDown size={13} />
            </label>
          </div>
          <div className="chart">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                accessibilityLayer
                data={
                  chartMode === "outcomes"
                    ? routes.map((route) => ({
                        label: routeLabels[route],
                        claims: overview?.breakdown[route] ?? 0,
                      }))
                    : overview?.volume || []
                }
                margin={{ top: 18, left: -22, right: 8, bottom: 0 }}
              >
                <CartesianGrid
                  vertical={false}
                  stroke="#eee9df"
                  strokeDasharray="3 4"
                />
                <XAxis
                  dataKey="label"
                  axisLine={false}
                  tickLine={false}
                  tick={{ fill: "#807c73", fontSize: 11 }}
                  dy={10}
                />
                <YAxis
                  allowDecimals={false}
                  axisLine={false}
                  tickLine={false}
                  tick={{ fill: "#969084", fontSize: 11 }}
                />
                <Tooltip
                  cursor={{ fill: "#f7f3eb" }}
                  contentStyle={{
                    background: "#fffdf9",
                    border: "1px solid #e1dbd0",
                    borderRadius: 9,
                    fontSize: 12,
                  }}
                />
                <Bar
                  dataKey="claims"
                  name="Claims"
                  fill="#fa6b2c"
                  radius={[7, 7, 0, 0]}
                  maxBarSize={54}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="chart-footer">
            <strong>{overview?.completed ?? 0}</strong> claims triaged{" "}
            <span className="separator">·</span>
            <span className="green-text">
              {overview?.processing ?? 0} in progress
            </span>
            <button className="text-button" onClick={() => navigate("Claims")}>
              View all claims <ArrowRight size={13} />
            </button>
          </div>
        </section>
        <section className="card outcome-card">
          <div className="card-heading">
            <div>
              <h2>Triage outcomes</h2>
              <p>One clear next step for every claim.</p>
            </div>
            <span className="live-label">
              <i />
              Live
            </span>
          </div>
          <div className="outcome-total">
            {overview?.completed ?? 0}
            <span>claims triaged</span>
          </div>
          <div className="outcome-bar">
            {routes.map((route) => (
              <span
                key={route}
                className={routeClass(route)}
                style={{ flex: overview?.breakdown[route] || 0 }}
              />
            ))}
          </div>
          <div className="outcome-list">
            {routes.map((route) => (
              <div key={route}>
                <span>
                  <i className={`route-dot ${routeClass(route)}`} />
                  {routeLabels[route]}
                </span>
                <strong>{overview?.breakdown[route] ?? 0}</strong>
              </div>
            ))}
          </div>
          <div className="small-note">
            <ShieldCheck size={14} />
            Human authority enforced above LKR 250,000
          </div>
        </section>
      </div>
      <section className="card recent-card">
        <div className="card-heading">
          <div>
            <h2>Recent claims</h2>
            <p>The latest submissions and their routing outcomes.</p>
          </div>
          <button
            className="button secondary compact"
            onClick={() => navigate("Claims")}
          >
            View all claims <ArrowRight size={14} />
          </button>
        </div>
        <ClaimTable
          claims={overview?.recent_claims || []}
          open={(id) => void openClaim(id)}
        />
      </section>
      <div className="workspace-footnote">
        <span>
          <ShieldCheck size={13} />
          Synthetic data only. Every decision retains its evidence.
        </span>
        <button className="text-button" onClick={() => navigate("Audit trail")}>
          Explore audit trail <ArrowRight size={12} />
        </button>
      </div>
    </>
  );
}
