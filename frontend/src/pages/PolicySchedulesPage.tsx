import { useEffect, useState } from "react";
import { api } from "../api";
import { coverLabels, scheduleDate, type SavedPolicySchedule, type ScheduleSummary } from "../components/ScheduleDetails";
import { PolicyScheduleLink } from "./PolicyScheduleView";

type WordingOption = ScheduleSummary["wording"];
type ScheduleCorrection = {
  id: string;
  wording: WordingOption;
  insured_name: string;
  registration: string;
  conflicts: string[];
  options: { id: string; label: string; effect: string; references: { passage: string; page: number }[] }[];
};

export default function PolicySchedulesPage() {
  const [wording, setWording] = useState<WordingOption[]>([]);
  const [schedules, setSchedules] = useState<(ScheduleSummary & { registration: string })[] | null>(null);
  const [latest, setLatest] = useState<ScheduleSummary | null>(null);
  const [corrections, setCorrections] = useState<ScheduleCorrection[]>([]);
  const [busy, setBusy] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    const options = { signal: controller.signal };
    void Promise.all([
      api<{ documents: WordingOption[] }>("/policy-documents?selectable=true", options),
      api<{ schedules: NonNullable<typeof schedules>; corrections: ScheduleCorrection[] }>("/policy-schedules", options),
    ]).then(([documents, saved]) => {
      if (!controller.signal.aborted) {
        setWording(documents.documents);
        setSchedules(saved.schedules);
        setCorrections(saved.corrections);
      }
    }).catch(failure => { if (!controller.signal.aborted) setError(String(failure)); });
    return () => controller.abort();
  }, []);

  function showSaved(saved: SavedPolicySchedule[]) {
    if (!saved.length) return;
    const newIds = new Set(saved.map(schedule => schedule.id));
    setSchedules(previous => [...saved.map(schedule => ({ ...schedule, registration: schedule.vehicle.registration })),
      ...(previous ?? []).filter(schedule => !newIds.has(schedule.id))]);
    setLatest(saved[0]);
  }

  async function generate(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (generating || busy) return;
    const form = new FormData(event.currentTarget);
    setError("");
    setGenerating(true);
    try {
      const result = await api<{ schedules: SavedPolicySchedule[]; corrections: ScheduleCorrection[] }>("/policy-schedules", {
        method: "POST", body: JSON.stringify({ policy_document_id: form.get("wording") }),
      });
      setCorrections(result.corrections);
      showSaved(result.schedules);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure));
    } finally {
      setGenerating(false);
    }
  }

  async function applyCorrection(identifier: string, optionId: string) {
    setError("");
    setBusy(true);
    try {
      const schedule = await api<SavedPolicySchedule>(`/policy-schedules/corrections/${identifier}`, {
        method: "POST", body: JSON.stringify({ option_id: optionId }),
      });
      setCorrections(current => current.filter(correction => correction.id !== identifier));
      showSaved([schedule]);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure));
    } finally {
      setBusy(false);
    }
  }

  return <main>
    <h1>Policy schedules</h1>
    {error && <p role="alert">{error}</p>}
    <form className="card policy-picker" onSubmit={generate}>
      <h2>Generate a policy schedule</h2>
      <fieldset disabled={generating || busy || !wording.length}>
        <label>Policy PDF<select name="wording" required defaultValue="">
          <option value="">{wording.length ? "Select indexed policy PDF" : schedules === null ? "Loading…" : "No indexed policy PDFs"}</option>
          {wording.map(option => <option key={option.id} value={option.id}>{option.title} · {option.product} · {option.version}
            {option.document_kind === "product_summary" ? " · Product summary" : ""}</option>)}
        </select></label>
        <div className="actions">
          <button className="primary">Generate schedule</button>
        </div>
      </fieldset>
    </form>
    {generating && <p role="status">Generating a schedule and checking it against the wording…</p>}
    {latest && <div><span role="status">Schedule saved. </span>
      <PolicyScheduleLink identifier={latest.id}>Open {latest.policy_number}</PolicyScheduleLink>
    </div>}
    {corrections.map(correction => <section key={correction.id} className="card" aria-labelledby={`correction-${correction.id}`}>
      <h2 id={`correction-${correction.id}`}>Choose a wording correction</h2>
      <p>{correction.insured_name} · {correction.registration} · {correction.wording.title}</p>
      <ul>{correction.conflicts.map(conflict => <li key={conflict}>{conflict}</li>)}</ul>
      {correction.options.map(option => <article className="finding" key={option.id}>
        <h3>{option.label}</h3>
        <p>{option.effect}</p>
        <div className="actions">
          <button type="button" disabled={busy || generating} aria-label={`Apply ${option.label}`} onClick={() => void applyCorrection(correction.id, option.id)}>Apply correction</button>
          {option.references.map(reference => <a key={reference.passage} target="_blank" rel="noreferrer"
            href={`/api/policy-documents/${correction.wording.id}/preview#page=${reference.page}`}>{reference.passage} · Page {reference.page}</a>)}
        </div>
      </article>)}
    </section>)}
    {schedules !== null && <section className="card">
      <h2>Saved schedules</h2>
      {schedules.length ? <div className="table-scroll"><table>
        <thead><tr><th>Policy number</th><th>Policyholder</th><th>Vehicle</th><th>Cover</th><th>Period</th></tr></thead>
        <tbody>{schedules.map(schedule => <tr key={schedule.id}>
          <td><PolicyScheduleLink identifier={schedule.id}>{schedule.policy_number}</PolicyScheduleLink></td>
          <td>{schedule.insured_name}</td><td>{schedule.registration}</td><td>{coverLabels[schedule.cover_type]}
            {schedule.wording.document_kind === "product_summary" ? " · Provisional" : ""}</td>
          <td>{scheduleDate(schedule.starts_on)} to {scheduleDate(schedule.ends_on)}</td>
        </tr>)}</tbody>
      </table></div> : <p>No policy schedules yet.</p>}
    </section>}
  </main>;
}
