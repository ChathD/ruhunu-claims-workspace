import { useEffect, useState } from "react";
import { LoaderCircle, Plus } from "lucide-react";
import { api } from "../api";
import { insuranceTypeLabel, lossTypeLabel, money } from "../components";
import type { PersonPolicyRecord } from "../types";
import PolicyDocumentPreview from "../components/PolicyDocumentPreview";
import PolicyTerms from "../components/PolicyTerms";

const wordingStatus = {
  queued: "Queued for indexing",
  indexing: "Indexing",
  ready: "Ready for Coverage",
  failed: "Indexing failed",
};

export default function PeoplePoliciesPage({ openClaims }: { openClaims: () => void }) {
  const [records, setRecords] = useState<PersonPolicyRecord[] | null>(null);
  const [count, setCount] = useState(5);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState("");
  const [wordingPreview, setWordingPreview] = useState<PersonPolicyRecord["wording"]>(null);
  const wordingPending = records?.some(({ wording }) => wording?.status === "queued" || wording?.status === "indexing") ?? false;

  async function load() {
    const result = await api<{ records: PersonPolicyRecord[] }>("/generator/people-policies");
    setRecords(result.records);
  }

  useEffect(() => {
    void load().catch((failure: Error) => setError(failure.message));
  }, []);

  useEffect(() => {
    if (!wordingPending || generating) return;
    const timer = setInterval(() => {
      void load().catch((failure: Error) => setError(failure.message));
    }, 5000);
    return () => clearInterval(timer);
  }, [wordingPending, generating]);

  async function generate() {
    setGenerating(true);
    setError("");
    try {
      await api("/generator/people-policies", { method: "POST", body: JSON.stringify({ count }) });
      await load();
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure));
    } finally {
      setGenerating(false);
    }
  }

  return (
    <div className="generator-layout">
      <section className="card generator-panel">
        <h2>Generate people and policies</h2>
        <p>Claude creates fictional people, a motor policy for each, and a wording document for Coverage to search.</p>
        <form onSubmit={(event) => { event.preventDefault(); void generate(); }}>
          <label className="form-field">
            Number of people and policies
            <input type="number" min="1" max="160" required value={Number.isNaN(count) ? "" : count}
              onChange={(event) => setCount(event.target.valueAsNumber)} disabled={generating} />
          </label>
          <button className="button primary" disabled={generating || !Number.isInteger(count) || count < 1 || count > 160}>
            {generating ? <LoaderCircle size={16} className="spin" /> : <Plus size={16} />}
            {generating ? "Generating…" : "Generate and save"}
          </button>
        </form>
        {error && <div className="message error" role="alert">{error}</div>}
      </section>
      <section className="card preview-panel">
        <div className="card-heading">
          <h2>Saved policies</h2>
          {!!records?.length && <button className="button secondary" onClick={openClaims}>Generate claims</button>}
        </div>
        {records === null && !error && <p>Loading saved people and policies…</p>}
        {records?.length === 0 && <p>Generate a person and policy to start making claims.</p>}
        <div className="preview-claims">
          {records?.map(({ person, policy, wording }) => (
            <details className="preview-claim" key={policy.policy_id}>
              <summary>
                <span>{person.full_name}<small>{[person.locality, person.district].filter(Boolean).join(", ")} · {policy.registration}</small></span>
                <span className="badge">{wording ? `Wording · ${wordingStatus[wording.status]}` : "Structured terms only"}</span>
              </summary>
              <div className="preview-claim-body">
                <section className="preview-wide">
                  <dl className="preview-facts">
                    <div><dt>Person ID</dt><dd>{person.person_id}</dd></div>
                    <div><dt>Policy ID</dt><dd>{policy.policy_id}</dd></div>
                    <div><dt>Effective dates</dt><dd>{policy.starts_on} to before {policy.ends_before}</dd></div>
                    {policy.insurance_type && <div><dt>Insurance type</dt><dd>{insuranceTypeLabel(policy.insurance_type)}</dd></div>}
                    <div><dt>Covered losses</dt><dd>{policy.covered_losses.map(lossTypeLabel).join(", ")}</dd></div>
                    <div><dt>Limit</dt><dd>LKR {money(policy.limit)}</dd></div>
                    <div><dt>Deductible</dt><dd>LKR {money(policy.deductible)}</dd></div>
                  </dl>
                  <PolicyTerms policy={policy} />
                </section>
                <section className="preview-wide">
                  <h3>Policy wording</h3>
                  {wording ? <>
                    <p>Version {wording.version}</p>
                    {wording.error && <p role="alert">{wording.error}</p>}
                    <button type="button" className="button secondary" onClick={() => setWordingPreview(wording)}>
                      View generated wording
                    </button>
                  </> : <p>This policy was saved without a generated wording document. Coverage uses the structured policy terms.</p>}
                </section>
              </div>
            </details>
          ))}
        </div>
      </section>
      {wordingPreview && <PolicyDocumentPreview key={wordingPreview.id} document={wordingPreview} onClose={() => setWordingPreview(null)} />}
    </div>
  );
}
