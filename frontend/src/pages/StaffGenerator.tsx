import { useEffect, useState } from "react";
import { api } from "../api";
import { claimHeading, vehicleMakeAndModel } from "../components";
import ClaimSources from "../components/ClaimSources";
import type { VehicleDetails } from "../types";

type Preview = {
  batch_id: string;
  fixtures: { scenario: string; claim: { claim_id: string; insured_name: string; vehicle?: VehicleDetails | null; unstructured_inputs: Record<string, string> } }[];
};

export default function StaffGenerator({ hidden, openClaim, openSchedules }: { hidden: boolean; openClaim: (id: string) => void; openSchedules: () => void }) {
  const [preview, setPreview] = useState<Preview | null>(null);
  const [saved, setSaved] = useState<{ claimId: string; batchId: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [hasSchedules, setHasSchedules] = useState<boolean | null>(null);

  useEffect(() => {
    if (hidden) return;
    const controller = new AbortController();
    void api<{ schedules: { id: string }[] }>("/policy-schedules", { signal: controller.signal })
      .then(result => setHasSchedules(result.schedules.length > 0), failure => {
        if (!controller.signal.aborted) setError(String(failure));
      });
    return () => controller.abort();
  }, [hidden]);

  async function run(action: () => Promise<void>) {
    setBusy(true);
    setError("");
    try { await action(); }
    catch (failure) { setError(String(failure)); }
    finally { setBusy(false); }
  }

  return <main hidden={hidden}>
    <h1>Claim generator</h1>
    {error && <p role="alert">{error}</p>}
    {hasSchedules === false && <p className="actions">Generate a policy schedule to supply a policyholder and vehicle.
      <button type="button" onClick={openSchedules}>Open policy schedules</button>
    </p>}
    <div className="card">
      <button type="button" className="primary" disabled={busy || !!preview || hasSchedules !== true} onClick={() => void run(async () => {
        setPreview(await api<Preview>("/generator/preview", { method: "POST", body: "{}" }));
      })}>Generate claim</button>
    </div>
    {busy && <p role="status">Working…</p>}
    {preview && <section className="card">
      <h2>Generated claim</h2>
      {preview.fixtures.map(({ scenario, claim }) => <details key={claim.claim_id}>
        <summary>{claimHeading(scenario, claim.insured_name, vehicleMakeAndModel(claim.vehicle))}</summary>
        <ClaimSources sources={claim.unstructured_inputs} />
      </details>)}
      <div className="actions">
        <button className="primary" disabled={busy} onClick={() => void run(async () => {
          const { assessment_ids: [id] } = await api<{ assessment_ids: string[] }>(
            `/generator/preview/${preview.batch_id}/commit`, { method: "POST" });
          if (!id) throw new Error("The saved claim reference was unavailable.");
          setSaved({ claimId: id, batchId: preview.batch_id });
          setPreview(null);
        })}>Save claim</button>
        <button disabled={busy} onClick={() => void run(async () => {
          await api(`/generator/preview/${preview.batch_id}`, { method: "DELETE" });
          setPreview(null);
        })}>Discard</button>
      </div>
    </section>}
    {saved && <p role="status" className="actions">Claim saved.
      <button type="button" onClick={() => openClaim(saved.claimId)}>Open claim</button>
      <a href={`/api/generator/batches/${saved.batchId}/dataset.zip`} download>Download evaluation files</a>
    </p>}
  </main>;
}
