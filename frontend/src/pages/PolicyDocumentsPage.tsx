import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import PolicyDocumentPreview from "../components/PolicyDocumentPreview";

type Wording = { id: string; title: string; product: string; version: string; status: string; error: string | null; filename: string; file_hash: string; requires_reindex: boolean };
type Catalog = { documents: Wording[] };

export default function PolicyDocumentsPage() {
  const [catalog, setCatalog] = useState<Catalog>({ documents: [] });
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [previewDocument, setPreviewDocument] = useState<Wording | null>(null);
  const catalogRequest = useRef(0);
  async function refresh() {
    const request = ++catalogRequest.current;
    const result = await api<Catalog>("/policy-documents");
    if (request === catalogRequest.current) setCatalog(result);
  }
  useEffect(() => {
    const update = () => { void refresh().catch(e => setError(String(e))); };
    update();
    const timer = setInterval(update, 5000);
    return () => { clearInterval(timer); catalogRequest.current++; };
  }, []);
  async function act(action: () => Promise<unknown>, success: string) {
    setBusy(true); setError(""); setNotice("");
    try { await action(); setNotice(success); await refresh(); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  }
  function removeDocument(document: Wording) {
    if (!window.confirm(`Remove “${document.title}” (version ${document.version}) from Policy wording? Existing claims can still open it.`)) return;
    void act(async () => {
      await api(`/policy-documents/${document.id}`, { method: "DELETE" });
      setCatalog(current => ({ documents: current.documents.filter(item => item.id !== document.id) }));
    }, "Document removed from Policy wording.");
  }
  return <div className="policy-documents">
    {error && <div role="alert" className="info-strip">{error}</div>}
    {notice && <div role="status" className="info-strip">{notice}</div>}
    <section className="card">
      <h2>Upload product wording</h2>
      <form className="policy-document-form" onSubmit={e => {
        e.preventDefault();
        if (!file) return;
        const form = e.currentTarget;
        const fields = new FormData(form);
        void act(async () => {
          if (file.size > 12 * 1024 * 1024) throw new Error("File must be at most 12 MiB");
          const content = await new Promise<string>((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(String(reader.result).split(",")[1]);
            reader.onerror = () => reject(new Error("Unable to read file"));
            reader.readAsDataURL(file);
          });
          await api("/policy-documents", {
            method: "POST",
            body: JSON.stringify({
              title: String(fields.get("title")).trim(),
              product: String(fields.get("product")).trim(),
              version: String(fields.get("version")).trim(),
              filename: file.name,
              media_type: "application/pdf",
              content,
            }),
          });
          form.reset();
          setFile(null);
        }, "Document queued for extraction and OpenAI indexing.");
      }}>
        <fieldset disabled={busy}>
          <div className="claim-summary">
            <label className="form-field">Document title<input name="title" required maxLength={150} /></label>
            <label className="form-field">Product identifier<input name="product" required maxLength={100} placeholder="e.g. MOTOR-COMPREHENSIVE" /></label>
            <label className="form-field">Wording version<input name="version" required maxLength={100} placeholder="e.g. 2026.1" /></label>
          </div>
          <label className="form-field">Policy PDF, up to 12 MiB<input required type="file" accept=".pdf,application/pdf" onChange={e => setFile(e.target.files?.[0] || null)} /></label>
          <div className="policy-document-actions"><button className="button primary" disabled={busy || !file}>{busy ? "Working…" : "Upload wording"}</button></div>
        </fieldset>
      </form>
    </section>
    <section className="card section-gap">
      <h2>Document versions</h2>
      {catalog.documents.length === 0 && <p>No wording uploaded yet.</p>}
      {catalog.documents.map(d => <article className="finding" key={d.id}>
        <div className="finding-heading"><h3>{d.title}</h3><span className="badge">{d.requires_reindex ? "Reindex required" : d.status}</span></div>
        <p>{d.product} - Version {d.version} - {d.filename}</p>
        {d.error && <p role="alert">{d.error}</p>}
        <div className="policy-document-actions">
          <button type="button" className="button primary" onClick={() => setPreviewDocument(d)}>View document</button>
          <a className="button secondary" href={`/api/policy-documents/${d.id}/original`}>Download original</a>
          {(d.requires_reindex || d.status === "failed") && <button className="button secondary" disabled={busy} onClick={() => void act(() => api(`/policy-documents/${d.id}/retry`, { method: "POST" }), "Indexing queued again.")}>{d.requires_reindex ? "Reindex with OpenAI" : "Retry indexing"}</button>}
          <button type="button" disabled={busy} onClick={() => removeDocument(d)}>Remove document</button>
        </div>
      </article>)}
    </section>
    {previewDocument && <PolicyDocumentPreview key={previewDocument.id} document={previewDocument} onClose={() => setPreviewDocument(null)} />}
  </div>;
}
