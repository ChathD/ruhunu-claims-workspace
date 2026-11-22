import { useEffect, useState } from "react";
import { api } from "../api";
import PolicyDocumentPreview from "../components/PolicyDocumentPreview";

type Wording = { id: string; title: string; product: string; version: string; status: string; error: string | null; filename: string; file_hash: string };
type Catalog = { documents: Wording[] };

export default function PolicyDocumentsPage() {
  const [catalog, setCatalog] = useState<Catalog>({ documents: [] });
  const [title, setTitle] = useState("");
  const [product, setProduct] = useState("");
  const [version, setVersion] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [previewDocument, setPreviewDocument] = useState<Wording | null>(null);
  async function refresh() { setCatalog(await api<Catalog>("/policy-documents")); }
  useEffect(() => {
    const update = () => { void refresh().catch(e => setError(String(e))); };
    update();
    const timer = setInterval(update, 5000);
    return () => clearInterval(timer);
  }, []);
  async function act(action: () => Promise<unknown>, success: string) {
    setBusy(true); setError(""); setNotice("");
    try { await action(); await refresh(); setNotice(success); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  }
  return <div className="policy-documents">
    {error && <div role="alert" className="info-strip">{error}</div>}
    {notice && <div role="status" className="info-strip">{notice}</div>}
    <section className="card" style={{ padding: 24 }}>
      <h2>Upload product wording</h2>
      <p>Upload English policy terms and exclusions as a text-based PDF or UTF-8 TXT (up to 10 MB). Each upload creates a new version. Scanned pages need OCR before upload.</p>
      <form className="policy-document-form" onSubmit={e => {
        e.preventDefault();
        if (!file) return;
        void act(async () => {
          if (file.size > 10 * 1024 * 1024) throw new Error("File must be at most 10 MB");
          const content = await new Promise<string>((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(String(reader.result).split(",")[1]);
            reader.onerror = () => reject(new Error("Unable to read file"));
            reader.readAsDataURL(file);
          });
          await api("/policy-documents", { method: "POST", body: JSON.stringify({ title: title.trim(), product: product.trim(), version: version.trim(), filename: file.name,
            media_type: file.name.toLowerCase().endsWith(".pdf") ? "application/pdf" : "text/plain", content }) });
        }, "Document queued for extraction and local indexing.");
      }}>
        <div className="claim-summary">
          <label className="form-field">Document title<input required maxLength={150} value={title} onChange={e => setTitle(e.target.value)} /></label>
          <label className="form-field">Product identifier<input required maxLength={100} value={product} onChange={e => setProduct(e.target.value)} placeholder="e.g. MOTOR-COMPREHENSIVE" /></label>
          <label className="form-field">Wording version<input required maxLength={100} value={version} onChange={e => setVersion(e.target.value)} placeholder="e.g. 2026.1" /></label>
        </div>
        <label className="form-field">Policy wording file<input required type="file" accept=".pdf,.txt" onChange={e => setFile(e.target.files?.[0] || null)} /></label>
        <div className="policy-document-actions"><button className="button primary" disabled={busy || !file}>{busy ? "Working…" : "Upload wording"}</button></div>
      </form>
    </section>
    <section className="card section-gap" style={{ padding: 24 }}>
      <h2>Document versions</h2>
      <p>Indexing runs in the background. Synthetic policies have their own wording versions. Coverage searches the version linked to the claim’s policy.</p>
      {catalog.documents.length === 0 && <p>No wording uploaded yet.</p>}
      {catalog.documents.map(d => <article className="finding" key={d.id}>
        <div className="finding-heading"><h3>{d.title}</h3><span className="badge">{d.status}</span></div>
        <p>{d.product} - Version {d.version} - {d.filename}</p>
        {d.error && <p role="alert">{d.error}</p>}
        <div className="policy-document-actions">
        <button type="button" className="button primary" onClick={() => setPreviewDocument(d)}>View document</button>
        <a className="button secondary" href={`/api/policy-documents/${d.id}/original`}>Download original</a>
        {d.status === "failed" && <button className="button secondary" disabled={busy} onClick={() => void act(() => api(`/policy-documents/${d.id}/retry`, { method: "POST" }), "Indexing queued again.")}>Retry indexing</button>}
        </div>
      </article>)}
    </section>
    {previewDocument && <PolicyDocumentPreview key={previewDocument.id} document={previewDocument} onClose={() => setPreviewDocument(null)} />}
  </div>;
}
