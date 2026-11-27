import { useEffect, useRef, useState } from "react";

export default function PolicyDocumentPreview({ document, onClose }: {
  document: { id: string; title: string; filename: string };
  onClose: () => void;
}) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const [preview, setPreview] = useState<{ url?: string; text?: string } | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    const dialog = dialogRef.current!;
    dialog.showModal();
    return () => dialog.close();
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    let objectUrl: string | undefined;
    async function load() {
      try {
        // Fetch the existing original-file route as a blob; its attachment header
        // does not trigger a download when the response is consumed by fetch.
        const response = await fetch(`/api/policy-documents/${document.id}/original`, { signal: controller.signal });
        if (!response.ok) {
          if (response.status === 401) throw new Error("Please sign in again to view this document.");
          if (response.status === 403) throw new Error("Your account cannot view this policy document.");
          if (response.status === 404) throw new Error("This document could not be found. Close the preview and refresh the document list.");
          throw new Error(`Unable to load this document (HTTP ${response.status}).`);
        }
        const blob = await response.blob();
        if (controller.signal.aborted) return;
        if (blob.type.startsWith("application/pdf")) {
          objectUrl = URL.createObjectURL(blob);
          setPreview({ url: objectUrl });
        } else if (blob.type.startsWith("text/plain")) {
          const text = await blob.text();
          if (!controller.signal.aborted) setPreview({ text });
        } else throw new Error("This file type cannot be previewed.");
      } catch (e) {
        if (!controller.signal.aborted) setError(e instanceof Error ? e.message : "Unable to load document.");
      }
    }
    void load();
    return () => { controller.abort(); if (objectUrl) URL.revokeObjectURL(objectUrl); };
  }, [document.id]);

  return <dialog ref={dialogRef} className="claim-detail-dialog policy-document-preview" aria-labelledby="policy-preview-title"
    onCancel={e => { e.preventDefault(); onClose(); }}>
    <header className="claim-dialog-header">
      <h2 id="policy-preview-title">{document.title}</h2>
      <button type="button" autoFocus onClick={onClose}>Close</button>
    </header>
    <div className="policy-preview-body">
      {error ? <p role="alert">{error}</p> : !preview ? <p role="status">Loading document…</p> : preview.url ?
        <iframe src={preview.url} title={`${document.title} PDF preview`} onError={() => setError("The PDF preview could not load. You can download the original below.")} /> :
        <pre className="policy-preview-text">{preview.text}</pre>}
    </div>
    <footer className="policy-preview-footer actions">
      {preview?.url && <a href={`/api/policy-documents/${document.id}/preview`} target="_blank" rel="noreferrer">Open PDF in new tab</a>}
      <a href={`/api/policy-documents/${document.id}/original`}>Download original</a>
    </footer>
  </dialog>;
}
