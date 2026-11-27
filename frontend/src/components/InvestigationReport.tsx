import { useEffect, useState } from "react";
import { api } from "../api";
import { PdfPages, PdfPreviewDialog, type PdfPreviewPage } from "./PdfPreview";

export default function InvestigationReport({ path }: { path: string }) {
  const [open, setOpen] = useState(false);
  return <>
    <button type="button" className="primary" aria-haspopup="dialog" onClick={() => setOpen(true)}>View investigation report</button>
    {open && <ReportPreview path={path} onClose={() => setOpen(false)} />}
  </>;
}

function ReportPreview({ path, onClose }: { path: string; onClose: () => void }) {
  const [pages, setPages] = useState<PdfPreviewPage[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    void api<{ pages: PdfPreviewPage[] }>(`${path}?preview=true`, { signal: controller.signal }).then(
      result => {
        if (controller.signal.aborted) return;
        if (Array.isArray(result?.pages) && result.pages.length) setPages(result.pages);
        else setError("The investigation report preview could not be loaded.");
      },
      failure => {
        if (!controller.signal.aborted) setError(failure instanceof Error ? failure.message : String(failure));
      },
    );
    return () => controller.abort();
  }, [path]);

  return <PdfPreviewDialog title="Investigation report" onClose={onClose}
    actions={pages.length > 0 && <a className="button primary" href={`/api${path}`}>Download PDF</a>}>
    {error && <p role="alert">{error}</p>}
    {!error && !pages.length && <p role="status">Loading investigation report…</p>}
    {pages.length > 0 && <PdfPages pages={pages} />}
  </PdfPreviewDialog>;
}
