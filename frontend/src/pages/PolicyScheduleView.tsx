import { useEffect, useState, type ReactNode } from "react";
import { api } from "../api";
import { PdfPages, PdfPreviewDialog, type PdfPreviewPage } from "../components/PdfPreview";

type SchedulePreview = {
  title: string;
  wording: { id: string; title: string; version: string };
  pages: PdfPreviewPage[];
};

export function schedulePreviewPath(identifier: string, fromClaim: boolean) {
  const id = encodeURIComponent(identifier);
  return fromClaim ? `/portal/claims/${id}/policy-schedule` : `/policy-schedules/${id}`;
}

export function PolicyScheduleLink({ identifier, fromClaim = false, children }: {
  identifier: string;
  fromClaim?: boolean;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(false);
  const query = fromClaim ? "claim-schedule" : "schedule";

  return <>
    <a href={`?${query}=${encodeURIComponent(identifier)}`} aria-haspopup="dialog" onClick={event => {
      if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      setOpen(true);
    }}>{children}</a>
    {open && <PdfPreviewDialog title="Motor policy schedule" onClose={() => setOpen(false)}>
      <PolicyScheduleView path={schedulePreviewPath(identifier, fromClaim)} />
    </PdfPreviewDialog>}
  </>;
}

export default function PolicyScheduleView({ path }: { path: string }) {
  const [schedule, setSchedule] = useState<SchedulePreview | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    void api<SchedulePreview>(`${path}?preview=true`, { signal: controller.signal }).then(
      result => {
        if (controller.signal.aborted) return;
        if (Array.isArray(result?.pages) && result.pages.length) setSchedule(result);
        else setError("The policy schedule preview could not be loaded.");
      },
      failure => { if (!controller.signal.aborted) setError(failure instanceof Error ? failure.message : String(failure)); },
    );
    return () => controller.abort();
  }, [path]);

  return <div>
    {error && <p role="alert">{error}</p>}
    {!error && !schedule && <p role="status">Loading policy schedule…</p>}
    {schedule && <section className="schedule-viewer" aria-label={schedule.title}>
      <p><a href={`/api/policy-documents/${schedule.wording.id}/preview`} target="_blank" rel="noreferrer">
        {schedule.wording.title} · {schedule.wording.version}
      </a></p>
      <PdfPages pages={schedule.pages} />
    </section>}
  </div>;
}
