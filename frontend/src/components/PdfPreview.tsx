import { useEffect, useId, useRef, useState, type ReactNode } from "react";

export type PdfPreviewPage = { png: string; text: string };

export function PdfPreviewDialog({ title, children, actions, onClose }: {
  title: string; children: ReactNode; actions?: ReactNode; onClose: () => void;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  const headingId = useId();

  useEffect(() => {
    const dialog = ref.current!;
    dialog.showModal();
    return () => dialog.close();
  }, []);

  return <dialog ref={ref} className="policy-document-preview pdf-preview-dialog" aria-labelledby={headingId}
    onCancel={event => { event.preventDefault(); onClose(); }}>
    <header className="claim-dialog-header">
      <h2 id={headingId}>{title}</h2>
      <div className="actions">
        {actions}
        <button type="button" onClick={onClose} autoFocus>Close</button>
      </div>
    </header>
    {children}
  </dialog>;
}

export function PdfPages({ pages }: { pages: PdfPreviewPage[] }) {
  const [imageError, setImageError] = useState(false);
  return <div className="pdf-preview-pages">
    {imageError && <p role="alert">One or more pages could not be displayed.</p>}
    {pages.map(({ png, text }, index) => (
      <img key={index} src={`data:image/png;base64,${png}`} alt={text} draggable={false}
        onError={() => setImageError(true)} />
    ))}
  </div>;
}
