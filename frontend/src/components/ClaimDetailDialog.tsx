import { useEffect, useRef } from "react";
import type { ReactNode } from "react";

/** A native modal keeps details in view and restores focus to the claims list. */
export default function ClaimDetailDialog({
  reference,
  status,
  onClose,
  children,
}: {
  reference: string;
  status: string;
  onClose: () => void;
  children: ReactNode;
}) {
  const ref = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const dialog = ref.current!;
    dialog.showModal();
    return () => dialog.close();
  }, []);

  return (
    <dialog
      ref={ref}
      className="claim-detail-dialog"
      aria-labelledby="claim-detail-title"
      onCancel={(event) => {
        event.preventDefault();
        onClose();
      }}
      onClick={(event) => {
        if (event.target === event.currentTarget) {
          const box = event.currentTarget.getBoundingClientRect();
          if (
            event.clientX < box.left ||
            event.clientX > box.right ||
            event.clientY < box.top ||
            event.clientY > box.bottom
          )
            onClose();
        }
      }}
    >
      <header className="claim-dialog-header">
        <div>
          <div className="eyebrow">CLAIM DETAILS</div>
          <h2 id="claim-detail-title">{reference}</h2>
        </div>
        <span
          className={`badge ${status === "approved" ? "green" : status === "rejected" ? "red" : "amber"}`}
        >
          {status}
        </span>
        <button
          autoFocus
          type="button"
          className="button secondary"
          onClick={onClose}
        >
          Close <span aria-hidden="true">×</span>
        </button>
      </header>
      <div className="claim-dialog-content">{children}</div>
    </dialog>
  );
}
