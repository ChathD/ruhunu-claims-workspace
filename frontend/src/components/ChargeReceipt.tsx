import { FileText } from "lucide-react";
import { money } from "../components";
import type { ChargeReceiptDocument } from "../types";

export default function ChargeReceipt({ receipt }: { receipt: ChargeReceiptDocument }) {
  return (
    <div className="document-preview">
      <div className="document-title">
        <FileText size={18} />
        <h3>{receipt.name}</h3>
        <span className="badge neutral">Additional charge receipt</span>
      </div>
      <dl>
        <dt>Service</dt><dd>{receipt.category.replaceAll("_", " ")}</dd>
        <dt>Vehicle registration</dt><dd>{receipt.registration}</dd>
        <dt>Receipt date</dt><dd>{receipt.date}</dd>
        <dt>Amount</dt><dd>LKR {money(receipt.total)}</dd>
        <dt>Details</dt><dd>{receipt.description}</dd>
      </dl>
    </div>
  );
}
