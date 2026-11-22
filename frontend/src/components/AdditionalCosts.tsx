import { money } from "../components";

export default function AdditionalCosts({ costs }: { costs?: { amount: string; reason: string }[] }) {
  if (!costs?.length) return null;
  return (
    <section className="section-gap">
      <h3>Additional costs</h3>
      <dl className="preview-facts">
        {costs.map((cost, index) => (
          <div key={index}>
            <dt>{cost.reason}</dt>
            <dd>LKR {money(cost.amount)}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}
