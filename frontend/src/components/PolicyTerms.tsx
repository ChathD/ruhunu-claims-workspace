import { lossTypeLabel, money } from "../components";
import type { PersonPolicyRecord, VehicleDetails } from "../types";
import { incidentDocumentNames } from "./SyntheticIncidentEvidence";

const costEvidenceNames: Record<string, string> = {
  glass: "Itemised repair estimate", repair: "Itemised repair estimate", theft: "Itemised vehicle valuation",
  third_party_property_damage: "Itemised third-party property damage demand", third_party_injury: "Itemised third-party injury demand",
};

function termLabel(value: string) {
  return value === "suv" ? "SUV" : value.charAt(0).toUpperCase() + value.slice(1).replaceAll("_", " ");
}

export function VehicleFacts({ vehicle }: { vehicle?: VehicleDetails | null }) {
  if (!vehicle) return null;
  const description = [vehicle.make, vehicle.model, vehicle.year].filter(Boolean).join(" ");
  return (
    <dl className="preview-facts">
      {description && <div><dt>Vehicle</dt><dd>{description}</dd></div>}
      {vehicle.chassis_number && <div><dt>Chassis number</dt><dd>{vehicle.chassis_number}</dd></div>}
    </dl>
  );
}

export default function PolicyTerms({ policy }: { policy: PersonPolicyRecord["policy"] }) {
  const reportRequiredFor = policy.police_report_required_for;
  const repair = policy.repair_terms;
  const additional = repair?.additional_charges;
  const liability = policy.liability_terms;
  return (
    <>
      <VehicleFacts vehicle={policy.vehicle} />
      {policy.vehicle_type && policy.permitted_use && (
        <section className="section-gap">
          <h3>Permitted vehicle use</h3>
          <dl className="preview-facts">
            <div><dt>Vehicle type</dt><dd>{termLabel(policy.vehicle_type)}</dd></div>
            <div><dt>Permitted use</dt><dd>{termLabel(policy.permitted_use)}</dd></div>
          </dl>
          {!!policy.use_conditions?.length && <ul>{policy.use_conditions.map(term => <li key={term}>{term}</li>)}</ul>}
        </section>
      )}
      <section className="section-gap">
        <h3>Documents for coverage assessment</h3>
        {policy.covered_losses.filter(loss => costEvidenceNames[loss]).map(loss => (
          <div key={loss}>
            <h4>{lossTypeLabel(loss)}</h4>
            <ul>
              <li>{costEvidenceNames[loss]} · Required</li>
              <li>Police report · {reportRequiredFor?.includes(loss) ? "Required" : reportRequiredFor ? "Optional" : "Requirement unspecified"}</li>
              {policy.supporting_documents?.filter(rule => rule.loss_type === loss).map(rule => (
                <li key={rule.document_type}>{incidentDocumentNames[rule.document_type]} · {rule.requirement === "required" ? "Required" : "Optional"}<p>{rule.purpose}</p></li>
              ))}
            </ul>
          </div>
        ))}
      </section>
      {liability && (
        <section className="section-gap">
          <h3>Third-party liability</h3>
          <p>{liability.description}</p>
          <h4>Conditions</h4>
          <ul>{liability.conditions.map(term => <li key={term}>{term}</li>)}</ul>
          <h4>Exclusions</h4>
          <ul>
            {policy.insurance_type === "third_party" && <li>Damage to or theft of the insured vehicle</li>}
            {liability.exclusions.map(term => <li key={term}>{term}</li>)}
          </ul>
        </section>
      )}
      {repair && (
        <section className="section-gap">
          <h3>Repair cover</h3>
          <p>{repair.description}</p>
          <dl className="preview-facts">
            {!!repair.covered_incidents?.length && <div><dt>Covered incidents</dt><dd>{repair.covered_incidents.join(", ")}</dd></div>}
            {repair.flood_damage && <div><dt>Flood damage</dt><dd>{repair.flood_damage === "covered" ? "Covered" : "Excluded"}</dd></div>}
            {!!repair.covered_repairs?.length && <div><dt>Covered repairs</dt><dd>{repair.covered_repairs.join(", ")}</dd></div>}
            <div><dt>Required on estimate</dt><dd>Garage name, garage address, vehicle registration, date and itemised charges.</dd></div>
            {repair.labour && <div><dt>Labour</dt><dd>{repair.labour.limit === null ? "No separate limit" : `LKR ${money(repair.labour.limit)} ${repair.labour.basis.replaceAll("_", " ")}`}</dd></div>}
            {repair.painting && <div><dt>Painting</dt><dd>
              {repair.painting.scope === "excluded" ? "Excluded" : repair.painting.scope === "affected_panels_only" ? "Affected panels only" : "Full respray if required"}
              {repair.painting.limit_per_panel !== null && ` · LKR ${money(repair.painting.limit_per_panel)} per panel`}
              {repair.painting.max_panels !== null && ` · Up to ${repair.painting.max_panels} panels`}
            </dd></div>}
          </dl>
          {repair.repair_benefits?.map(benefit => (
            <section className="section-gap" key={benefit.cause}>
              <h4>{termLabel(benefit.cause)} repairs</h4>
              <p>{benefit.description}</p>
              <dl className="preview-facts">
                <div><dt>Covered work</dt><dd>{benefit.covered_repairs.join("; ")}</dd></div>
                {!!benefit.conditions.length && <div><dt>Conditions</dt><dd>{benefit.conditions.join("; ")}</dd></div>}
                {!!benefit.exclusions.length && <div><dt>Exclusions</dt><dd>{benefit.exclusions.join("; ")}</dd></div>}
              </dl>
            </section>
          ))}
          {additional && (
            <section className="section-gap">
              <h4>Additional charges</h4>
              {additional.covered ? <ul>{additional.eligible_charges.map(charge => (
                <li key={charge.category}>
                  <strong>{charge.name}</strong>
                  <p>{charge.conditions} {charge.limit_per_incident === null ? "No separate limit." : `Limit of LKR ${money(charge.limit_per_incident)} per incident.`} {charge.receipt_required ? "Receipt required." : "Receipt optional."}</p>
                </li>
              ))}</ul> : <p>All additional charges are excluded.</p>}
            </section>
          )}
          <section className="section-gap">
            <h4>Exclusions</h4>
            <ul>
              {repair.exclusions.map(item => <li key={item}>{item}</li>)}
              {additional?.exclusions.map(item => <li key={item.category}><strong>{item.name}</strong> · {item.reason}</li>)}
            </ul>
          </section>
        </section>
      )}
      {policy.theft_terms && (
        <section className="section-gap">
          <h3>Theft cover</h3>
          <p>{policy.theft_terms.description}</p>
          <dl className="preview-facts">
            <div><dt>Basis</dt><dd>{policy.theft_terms.basis === "vehicle_valuation_only"
              ? "Vehicle valuation only" : "Vehicle valuation and listed fittings"}</dd></div>
            {policy.theft_terms.covered_fittings.length > 0 && (
              <div><dt>Covered fittings</dt><dd>{policy.theft_terms.covered_fittings.join(", ")}</dd></div>
            )}
          </dl>
        </section>
      )}
    </>
  );
}
