import type { InvestigationToolCall } from "./types";

type Evidence = {
  tool: string;
  request: unknown;
  result: unknown;
};

function evidenceKey({ tool, request, result }: Evidence, supporting?: unknown) {
  return JSON.stringify([tool, request, result, supporting], (_, value: unknown) =>
    value && typeof value === "object" && !Array.isArray(value)
      ? Object.fromEntries(Object.entries(value).sort(([a], [b]) => a.localeCompare(b)))
      : value,
  );
}

export function groupInvestigationEvidence(calls: InvestigationToolCall[]) {
  const groups = new Map<string, {
    id: number;
    stages: Set<string>;
    evidence: Evidence;
    supporting?: Evidence[];
    references: Map<string, { stage: string; evidence_ref: string }>;
  }>();

  for (const call of calls) {
    const reference = call.response.evidence_ref;
    const wrapped = typeof reference === "string" && "result" in call.response;
    const evidence = { tool: call.tool, request: call.request, result: wrapped ? call.response.result : call.response };
    const supporting = call.tool === "execute_checks" && Array.isArray(call.response.supporting_evidence)
      ? call.response.supporting_evidence as Evidence[]
      : undefined;
    const key = evidenceKey(evidence, supporting);
    let group = groups.get(key);
    if (!group) {
      group = { id: call.id, stages: new Set(), evidence, supporting, references: new Map() };
      groups.set(key, group);
    }
    group.stages.add(call.stage);
    if (wrapped) group.references.set(reference, { stage: call.stage, evidence_ref: reference });
  }

  return Array.from(groups.values(), ({ id, stages, evidence, supporting, references }) => {
    const stage = Array.from(stages).join(", ");
    const title = `${stage} · ${evidence.tool}`;
    const payload = { request: evidence.request, response: evidence.result };
    const supportingEvidence = supporting?.map(entry => {
      const linked = groups.get(evidenceKey(entry));
      return linked ? { tool: entry.tool, shown_under: `${Array.from(linked.stages).join(", ")} · ${entry.tool}` } : entry;
    });
    return {
      id,
      title,
      payload: {
        ...payload,
        ...(references.size ? { evidence_references: Array.from(references.values()) } : {}),
        ...(supportingEvidence?.length ? { supporting_evidence: supportingEvidence } : {}),
      },
    };
  });
}
