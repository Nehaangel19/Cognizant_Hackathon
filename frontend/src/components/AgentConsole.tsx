import { useEffect, useState } from 'react';
import { api } from '../api';
import type { AnalyzeResponse } from '../types';
import { Card, ConfidenceChip, SeverityChip, fmtPct, fmtUsd } from './ui';

const DECISION_STYLE: Record<string, { label: string; tone: string; note: string }> = {
  'escalate-now': { label: 'ESCALATE NOW', tone: 'var(--critical)', note: 'Stop the line before the next cycle.' },
  'schedule': { label: 'SCHEDULE', tone: 'var(--caution)', note: 'Fix at the next planned stop. No line stop needed.' },
  'monitor': { label: 'MONITOR', tone: 'var(--nominal)', note: 'Healthy. Continue monitoring.' },
};

export function AgentConsole({ cycleId, onCycleChange }:
  { cycleId: number; onCycleChange: (id: number) => void }) {
  const [res, setRes] = useState<AnalyzeResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const run = (id: number) => {
    setBusy(true); setErr(null);
    api.analyze(id).then(setRes).catch((e) => setErr(String(e))).finally(() => setBusy(false));
  };

  useEffect(() => { run(cycleId); }, [cycleId]);

  const wo = res?.workOrder;
  const dec = wo ? (DECISION_STYLE[wo.decision] ?? DECISION_STYLE['monitor']) : null;

  return (
    <>
      <div className="controls">
        <span className="muted" style={{ fontSize: 12 }}>Run the agent on cycle</span>
        {[70, 51, 3237, 78].map((id) => (
          <button key={id} className="btn chipbtn" aria-pressed={cycleId === id}
                  onClick={() => onCycleChange(id)}>{id}</button>
        ))}
        <button className="btn primary" disabled={busy} onClick={() => run(cycleId)}>
          {busy ? 'Running…' : `↻  Re-run on ${cycleId}`}
        </button>
      </div>

      {err && <div className="banner err">{err}</div>}

      {wo && dec && (
        <>
          <Card style={{ marginBottom: 14 }}>
            <div style={{ display: 'flex', gap: 24, alignItems: 'flex-start', flexWrap: 'wrap' }}>
              <div style={{ minWidth: 260 }}>
                <div className="tile-label">Agent decision</div>
                {/* Hero figure: proportional figures, exactly one per view */}
                <div className="hero" style={{ color: dec.tone }}>{dec.label}</div>
                {/* Prefer the agent's own justification over a static string — a CONFLICT
                    escalation is routed to a human, which is not the same as "stop the line". */}
                <div className="tile-sub">{res.decisionReason || dec.note}</div>
              </div>
              <div style={{ flex: 1, minWidth: 260 }}>
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 10 }}>
                  <SeverityChip severity={wo.severity} />
                  <ConfidenceChip confidence={wo.confidence} />
                  <span className="chip info"><span className="chip-dot" />
                    {res.mode === 'llm' ? 'LLM TOOL-CALLING' : 'DETERMINISTIC'}
                  </span>
                </div>
                <p className="sec" style={{ margin: 0, fontSize: 13 }}>{res.narration}</p>
              </div>
            </div>
          </Card>

          <div className="grid grid-2">
            <Card title={`Work order ${wo.id}`}>
              <dl style={{ margin: 0 }}>
                {[
                  ['Machine', wo.machineId],
                  ['Cycle', String(wo.cycleId)],
                  ['Failure probability', fmtPct(wo.failureProbability, 1)],
                  ['Verified root cause', wo.rootCause ? wo.rootCauseLabel : '— none verified —'],
                  ['Estimated repair', `${wo.estimatedRepairHours} hrs`],
                  ['Downtime avoided', `${wo.downtimeAvoidedHours} hrs`],
                ].map(([k, v]) => (
                  <div className="kv" key={k}><dt>{k}</dt><dd className="mono">{v}</dd></div>
                ))}
                <div className="kv">
                  <dt>Cost avoided</dt>
                  <dd className="mono" style={{ color: 'var(--nominal)', fontWeight: 600 }}>
                    {wo.costAvoidedUsd ? fmtUsd(wo.costAvoidedUsd) : '—'}
                  </dd>
                </div>
              </dl>
            </Card>

            <Card title="Recommended actions"
                  note="Retrieved from the maintenance playbook, keyed by the VERIFIED mode. The agent never writes these itself.">
              {wo.recommendedActions.length ? (
                <ol style={{ margin: 0, paddingLeft: 18, fontSize: 13 }}>
                  {wo.recommendedActions.map((a, i) => (
                    <li key={i} style={{ marginBottom: 6 }}>{a}</li>
                  ))}
                </ol>
              ) : (
                <p className="muted" style={{ margin: 0, fontSize: 13 }}>
                  No actions — no root cause was verified, so the playbook was not consulted.
                  The agent will not invent a repair for a fault it cannot name.
                </p>
              )}
              {wo.partsRequired.length > 0 && (
                <>
                  <div className="tile-label" style={{ marginTop: 14 }}>Parts required</div>
                  <ul style={{ margin: '4px 0 0', paddingLeft: 18, fontSize: 13 }} className="sec">
                    {wo.partsRequired.map((p, i) => <li key={i}>{p}</li>)}
                  </ul>
                </>
              )}
            </Card>
          </div>

          <div className="grid grid-2" style={{ marginTop: 14 }}>
            <Card title="Evidence cited">
              <ul className="evidence">
                {wo.evidence.map((e, i) => (
                  <li key={i} className={e.startsWith('SHAP') ? '' : 'rule'}>{e}</li>
                ))}
              </ul>
            </Card>
            <Card title="Tool trace" note="Which tools the agent called, in order.">
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                {res.toolTrace.map((t, i) => (
                  <span key={i} className="chip info" style={{ fontFamily: 'var(--mono)' }}>
                    {i + 1}. {t}
                  </span>
                ))}
              </div>
              {res.fallbackReason && (
                <div className="banner warn" style={{ marginTop: 12, marginBottom: 0 }}>
                  LLM path unavailable, ran deterministically instead: {res.fallbackReason}
                </div>
              )}
              <p className="muted" style={{ fontSize: 12, marginTop: 12, marginBottom: 0 }}>
                The decision policy is deterministic by design. A verdict that can stop a
                production line must be reproducible and auditable — the same reading always
                yields the same decision.
              </p>
            </Card>
          </div>
        </>
      )}
    </>
  );
}
