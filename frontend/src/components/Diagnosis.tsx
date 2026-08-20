import { useEffect, useState } from 'react';
import { api } from '../api';
import type { Signal } from '../types';
import { ShapChart } from './Charts';
import { Card, ConfidenceChip, SeverityChip, fmtPct } from './ui';

/* The thesis screen: the SHAP attribution and the physics rule table sit side
   by side. The model's opinion on the left, the deterministic physics check on
   the right. When they agree, that agreement IS the explanation. */
export function Diagnosis({ cycleId, onCycleChange }:
  { cycleId: number; onCycleChange: (id: number) => void }) {
  const [sig, setSig] = useState<Signal | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [input, setInput] = useState(String(cycleId));

  useEffect(() => {
    setInput(String(cycleId));
    setErr(null);
    api.reading(cycleId).then(setSig).catch((e) => setErr(String(e)));
  }, [cycleId]);

  if (err) return <div className="banner err">{err}</div>;
  if (!sig) return <div className="center"><span className="muted">Loading cycle {cycleId}…</span></div>;

  const r = sig.reading;
  const fired = sig.ruleChecks.filter((c) => c.triggered);

  return (
    <>
      <div className="controls">
        <span className="muted" style={{ fontSize: 12 }}>Inspect cycle</span>
        <input className="btn mono" style={{ width: 110 }} value={input}
               onChange={(e) => setInput(e.target.value)}
               onKeyDown={(e) => {
                 if (e.key === 'Enter') {
                   const n = parseInt(input, 10);
                   if (n >= 1 && n <= 10000) onCycleChange(n);
                 }
               }} />
        <button className="btn" onClick={() => {
          const n = parseInt(input, 10);
          if (n >= 1 && n <= 10000) onCycleChange(n);
        }}>Load</button>
        <span className="sep" />
        {[70, 51, 3237, 78].map((id) => (
          <button key={id} className="btn chipbtn" aria-pressed={cycleId === id}
                  onClick={() => onCycleChange(id)}>{id}</button>
        ))}
      </div>

      <Card style={{ marginBottom: 14 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
          <span className="mono" style={{ fontSize: 20, fontWeight: 600 }}>{r.machineId}</span>
          <SeverityChip severity={sig.severity} />
          <ConfidenceChip confidence={sig.confidence} />
          <span style={{ marginLeft: 'auto', textAlign: 'right' }}>
            <div className="tile-label">Model risk</div>
            <div className="mono" style={{ fontSize: 22, fontWeight: 600 }}>
              {fmtPct(r.failureProbability, 1)}
            </div>
          </span>
        </div>
        <p className="sec" style={{ margin: '12px 0 0', fontSize: 13 }}>{sig.rationale}</p>
      </Card>

      <div className="grid grid-2">
        <Card title="What the model looked at"
              note="SHAP contribution for this specific cycle — how each feature pushed the risk score.">
          <ShapChart values={sig.shapValues} />
        </Card>

        <Card title="What the physics rules verified"
              note="Deterministic threshold checks re-derived from the dataset's documented failure conditions. Non-triggered rules are shown too — 'we checked and it was fine' is evidence.">
          <div className="scroll-x">
            <table>
              <thead>
                <tr><th>Mode</th><th>Condition</th><th>Measured</th><th>Result</th></tr>
              </thead>
              <tbody>
                {sig.ruleChecks.map((c) => (
                  <tr key={c.mode} className={c.triggered ? 'row-hit' : ''}>
                    <td style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{c.mode}</td>
                    <td style={{ fontSize: 12 }}>{c.condition}</td>
                    <td className="num" style={{ fontSize: 12 }}>{c.measuredValue}</td>
                    <td>
                      {c.triggered
                        ? <span className="chip critical"><span className="chip-dot" />BREACHED</span>
                        : <span className="chip nominal"><span className="chip-dot" />OK</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="muted" style={{ fontSize: 12, marginTop: 12, marginBottom: 0 }}>
            {fired.length === 0
              ? 'No physical limit was breached on this cycle.'
              : `${fired.length} of ${sig.ruleChecks.length} rules breached.`}
            {' '}HDF, PWF and OSF each recover 100% of their true cases with zero false positives.
          </p>
        </Card>
      </div>

      <Card title="Evidence the agent will cite" style={{ marginTop: 14 }}
            note="Physics first, model attribution second — the deterministic check leads and the statistical model corroborates.">
        <ul className="evidence">
          {sig.evidence.map((e, i) => (
            <li key={i} className={e.startsWith('SHAP') ? '' : 'rule'}>{e}</li>
          ))}
        </ul>
        {sig.confidence === 'CONFLICT' && (
          <div className="banner warn" style={{ marginTop: 12, marginBottom: 0 }}>
            The model flags elevated risk but no physical rule fired, so no root cause is named.
            The agent escalates for human review rather than guessing — this is the designed
            behaviour, not a gap.
          </div>
        )}
      </Card>
    </>
  );
}
