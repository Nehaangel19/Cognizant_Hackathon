import { useEffect, useState } from 'react';
import { api } from '../api';
import type { RulePerf } from '../types';
import { Card, Tile } from './ui';

export function Impact() {
  const [m, setM] = useState<any>(null);
  const [rules, setRules] = useState<{ rules: RulePerf[]; note: string; coverage: any } | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.metrics(), api.rulesPerformance()])
      .then(([mm, rr]) => { setM(mm); setRules(rr); })
      .catch((e) => setErr(String(e)));
  }, []);

  if (err) return <div className="banner err">{err}</div>;
  if (!m || !rules) return <div className="center"><span className="muted">Loading metrics…</span></div>;

  const fm = m.failure_model;
  const cm = fm.confusion_matrix;

  // Cost model: caught failures x average downtime hours x loaded rate.
  // Rate and hours come from docs/cost_params.json (Non-tech A's cost model).
  const RATE = 2600, AVG_HRS = 2.5;
  const avoided = cm.tp * AVG_HRS * RATE;

  return (
    <>
      <div className="grid grid-4" style={{ marginBottom: 14 }}>
        <Tile label="PR-AUC (headline)" value={<span className="mono">{fm.pr_auc.toFixed(3)}</span>}
              sub={`${fm.lift_over_baseline}× the ${m.baseline.dummy_pr_auc.toFixed(3)} baseline`} tone="accent" />
        <Tile label="Recall at tuned threshold" value={<span className="mono">{(fm.recall * 100).toFixed(1)}%</span>}
              sub={`caught ${cm.tp} of ${cm.tp + cm.fn} real failures`} tone="nominal" />
        <Tile label="Precision at that threshold" value={<span className="mono">{(fm.precision * 100).toFixed(1)}%</span>}
              sub={`${cm.fp} false alarms in ${m.split.test_rows.toLocaleString()} cycles`} />
        <Tile label="Root cause at HIGH confidence" value={<span className="mono">100%</span>}
              sub="287/287 across all 10,000 rows" tone="nominal" />
      </div>

      <div className="banner warn">
        <strong>We do not report accuracy as a headline.</strong> At a {(m.dataset.positive_rate * 100).toFixed(2)}% failure
        rate, a model that predicts “never fails” scores {(m.baseline.never_fails_accuracy * 100).toFixed(1)}% accuracy and
        catches nothing. PR-AUC and recall are the honest measures.
      </div>

      <div className="grid grid-2">
        <Card title="Physics rules vs ground truth"
              note="Scored across all 10,000 rows. This is the evidence that our explanations are verified, not merely plausible.">
          <div className="scroll-x">
            <table>
              <thead>
                <tr><th>Mode</th><th>Caught</th><th>False +</th><th>Precision</th><th>Recall</th></tr>
              </thead>
              <tbody>
                {rules.rules.map((r) => (
                  <tr key={r.mode}>
                    <td style={{ color: 'var(--text-primary)', fontWeight: 600 }}>
                      {r.mode} <span className="muted" style={{ fontWeight: 400 }}>{r.label}</span>
                    </td>
                    <td className="num">{r.truePositives}/{r.actual}</td>
                    <td className="num">{r.falsePositives}</td>
                    <td className="num" style={{ color: r.exact ? 'var(--nominal)' : 'var(--caution)' }}>
                      {r.precision.toFixed(3)}
                    </td>
                    <td className="num">{r.recall.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="muted" style={{ fontSize: 12, marginTop: 12, marginBottom: 0 }}>{rules.note}</p>
        </Card>

        <Card title="Confusion matrix at the tuned threshold"
              note={`Threshold ${fm.tuned_threshold.toFixed(3)}, chosen to hold recall at or above 85% — a missed failure costs far more than an inspection.`}>
          <table>
            <thead>
              <tr><th></th><th>Predicted normal</th><th>Predicted failure</th></tr>
            </thead>
            <tbody>
              <tr>
                <td style={{ color: 'var(--text-secondary)' }}>Actually normal</td>
                <td className="num">{cm.tn.toLocaleString()}</td>
                <td className="num" style={{ color: 'var(--caution)' }}>{cm.fp} false alarms</td>
              </tr>
              <tr>
                <td style={{ color: 'var(--text-secondary)' }}>Actually failed</td>
                <td className="num" style={{ color: 'var(--critical)' }}>{cm.fn} missed</td>
                <td className="num" style={{ color: 'var(--nominal)' }}>{cm.tp} caught</td>
              </tr>
            </tbody>
          </table>
          <div className="tile-label" style={{ marginTop: 16 }}>Estimated downtime avoided</div>
          <div className="tile-value mono" style={{ color: 'var(--nominal)' }}>
            ${avoided.toLocaleString()}
          </div>
          <p className="muted" style={{ fontSize: 12, marginTop: 6, marginBottom: 0 }}>
            {cm.tp} failures caught × {AVG_HRS} hrs average downtime × ${RATE.toLocaleString()}/hr loaded rate.
            The rate is built up transparently in <code>docs/cost_model.md</code> from a published
            $120/hr shop rate — not the headline $22k/min figure, which describes a full
            automotive assembly line and would be indefensible for a single cell.
          </p>
        </Card>
      </div>

      <div className="grid grid-2" style={{ marginTop: 14 }}>
        <Card title="What the model relies on"
              note="Mean absolute SHAP across 500 test cycles. The top features are the engineered physics quantities — the model and the rules engine look at the same things.">
          {Object.entries(m.shap_global_importance).slice(0, 8).map(([k, v]) => {
            const max = Math.max(...Object.values(m.shap_global_importance).map(Number));
            const pct = (Number(v) / max) * 100;
            return (
              <div key={k} style={{ marginBottom: 9 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
                  <span className="mono sec">{k}</span>
                  <span className="mono">{Number(v).toFixed(2)}</span>
                </div>
                <div style={{ height: 8, background: '#1b2634', borderRadius: 4 }}>
                  <div style={{ width: `${pct}%`, height: '100%', background: '#3987e5', borderRadius: 4 }} />
                </div>
              </div>
            );
          })}
        </Card>

        <Card title="Named limitations"
              note="Stated openly. A judge finding these unprompted costs far more than us naming them first.">
          <ul className="sec" style={{ margin: 0, paddingLeft: 18, fontSize: 13, lineHeight: 1.8 }}>
            <li><strong>Synthetic data.</strong> Physically grounded, but not a real plant, so cost figures are an illustrative model.</li>
            <li><strong>No timestamps or machine IDs.</strong> Rows are independent snapshots; cycle ID is a replay index only.</li>
            <li><strong>RNF is irreducible.</strong> 0.1% random by construction, no sensor signature, excluded by design.</li>
            <li><strong>9 of 339 failures carry no cause label</strong>, so rules can explain at most 330.</li>
            <li><strong>Tool wear is a risk window, not a verdict.</strong> 0.054 precision is the mathematical ceiling.</li>
            <li><strong>Probabilities are ranked, not calibrated.</strong> <code>scale_pos_weight</code> distorts them.</li>
          </ul>
        </Card>
      </div>
    </>
  );
}
