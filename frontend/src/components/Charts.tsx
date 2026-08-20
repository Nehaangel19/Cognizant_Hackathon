/* Hand-built SVG charts.

   No chart library on purpose: the mark specs (2px strokes, >=8px markers, 4px
   rounded data-ends, 2px surface gaps between adjacent fills, recessive grid,
   crosshair tooltips) are easier to honour exactly when we own the geometry,
   and it keeps the dependency list to React alone.

   No dual-axis charts anywhere: where two measures of different scale need to
   be compared they get two charts, never two y-scales. */

import { useState } from 'react';
import type { ShapContribution, Signal } from '../types';

const AXIS = '#6C7F94';
const GRID = '#1e2a38';
const SURFACE = '#121A24';

type Tip = { x: number; y: number; rows: [string, string][]; title: string } | null;

function Tooltip({ tip }: { tip: Tip }) {
  if (!tip) return null;
  return (
    <div className="tip" style={{ left: tip.x + 14, top: tip.y + 14 }}>
      <div style={{ fontWeight: 600, marginBottom: 4 }}>{tip.title}</div>
      {tip.rows.map(([k, v]) => (
        <div className="tip-row" key={k}>
          <span className="tip-k">{k}</span><span className="tip-v">{v}</span>
        </div>
      ))}
    </div>
  );
}

/* ---------- Risk over cycles: single series, so no legend box — the title
     names it. Threshold drawn as a reference rule, not a second series. ---------- */
export function RiskChart({ data, threshold, selected, onSelect }: {
  data: Signal[]; threshold: number; selected: number | null; onSelect: (id: number) => void;
}) {
  const [tip, setTip] = useState<Tip>(null);
  const W = 900, H = 220, PL = 44, PR = 12, PT = 12, PB = 26;
  if (!data.length) return <div className="muted">No data yet.</div>;

  const iw = W - PL - PR, ih = H - PT - PB;
  const x = (i: number) => PL + (data.length === 1 ? iw / 2 : (i / (data.length - 1)) * iw);
  const y = (v: number) => PT + ih - v * ih;

  const path = data.map((d, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(d.reading.failureProbability).toFixed(1)}`).join('');
  const area = `${path}L${x(data.length - 1).toFixed(1)},${y(0)}L${x(0).toFixed(1)},${y(0)}Z`;

  return (
    <div style={{ position: 'relative' }}>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} role="img"
           aria-label="Failure probability across replayed cycles">
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line x1={PL} x2={W - PR} y1={y(t)} y2={y(t)} stroke={GRID} strokeWidth={1} />
            <text x={PL - 8} y={y(t) + 4} textAnchor="end" fontSize={10} fill={AXIS}
                  fontFamily="JetBrains Mono, monospace">{(t * 100).toFixed(0)}%</text>
          </g>
        ))}

        <defs>
          <linearGradient id="riskFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#3987e5" stopOpacity="0.28" />
            <stop offset="100%" stopColor="#3987e5" stopOpacity="0.02" />
          </linearGradient>
        </defs>
        <path d={area} fill="url(#riskFill)" />
        <path d={path} fill="none" stroke="#3987e5" strokeWidth={2}
              strokeLinejoin="round" strokeLinecap="round" />

        {/* Decision threshold — a reference rule, directly labelled */}
        <line x1={PL} x2={W - PR} y1={y(threshold)} y2={y(threshold)}
              stroke="#c98500" strokeWidth={2} strokeDasharray="5 4" />
        <text x={W - PR} y={y(threshold) - 6} textAnchor="end" fontSize={10} fill="#c98500"
              fontFamily="JetBrains Mono, monospace">
          alert threshold {(threshold * 100).toFixed(0)}%
        </text>

        {/* Markers only on alerting cycles — never a dot on every point */}
        {data.map((d, i) => {
          if (d.severity === 'NOMINAL') return null;
          const c = d.severity === 'CRITICAL' ? '#e34948' : '#c98500';
          const isSel = selected === d.reading.cycleId;
          return (
            <circle key={d.reading.cycleId} cx={x(i)} cy={y(d.reading.failureProbability)}
                    r={isSel ? 6 : 4.5} fill={c} stroke={SURFACE} strokeWidth={2}
                    style={{ cursor: 'pointer' }}
                    onClick={() => onSelect(d.reading.cycleId)}
                    onMouseMove={(e) => setTip({
                      x: e.clientX, y: e.clientY,
                      title: `Cycle ${d.reading.cycleId} · ${d.severity}`,
                      rows: [
                        ['Risk', `${(d.reading.failureProbability * 100).toFixed(1)}%`],
                        ['Cause', d.rootCause ?? 'unverified'],
                        ['Confidence', d.confidence],
                      ],
                    })}
                    onMouseLeave={() => setTip(null)} />
          );
        })}

        {/* Invisible wide hit targets so hovering the line is easy */}
        {data.map((d, i) => (
          <rect key={`h${d.reading.cycleId}`} x={x(i) - iw / data.length / 2} y={PT}
                width={Math.max(iw / data.length, 6)} height={ih} fill="transparent"
                style={{ cursor: 'pointer' }}
                onClick={() => onSelect(d.reading.cycleId)}
                onMouseMove={(e) => setTip({
                  x: e.clientX, y: e.clientY,
                  title: `Cycle ${d.reading.cycleId}`,
                  rows: [
                    ['Risk', `${(d.reading.failureProbability * 100).toFixed(1)}%`],
                    ['Anomaly', d.reading.anomalyScore.toFixed(2)],
                    ['Severity', d.severity],
                  ],
                })}
                onMouseLeave={() => setTip(null)} />
        ))}

        <line x1={PL} x2={W - PR} y1={y(0)} y2={y(0)} stroke={GRID} strokeWidth={1} />
        <text x={PL} y={H - 6} fontSize={10} fill={AXIS} fontFamily="JetBrains Mono, monospace">
          cycle {data[0].reading.cycleId}
        </text>
        <text x={W - PR} y={H - 6} textAnchor="end" fontSize={10} fill={AXIS}
              fontFamily="JetBrains Mono, monospace">
          cycle {data[data.length - 1].reading.cycleId}
        </text>
      </svg>
      <Tooltip tip={tip} />
    </div>
  );
}

/* ---------- SHAP: a diverging form. Two hues either side of a neutral zero
     rule — red raises risk, blue lowers it. Never a rainbow, never a hue at
     the midpoint. ---------- */
export function ShapChart({ values }: { values: ShapContribution[] }) {
  const [tip, setTip] = useState<Tip>(null);
  if (!values.length) return <div className="muted">No attribution available.</div>;

  const rowH = 30, W = 460, PL = 150, PR = 54;
  const H = values.length * rowH + 26;
  const max = Math.max(...values.map((v) => Math.abs(v.contribution))) || 1;
  const iw = W - PL - PR, mid = PL + iw / 2;
  const scale = (v: number) => (v / max) * (iw / 2);

  return (
    <div style={{ position: 'relative' }}>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} role="img"
           aria-label="Feature contributions to this cycle's risk score">
        <line x1={mid} x2={mid} y1={4} y2={H - 22} stroke={AXIS} strokeWidth={1} />
        {values.map((v, i) => {
          const yy = i * rowH + 6;
          const w = Math.abs(scale(v.contribution));
          const pos = v.contribution > 0;
          const xx = pos ? mid : mid - w;
          const c = pos ? '#e34948' : '#3987e5';
          return (
            <g key={v.feature}
               onMouseMove={(e) => setTip({
                 x: e.clientX, y: e.clientY, title: v.feature,
                 rows: [
                   ['Value', v.value.toLocaleString()],
                   ['Contribution', `${v.contribution > 0 ? '+' : ''}${v.contribution.toFixed(2)}`],
                   ['Effect', pos ? 'raises risk' : 'lowers risk'],
                 ],
               })}
               onMouseLeave={() => setTip(null)} style={{ cursor: 'default' }}>
              <rect x={PL - 148} y={yy - 4} width={W} height={rowH - 4} fill="transparent" />
              <text x={PL - 10} y={yy + 13} textAnchor="end" fontSize={11} fill="#9FB0C4"
                    fontFamily="JetBrains Mono, monospace">{v.feature}</text>
              {/* 4px rounded data-end, anchored to the zero rule */}
              <rect x={xx} y={yy} width={Math.max(w, 2)} height={16} fill={c}
                    rx={4} ry={4} />
              {/* 2px surface gap so the fill never touches the zero rule */}
              <rect x={mid - 1} y={yy - 1} width={2} height={18} fill={SURFACE} />
              <text x={pos ? mid + w + 6 : mid - w - 6} y={yy + 13}
                    textAnchor={pos ? 'start' : 'end'} fontSize={11} fill="#E6EDF5"
                    fontFamily="JetBrains Mono, monospace">
                {v.contribution > 0 ? '+' : ''}{v.contribution.toFixed(2)}
              </text>
            </g>
          );
        })}
        <text x={mid - 8} y={H - 6} textAnchor="end" fontSize={10} fill={AXIS}>lowers risk</text>
        <text x={mid + 8} y={H - 6} fontSize={10} fill={AXIS}>raises risk</text>
      </svg>
      <div className="legend">
        <span className="legend-item">
          <span className="legend-swatch" style={{ background: '#e34948' }} />raises risk
        </span>
        <span className="legend-item">
          <span className="legend-swatch" style={{ background: '#3987e5' }} />lowers risk
        </span>
      </div>
      <Tooltip tip={tip} />
    </div>
  );
}

/* ---------- Severity mix: status colours, each bar directly labelled ---------- */
export function SeverityBars({ counts }: { counts: Record<string, number> }) {
  const order = ['NOMINAL', 'ADVISORY', 'WARNING', 'CRITICAL'] as const;
  const colors: Record<string, string> = {
    NOMINAL: '#199e70', ADVISORY: '#c98500', WARNING: '#c98500', CRITICAL: '#e34948',
  };
  const total = order.reduce((s, k) => s + (counts[k] ?? 0), 0) || 1;
  return (
    <div>
      {order.map((k) => {
        const v = counts[k] ?? 0;
        const pct = (v / total) * 100;
        return (
          <div key={k} style={{ marginBottom: 10 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
              <span className="sec">{k}</span>
              <span className="mono">{v} · {pct.toFixed(1)}%</span>
            </div>
            <div style={{ height: 8, background: '#1b2634', borderRadius: 4, overflow: 'hidden' }}>
              <div style={{ width: `${pct}%`, height: '100%', background: colors[k], borderRadius: 4 }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}
