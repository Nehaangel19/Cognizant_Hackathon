import type { ReactNode } from 'react';
import type { Confidence, Severity } from '../types';

export function Card({ title, note, children, style }:
  { title?: string; note?: string; children: ReactNode; style?: React.CSSProperties }) {
  return (
    <section className="card" style={style}>
      {title && <h2 className="card-title">{title}</h2>}
      {note && <p className="card-note">{note}</p>}
      {children}
    </section>
  );
}

/* Status is ALWAYS colour + text label + dot. The validated palette's CVD
   separation sits in the floor band, which is only legal with secondary
   encoding — so a bare coloured dot with no label is never rendered. */
export function SeverityChip({ severity }: { severity: Severity }) {
  return (
    <span className={`chip ${severity.toLowerCase()}`}>
      <span className="chip-dot" aria-hidden="true" />{severity}
    </span>
  );
}

export function ConfidenceChip({ confidence }: { confidence: Confidence }) {
  const cls = confidence === 'HIGH' ? 'nominal' : confidence === 'MEDIUM' ? 'advisory' : 'critical';
  const label = confidence === 'CONFLICT' ? 'CONFLICT — needs a human' : `${confidence} CONFIDENCE`;
  return (
    <span className={`chip ${cls}`}>
      <span className="chip-dot" aria-hidden="true" />{label}
    </span>
  );
}

export function Tile({ label, value, sub, tone }:
  { label: string; value: ReactNode; sub?: string; tone?: 'nominal' | 'caution' | 'critical' | 'accent' }) {
  const color = tone ? `var(--${tone})` : undefined;
  return (
    <div className="card">
      <div className="tile-label">{label}</div>
      <div className="tile-value" style={{ color }}>{value}</div>
      {sub && <div className="tile-sub">{sub}</div>}
    </div>
  );
}

export function severityTone(s: Severity) {
  return s === 'CRITICAL' ? 'critical' : s === 'WARNING' || s === 'ADVISORY' ? 'caution' : 'nominal';
}

export function fmtPct(n: number, digits = 0) { return `${(n * 100).toFixed(digits)}%`; }
export function fmtUsd(n: number) {
  return n >= 1000 ? `$${(n / 1000).toFixed(1)}k` : `$${n.toFixed(0)}`;
}
