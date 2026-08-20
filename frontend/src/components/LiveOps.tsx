import { useEffect, useRef, useState } from 'react';
import { api } from '../api';
import type { Signal } from '../types';
import { RiskChart, SeverityBars } from './Charts';
import { Card, SeverityChip, Tile, fmtPct } from './ui';

const WINDOW = 60;   // cycles visible in the stream at once

export function LiveOps({ threshold, onInspect, onSelectCycle }: {
  threshold: number;
  /** open this cycle in the Diagnosis tab */
  onInspect: (id: number) => void;
  /** make this the app-wide selected cycle WITHOUT switching tabs, so that
      jumping to a seeded case here also arms Diagnosis and Agent Console */
  onSelectCycle: (id: number) => void;
}) {
  const [buffer, setBuffer] = useState<Signal[]>([]);
  const [cursor, setCursor] = useState(1);
  const [playing, setPlaying] = useState(false);
  const [selected, setSelected] = useState<number | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const timer = useRef<number | null>(null);

  // Prime the stream
  useEffect(() => {
    api.replay(1, 40)
      .then((r) => { setBuffer(r.readings); setCursor(41); })
      .catch((e) => setErr(String(e)));
  }, []);

  // Replay at ~2 cycles/sec. The dataset has no timestamps, so cycle id is the
  // replay index — we say so on screen rather than implying a real time series.
  useEffect(() => {
    if (!playing) return;
    timer.current = window.setInterval(async () => {
      try {
        const r = await api.replay(cursor, cursor + 1);
        setBuffer((b) => [...b, ...r.readings].slice(-WINDOW));
        setCursor((c) => (c + 2 > 9999 ? 1 : c + 2));
      } catch (e) { setErr(String(e)); setPlaying(false); }
    }, 500);
    return () => { if (timer.current) window.clearInterval(timer.current); };
  }, [playing, cursor]);

  const jump = async (id: number) => {
    setPlaying(false);
    const start = Math.max(1, id - 30);
    const r = await api.replay(start, id);
    setBuffer(r.readings);
    setCursor(id + 1);
    setSelected(id);
    onSelectCycle(id);   // keep the other tabs in sync with the seeded case
  };

  const latest = buffer[buffer.length - 1];
  const alerts = buffer.filter((s) => s.severity !== 'NOMINAL').slice().reverse();
  const counts = buffer.reduce<Record<string, number>>((a, s) => {
    a[s.severity] = (a[s.severity] ?? 0) + 1; return a;
  }, {});

  if (err) return <div className="banner err">Cannot reach the API — {err}<br />
    Start it with <code>uvicorn src.api.main:app --port 8000</code></div>;
  if (!latest) return <div className="center"><span className="muted">Loading stream…</span></div>;

  const r = latest.reading;

  return (
    <>
      <div className="controls">
        <button className="btn primary" onClick={() => setPlaying((p) => !p)}>
          {playing ? '❚❚  Pause replay' : '▶  Start replay'}
        </button>
        <span className="sep" />
        <span className="muted" style={{ fontSize: 12 }}>Jump to a seeded case:</span>
        <button className="btn chipbtn" onClick={() => jump(70)}>70 · overstrain</button>
        <button className="btn chipbtn" onClick={() => jump(51)}>51 · power</button>
        <button className="btn chipbtn" onClick={() => jump(3237)}>3237 · heat</button>
        <button className="btn chipbtn" onClick={() => jump(78)}>78 · conflict</button>
      </div>

      <div className="grid grid-4" style={{ marginBottom: 14 }}>
        <Tile label="Current cycle" value={<span className="mono">{r.cycleId}</span>}
              sub={`${r.machineId} · variant ${r.productType}`} />
        <Tile label="Failure risk"
              value={<span className="mono">{fmtPct(r.failureProbability, 1)}</span>}
              sub={`alert threshold ${fmtPct(threshold)}`}
              tone={r.failureProbability >= threshold ? 'critical' : 'nominal'} />
        <Tile label="Anomaly score" value={<span className="mono">{r.anomalyScore.toFixed(2)}</span>}
              sub="unsupervised, corroborating only" />
        <div className="card">
          <div className="tile-label">Status</div>
          <div style={{ marginTop: 6, marginBottom: 6 }}><SeverityChip severity={latest.severity} /></div>
          <div className="tile-sub">
            {latest.rootCause ? latest.rootCauseLabel
              : latest.confidence === 'CONFLICT' ? 'Flagged, no cause verified'
              : 'No fault detected'}
          </div>
        </div>
      </div>

      <div className="grid grid-3-1">
        <Card title="Failure risk across the replayed stream"
              note="Click any point to open it in Diagnosis. The dataset has no timestamps — cycle ID is the replay index, not a clock.">
          <RiskChart data={buffer} threshold={threshold} selected={selected}
                     onSelect={(id) => { setSelected(id); onInspect(id); }} />
        </Card>

        <div className="grid" style={{ gap: 14, alignContent: 'start' }}>
          <Card title="Live sensor readings">
            <dl style={{ margin: 0 }}>
              {[
                ['Air temperature', `${r.airTemperatureK.toFixed(1)} K`],
                ['Process temperature', `${r.processTemperatureK.toFixed(1)} K`],
                ['Temp difference', `${r.tempDiffK.toFixed(1)} K`],
                ['Rotational speed', `${r.rotationalSpeedRpm.toLocaleString()} rpm`],
                ['Torque', `${r.torqueNm.toFixed(1)} Nm`],
                ['Tool wear', `${r.toolWearMin} min`],
                ['Power', `${r.powerW.toLocaleString(undefined, { maximumFractionDigits: 0 })} W`],
                ['Wear × torque', `${r.wearTorqueMinNm.toLocaleString(undefined, { maximumFractionDigits: 0 })} minNm`],
              ].map(([k, v]) => (
                <div className="kv" key={k}><dt>{k}</dt><dd className="mono">{v}</dd></div>
              ))}
            </dl>
          </Card>

          <Card title="Severity mix" note={`Across the ${buffer.length} cycles on screen.`}>
            <SeverityBars counts={counts} />
          </Card>
        </div>
      </div>

      <Card title="Alert feed" note="Only cycles the agent flagged. Click one to inspect the evidence."
            style={{ marginTop: 14 }}>
        {alerts.length === 0
          ? <p className="muted" style={{ margin: 0 }}>No alerts in the current window — all cycles nominal.</p>
          : <div className="feed">
              {alerts.map((s) => (
                <button key={s.reading.cycleId} className="feed-item"
                        aria-current={selected === s.reading.cycleId}
                        onClick={() => { setSelected(s.reading.cycleId); onInspect(s.reading.cycleId); }}>
                  <span className="feed-cycle">#{s.reading.cycleId}</span>
                  <SeverityChip severity={s.severity} />
                  <span className="feed-cause">{s.rootCauseLabel}</span>
                  <span className="feed-prob"
                        style={{ color: s.severity === 'CRITICAL' ? 'var(--critical)' : 'var(--caution)' }}>
                    {fmtPct(s.reading.failureProbability, 1)}
                  </span>
                </button>
              ))}
            </div>}
      </Card>
    </>
  );
}
