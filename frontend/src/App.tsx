// build: force fresh asset hashes for the AWS Amplify redeploy
import { useEffect, useState } from 'react';
import { api } from './api';
import type { Health } from './types';
import { AgentConsole } from './components/AgentConsole';
import { Diagnosis } from './components/Diagnosis';
import { Impact } from './components/Impact';
import { LiveOps } from './components/LiveOps';

type Tab = 'live' | 'diagnosis' | 'agent' | 'impact';

const TABS: { id: Tab; label: string }[] = [
  { id: 'live', label: 'Live Operations' },
  { id: 'diagnosis', label: 'Diagnosis' },
  { id: 'agent', label: 'Agent Console' },
  { id: 'impact', label: 'Impact & Performance' },
];

export default function App() {
  console.info('[sentinelops] build cb=aws-deploy-2');
  const [tab, setTab] = useState<Tab>('live');
  const [cycleId, setCycleId] = useState(70);
  const [health, setHealth] = useState<Health | null>(null);
  const [down, setDown] = useState(false);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setDown(true));
  }, []);

  const inspect = (id: number) => { setCycleId(id); setTab('diagnosis'); };

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">SENTINEL<span style={{ color: 'var(--accent)' }}>OPS</span></span>
          <span className="brand-sub">Predictive Maintenance Agent</span>
        </div>
        <nav className="tabs" role="tablist">
          {TABS.map((t) => (
            <button key={t.id} className="tab" role="tab" aria-selected={tab === t.id}
                    onClick={() => setTab(t.id)}>{t.label}</button>
          ))}
        </nav>
        <span className={`chip ${down ? 'critical' : health?.modelsLoaded ? 'nominal' : 'advisory'}`}>
          <span className="chip-dot" aria-hidden="true" />
          {down ? 'API OFFLINE' : health?.modelsLoaded ? 'MODELS LOADED' : 'STARTING'}
        </span>
      </header>

      <main className="main">
        {down && (
          <div className="banner err">
            <strong>Cannot reach the API.</strong> Start the backend from the repo root:
            <br />
            <code>uvicorn src.api.main:app --reload --port 8000</code>
            <br />
            If it is running, check that <code>python src/train.py</code> has been run so
            <code> models/</code> exists.
          </div>
        )}

        {tab === 'live' && (
          <LiveOps threshold={health?.threshold ?? 0.25} onInspect={inspect}
                   onSelectCycle={setCycleId} />
        )}
        {tab === 'diagnosis' && (
          <Diagnosis cycleId={cycleId} onCycleChange={setCycleId} />
        )}
        {tab === 'agent' && (
          <AgentConsole cycleId={cycleId} onCycleChange={setCycleId} />
        )}
        {tab === 'impact' && <Impact />}
      </main>
    </div>
  );
}
