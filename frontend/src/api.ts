/* Single place the UI talks to the backend.

   Requests go to /api/* and Vite proxies them to the FastAPI server on :8000
   (see vite.config.ts), so the browser only ever makes same-origin calls and
   CORS cannot become a demo-day problem. Override with VITE_API_BASE if you
   run the API somewhere else. */

import type { AnalyzeResponse, Health, RulePerf, Signal } from './types';

const BASE = (import.meta as any).env?.VITE_API_BASE ?? '/api';

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`${res.status} ${res.statusText} — ${body.slice(0, 200)}`);
  }
  return res.json();
}

export const api = {
  health: () => get<Health>('/health'),
  reading: (id: number) => get<Signal>(`/reading/${id}`),
  replay: (start: number, end: number) =>
    get<{ start: number; end: number; readings: Signal[] }>(`/replay?start=${start}&end=${end}`),
  demoCases: () => get<{ demoCases: Record<string, number[]> }>('/demo-cases'),
  metrics: () => get<any>('/metrics'),
  rulesPerformance: () => get<{ rules: RulePerf[]; note: string; coverage: any }>('/rules-performance'),
  analyze: async (cycleId: number): Promise<AnalyzeResponse> => {
    const res = await fetch(`${BASE}/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ cycleId, mode: 'offline' }),
    });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return res.json();
  },
};
