# SentinelOps — frontend

React + TypeScript + Vite. No chart library and no UI framework: the charts are
hand-built SVG so the mark specs (2px strokes, 4px rounded data-ends, 2px surface
gaps, recessive grid, crosshair tooltips) are exact, and the dependency list stays
at React alone.

## Run it

The dashboard needs the API. **Two terminals, both from the repo root.**

```bash
# Terminal 1 — backend
python src/train.py                            # once, if models/ is empty (~30s)
uvicorn src.api.main:app --reload --port 8000

# Terminal 2 — frontend
cd frontend
npm install
npm run dev                                    # http://localhost:5173
```

Vite proxies `/api/*` to `localhost:8000` (see `vite.config.ts`), so the browser
only makes same-origin requests and CORS can never break the demo. Point it
elsewhere with `VITE_API_BASE` if you need to.

## The four screens

| Screen | What it is for |
|---|---|
| **Live Operations** | Replay stream at ~2 cycles/sec, live sensor values, alert feed |
| **Diagnosis** | SHAP attribution beside the physics rule table — the project's thesis |
| **Agent Console** | The decision, the work order, actions, parts, cost avoided, tool trace |
| **Impact & Performance** | PR-AUC, rules-vs-ground-truth, confusion matrix, limitations |

## Demo path (3 minutes)

1. **Live Operations** → *Start replay*. Gauges green, risk flat.
2. Click **70 · overstrain**. Risk spikes, a CRITICAL alert appears.
3. **Diagnosis** → SHAP says `osf_margin`; the rule table shows wear × torque at
   12,549 minNm against an 11,000 limit. Model and physics, independently, on the
   same quantity.
4. **Agent Console** → ESCALATE NOW, work order, ~$19.6k avoided.
5. Click **78 · conflict** → the agent reports 97% risk, names **no** cause, and
   routes to a human. This is the credibility beat.
6. **Impact & Performance** → PR-AUC 0.87, rules at 1.000/1.000, limitations.

## A note on the palette

The handoff's original accent (`#D97706`) was **replaced**. Validated against the
dark surface, it sat at ΔE 4.4 from the critical red under deuteranopia — an amber
"accent" would have read as an alarm to a colourblind operator. The current set
(`#199e70` / `#c98500` / `#e34948` / `#3987e5`) passes every check. Its CVD
separation sits in the floor band, which is only legal with secondary encoding, so
**every status is rendered as colour + dot + text label, never colour alone.**
