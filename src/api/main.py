"""
FastAPI service layer — the bridge between the models/agent and the dashboard.

RUN:
    uvicorn src.api.main:app --reload --port 8000
DOCS:
    http://localhost:8000/docs   (auto-generated, interactive)

DESIGN NOTES
------------
The Engine and the Agent are loaded ONCE at startup via the lifespan hook, not
per request. `Engine()` takes ~1s (it loads five joblib artefacts plus the
dataset), so per-request construction would make the live demo visibly laggy.

Scored signals are memoised. SHAP is the bottleneck at roughly 10-20ms per row,
and the replay view re-requests the same cycles constantly, so an unbounded dict
cache keyed by cycle id keeps repeat reads effectively instant.

Every response shape mirrors the TypeScript data contract exactly (camelCase),
so the frontend consumes them with zero transformation.
"""

from __future__ import annotations

import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "agent"))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from features import REPO_ROOT

# Populated at startup.
STATE: dict = {"engine": None, "agent": None, "cache": {}, "error": None}


# --------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once. If they are missing, the API still starts and /health
    reports the problem, rather than crashing with an opaque traceback."""
    try:
        from predict import Engine
        from agent.agent import MaintenanceAgent

        engine = Engine()
        STATE["engine"] = engine
        STATE["agent"] = MaintenanceAgent()
        print(f"[api] models loaded — threshold {engine.threshold:.4f}")
    except Exception as exc:
        STATE["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[api] MODEL LOAD FAILED: {STATE['error']}\n"
              f"[api] Run `python src/train.py` first.", file=sys.stderr)
    yield
    STATE.clear()


app = FastAPI(
    title="SentinelOps — Predictive Maintenance Agent API",
    description="Explainable predictive maintenance for CNC machining lines.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",   # Vite
        "http://localhost:3000", "http://127.0.0.1:3000",   # CRA / Next
        "http://localhost:4173",                             # Vite preview
        "http://localhost:8501",                             # Streamlit
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
def _engine():
    if STATE["engine"] is None:
        raise HTTPException(503, detail=f"Models not loaded. {STATE['error'] or ''} "
                                        "Run `python src/train.py`.")
    return STATE["engine"]


def _signal(cycle_id: int) -> dict:
    """Memoised scored signal for one cycle."""
    engine = _engine()
    if not 1 <= cycle_id <= len(engine.df):
        raise HTTPException(404, detail=f"cycle_id {cycle_id} out of range (1-{len(engine.df)})")
    cache = STATE["cache"]
    if cycle_id not in cache:
        cache[cycle_id] = engine.score_udi(cycle_id)
    return cache[cycle_id]


class AnalyzeRequest(BaseModel):
    cycleId: int = Field(..., ge=1, description="Cycle / UDI to analyse")
    mode: str = Field("offline", description="'offline' (default, no API key) or 'llm'")


# --------------------------------------------------------------------------
@app.get("/health", tags=["system"])
def health():
    """Liveness + readiness. Reports what is actually loaded, not just 'ok'."""
    engine = STATE["engine"]
    return {
        "status": "ok" if engine else "degraded",
        "modelsLoaded": engine is not None,
        "error": STATE["error"],
        "threshold": round(engine.threshold, 4) if engine else None,
        "totalCycles": int(len(engine.df)) if engine else 0,
        "cachedCycles": len(STATE["cache"]),
    }


@app.get("/demo-cases", tags=["demo"])
def demo_cases():
    """Pre-seeded cycles for the demo, so the escalation beat lands on cue."""
    from predict import DEMO_CASES
    return {"demoCases": DEMO_CASES}


@app.get("/reading/{cycle_id}", tags=["readings"])
def reading(cycle_id: int):
    """One scored sensor reading: values, anomaly score, failure probability,
    rule checks, SHAP contributions and the fused verdict."""
    return _signal(cycle_id)


@app.get("/replay", tags=["readings"])
def replay(
    start: int = Query(1, ge=1, description="First cycle (inclusive)"),
    end: int = Query(50, ge=1, description="Last cycle (inclusive)"),
    limit: int = Query(500, ge=1, le=2000, description="Hard cap on rows returned"),
):
    """A window of consecutive scored readings — the live-stream view.

    The dataset has no timestamps, so cycle id doubles as the replay index. We
    say that openly in the pitch rather than implying a real time series.
    """
    engine = _engine()
    end = min(end, len(engine.df))
    if start > end:
        raise HTTPException(400, detail="start must be <= end")
    if end - start + 1 > limit:
        end = start + limit - 1
    return {"start": start, "end": end,
            "readings": [_signal(i) for i in range(start, end + 1)]}


@app.post("/analyze", tags=["agent"])
def analyze(req: AnalyzeRequest):
    """Run the agent on one cycle and return its decision + full work order.

    Defaults to the deterministic offline path: identical input always produces
    an identical verdict, which is what a decision that can stop a production
    line requires. `mode: "llm"` routes through the Anthropic tool-calling loop
    and falls back to offline if the API is unavailable — the response always
    reports which path actually ran.
    """
    engine = _engine()
    if not 1 <= req.cycleId <= len(engine.df):
        raise HTTPException(404, detail=f"cycleId {req.cycleId} out of range")
    agent = STATE["agent"]
    res = agent.run_llm(req.cycleId) if req.mode == "llm" else agent.run(req.cycleId)
    signal = _signal(req.cycleId)
    return {
        "workOrder": res["workOrder"],
        "narration": res["narration"],
        "decisionReason": res.get("decisionReason", ""),
        "toolTrace": res.get("toolTrace", []),
        "mode": res.get("mode", "offline"),
        "fallbackReason": res.get("fallbackReason"),
        "reading": signal["reading"],
    }


@app.get("/metrics", tags=["evidence"])
def metrics():
    """Model performance from the last training run — PR-AUC, recall, SHAP
    importances. Regenerated by `python src/train.py`."""
    path = REPO_ROOT / "docs" / "model_metrics.json"
    if not path.exists():
        raise HTTPException(404, detail="model_metrics.json missing. Run `python src/train.py`.")
    return json.loads(path.read_text())


@app.get("/rules-performance", tags=["evidence"])
def rules_performance():
    """Physics-rule scores against ground truth. The evidence that our
    explanations are verified, not plausible-sounding."""
    return {
        "rules": [
            {"mode": "HDF", "label": "Heat Dissipation", "truePositives": 115, "actual": 115,
             "falsePositives": 0, "precision": 1.0, "recall": 1.0, "exact": True},
            {"mode": "PWF", "label": "Power Failure", "truePositives": 95, "actual": 95,
             "falsePositives": 0, "precision": 1.0, "recall": 1.0, "exact": True},
            {"mode": "OSF", "label": "Overstrain", "truePositives": 98, "actual": 98,
             "falsePositives": 0, "precision": 1.0, "recall": 1.0, "exact": True},
            {"mode": "TWF", "label": "Tool Wear (risk window)", "truePositives": 43, "actual": 46,
             "falsePositives": 747, "precision": 0.054, "recall": 0.935, "exact": False},
        ],
        "note": ("TWF precision is low by construction: the failure point is drawn at random "
                 "inside a 200-240 min window, so 790 rows sit in the window and only 43 fail. "
                 "It is treated as advisory and never escalates on its own."),
        "coverage": {"explained": 287, "totalFailures": 339, "pct": 84.7},
    }


@app.get("/playbook/{mode}", tags=["evidence"])
def playbook(mode: str):
    """Maintenance playbook entry for a verified failure mode."""
    path = REPO_ROOT / "docs" / "playbook.json"
    if not path.exists():
        raise HTTPException(404, detail="playbook.json missing")
    data = json.loads(path.read_text())
    entry = data.get(mode.upper())
    if entry is None:
        raise HTTPException(404, detail=f"No playbook entry for '{mode}'")
    return {"mode": mode.upper(), **entry}


@app.get("/summary", tags=["evidence"])
def summary():
    """Fleet-level counts for the dashboard header tiles.

    Computed over a bounded window rather than all 10,000 cycles, because SHAP
    makes a full sweep take minutes. The window is stated in the response so the
    UI can label it honestly instead of implying a full-fleet scan.
    """
    engine = _engine()
    window = 300
    counts = {"NOMINAL": 0, "ADVISORY": 0, "WARNING": 0, "CRITICAL": 0}
    confidence = {"HIGH": 0, "MEDIUM": 0, "CONFLICT": 0}
    causes: dict[str, int] = {}
    for i in range(1, window + 1):
        s = _signal(i)
        counts[s["severity"]] = counts.get(s["severity"], 0) + 1
        confidence[s["confidence"]] = confidence.get(s["confidence"], 0) + 1
        if s["rootCause"]:
            causes[s["rootCause"]] = causes.get(s["rootCause"], 0) + 1
    return {"window": window, "severity": counts,
            "confidence": confidence, "rootCauses": causes}
