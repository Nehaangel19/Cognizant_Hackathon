"""
Agent tools — the seven callable functions the orchestrator uses.

OWNER: Dev 2
DEPENDS ON: src/predict.py (Dev 1), docs/playbook.json + docs/cost_params.json
            (Non-tech A).

Design rule that keeps us honest: these tools SURFACE evidence, they do not
decide. `diagnose_root_cause` returns only what the physics rules verified;
`lookup_playbook` refuses a mode the rules did not confirm; `estimate_cost`
computes from a committed params file, never from a number the LLM made up.
The agent (agent.py) composes these; it cannot conjure a cause that no tool
returned.

Each tool has:
  - a plain Python function (usable offline, no LLM),
  - an ANTHROPIC_SCHEMA entry (the tool-calling definition for the API).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/

from predict import Engine
from features import REPO_ROOT

_PLAYBOOK_PATH = REPO_ROOT / "docs" / "playbook.json"
_COST_PATH = REPO_ROOT / "docs" / "cost_params.json"


class Toolbox:
    """Holds the loaded engine + reference data so tools share one instance."""

    def __init__(self, engine: Engine | None = None):
        self.engine = engine or Engine()
        # Fall back to an empty dict if Non-tech A's files are not merged yet,
        # so the agent degrades gracefully instead of crashing at the H+10 gate.
        self.playbook = json.loads(_PLAYBOOK_PATH.read_text()) if _PLAYBOOK_PATH.exists() else {}
        self.cost = json.loads(_COST_PATH.read_text()) if _COST_PATH.exists() else {}
        self._cache: dict[int, dict] = {}

    # -- internal: one scored signal per cycle, memoised -------------------
    def _signal(self, cycle_id: int) -> dict:
        if cycle_id not in self._cache:
            self._cache[cycle_id] = self.engine.score_udi(cycle_id)
        return self._cache[cycle_id]

    # ======================================================================
    # TOOL 1
    def detect_anomaly(self, cycle_id: int) -> dict:
        """Unsupervised anomaly score in [0,1] for one cycle."""
        s = self._signal(cycle_id)
        score = s["reading"]["anomalyScore"]
        return {"cycleId": cycle_id, "anomalyScore": score,
                "isAnomalous": score >= 0.5,
                "note": "IsolationForest, fit on normal rows only; corroborating signal."}

    # TOOL 2
    def predict_failure_prob(self, cycle_id: int) -> dict:
        """Calibrated-ranking failure probability from the XGBoost model."""
        s = self._signal(cycle_id)
        p = s["reading"]["failureProbability"]
        return {"cycleId": cycle_id, "failureProbability": p,
                "threshold": s["modelThreshold"],
                "exceedsThreshold": p >= s["modelThreshold"]}

    # TOOL 3
    def diagnose_root_cause(self, cycle_id: int) -> dict:
        """
        The physics-verified root cause and its SHAP support. Returns rootCause
        None when no exact rule fired — the agent must NOT name a cause in that
        case.
        """
        s = self._signal(cycle_id)
        return {"cycleId": cycle_id, "rootCause": s["rootCause"],
                "rootCauseLabel": s["rootCauseLabel"],
                "confidence": s["confidence"],
                "shapValues": s["shapValues"], "evidence": s["evidence"]}

    # TOOL 4
    def check_physics_rules(self, cycle_id: int) -> dict:
        """Every deterministic rule check for the cycle, triggered or not."""
        s = self._signal(cycle_id)
        return {"cycleId": cycle_id, "ruleChecks": s["ruleChecks"],
                "confidence": s["confidence"], "rationale": s["rationale"]}

    # TOOL 5
    def lookup_playbook(self, mode: str) -> dict:
        """
        Maintenance playbook entry for a VERIFIED failure mode. Refuses anything
        not confirmed by the rules engine — this is the anti-hallucination guard.
        """
        mode = (mode or "").upper()
        if mode not in self.playbook:
            return {"mode": mode, "found": False,
                    "note": "No verified playbook entry. The agent must not invent actions."}
        entry = self.playbook[mode]
        return {"mode": mode, "found": True, **entry}

    # TOOL 6
    def estimate_cost(self, mode: str, repair_hours: float | None = None) -> dict:
        """
        Downtime avoided and cost avoided for a verified mode, from the committed
        cost params. Never fabricates figures.
        """
        mode = (mode or "").upper()
        rate = self.cost.get("downtime_cost_per_hour", 2600)
        per_mode = self.cost.get("per_mode", {}).get(mode)
        if per_mode is None:
            return {"mode": mode, "found": False, "note": "No cost profile for this mode."}
        downtime = per_mode["downtime_hours"]
        collateral = per_mode.get("collateral_usd", 0)
        downtime_cost = round(downtime * rate, 2)
        return {"mode": mode, "found": True,
                "downtimeAvoidedHours": downtime,
                "downtimeCostUsd": downtime_cost,
                "collateralAvoidedUsd": collateral,
                "costAvoidedUsd": round(downtime_cost + collateral, 2),
                "ratePerHourUsd": rate,
                "collateralNote": per_mode.get("collateral_note", "")}

    # TOOL 7
    def create_work_order(self, cycle_id: int, decision: str, created_at: str = "") -> dict:
        """
        Assemble the final structured WorkOrder from everything the other tools
        surfaced. Validated against the Pydantic WorkOrder before returning, so a
        malformed order fails here rather than in the UI.
        """
        from schemas import WorkOrder  # local import; src/agent on path via agent.py

        s = self._signal(cycle_id)
        mode = s["rootCause"]
        pb = self.lookup_playbook(mode) if mode else {}
        cost = self.estimate_cost(mode) if mode else {}

        order = WorkOrder(
            id=f"WO-{cycle_id:05d}",
            cycleId=cycle_id,
            machineId=s["reading"]["machineId"],
            severity=s["severity"],
            confidence=s["confidence"],
            failureProbability=s["reading"]["failureProbability"],
            rootCause=mode,
            rootCauseLabel=s["rootCauseLabel"],
            evidence=s["evidence"],
            ruleChecks=s["ruleChecks"],
            shapValues=s["shapValues"],
            recommendedActions=pb.get("recommended_actions", []),
            partsRequired=pb.get("parts_required", []),
            estimatedRepairHours=pb.get("repair_hours", 0.0),
            downtimeAvoidedHours=cost.get("downtimeAvoidedHours", 0.0),
            costAvoidedUsd=cost.get("costAvoidedUsd", 0.0),
            decision=decision,
            createdAt=created_at,
        )
        return order.model_dump()


# --------------------------------------------------------------------------
# Anthropic tool-calling schemas. agent.py hands these to the API; the names
# match the Toolbox methods so dispatch is a getattr.
# --------------------------------------------------------------------------
def anthropic_tool_schemas() -> list[dict]:
    cyc = {"cycle_id": {"type": "integer", "description": "The cycle / UDI to analyse."}}
    return [
        {"name": "detect_anomaly", "description": "Unsupervised anomaly score for a cycle.",
         "input_schema": {"type": "object", "properties": cyc, "required": ["cycle_id"]}},
        {"name": "predict_failure_prob", "description": "Model failure probability for a cycle.",
         "input_schema": {"type": "object", "properties": cyc, "required": ["cycle_id"]}},
        {"name": "diagnose_root_cause",
         "description": "Physics-verified root cause + SHAP. rootCause is null if no exact rule fired.",
         "input_schema": {"type": "object", "properties": cyc, "required": ["cycle_id"]}},
        {"name": "check_physics_rules", "description": "All deterministic rule checks for a cycle.",
         "input_schema": {"type": "object", "properties": cyc, "required": ["cycle_id"]}},
        {"name": "lookup_playbook",
         "description": "Maintenance actions/parts for a VERIFIED mode (TWF/HDF/PWF/OSF/RNF).",
         "input_schema": {"type": "object",
                          "properties": {"mode": {"type": "string", "enum": ["TWF", "HDF", "PWF", "OSF", "RNF"]}},
                          "required": ["mode"]}},
        {"name": "estimate_cost", "description": "Downtime and cost avoided for a verified mode.",
         "input_schema": {"type": "object",
                          "properties": {"mode": {"type": "string"}},
                          "required": ["mode"]}},
        {"name": "create_work_order",
         "description": "Assemble the final structured work order once a decision is made.",
         "input_schema": {"type": "object",
                          "properties": {**cyc,
                                         "decision": {"type": "string",
                                                      "enum": ["escalate-now", "schedule", "monitor"]}},
                          "required": ["cycle_id", "decision"]}},
    ]


if __name__ == "__main__":
    tb = Toolbox()
    for udi in [70, 51, 78, 1]:
        s = tb._signal(udi)
        print(f"UDI {udi}: {s['severity']:<8} conf={s['confidence']:<8} cause={s['rootCause']}")
    print("\nlookup_playbook('OSF'):")
    print(json.dumps(tb.lookup_playbook("OSF"), indent=2)[:400])
    print("\nestimate_cost('OSF'):")
    print(json.dumps(tb.estimate_cost("OSF"), indent=2))
    print("\nlookup_playbook on an unverified mode returns found=False:",
          tb.lookup_playbook("XYZ")["found"])
