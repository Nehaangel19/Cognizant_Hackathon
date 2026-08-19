"""
The maintenance agent — the layer that DECIDES.

OWNER: Dev 2
This is what makes the project an agent rather than a dashboard: given a cycle,
it gathers evidence through the tools, then commits to one of three decisions —
escalate now, schedule at the next stop, or keep monitoring — and emits a
structured work order justifying it.

Two run modes, same output shape:

  MODE A (default, offline): a deterministic policy over the confidence gate.
    No API key, no network, fully reproducible. This is the demo-safe path and
    the H+18 gate fallback ("drop tool calling, inject signals").

  MODE B (--llm): a real Anthropic tool-calling loop. The model calls the same
    seven tools, but the decision policy and the anti-hallucination guard live
    in the tools, so the LLM cannot invent a cause or a cost.

Run:
    python src/agent/agent.py 70            # offline
    python src/agent/agent.py 70 --llm      # tool-calling (needs ANTHROPIC_API_KEY)
    python src/agent/agent.py --demo        # runs the pre-seeded demo cases offline
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))          # src/agent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # src

from tools import Toolbox, anthropic_tool_schemas
from predict import DEMO_CASES

# --------------------------------------------------------------------------
# The decision policy — the core of "it decides something"
# --------------------------------------------------------------------------
# Severity and confidence come from the fused ML+physics signal. The decision
# is a deterministic function of them, so it is auditable and identical whether
# an LLM or the offline path produced it.
def decide(severity: str, confidence: str) -> tuple[str, str]:
    """Return (decision, one-line justification)."""
    if severity == "CRITICAL" and confidence == "HIGH":
        return "escalate-now", ("Verified critical failure imminent — stop the line before "
                                "the next cycle.")
    if confidence == "CONFLICT":
        return "escalate-now", ("Model and physics disagree — no verified cause. Escalate to "
                                "a human rather than auto-closing or guessing.")
    if severity == "WARNING" and confidence == "HIGH":
        return "schedule", ("Verified fault, not yet critical — schedule a fix at the next "
                            "planned stop.")
    if severity in ("ADVISORY", "MEDIUM") or confidence == "MEDIUM":
        return "schedule", ("Advisory signal (tool nearing its replacement window) — schedule "
                            "a tool change; no line stop needed.")
    return "monitor", "Nominal — model and physics agree the cycle is healthy. Continue monitoring."


SYSTEM_PROMPT = """You are a predictive-maintenance agent for a CNC machining line.
For the given cycle you must reach ONE decision: escalate-now, schedule, or monitor.

Hard rules:
- You may only name a root cause that check_physics_rules / diagnose_root_cause
  VERIFIED. If confidence is CONFLICT, you must NOT name a cause — escalate for
  human review instead.
- Never invent maintenance actions, parts, or costs. Get them from lookup_playbook
  and estimate_cost, which read committed reference data.
- Always finish by calling create_work_order with your decision.

Work in this order: check_physics_rules -> predict_failure_prob ->
diagnose_root_cause -> (if a cause is verified) lookup_playbook + estimate_cost ->
create_work_order. Then give a two-sentence spoken summary for the operator."""


class MaintenanceAgent:
    def __init__(self, toolbox: Toolbox | None = None):
        self.tb = toolbox or Toolbox()

    # ---- MODE A: offline deterministic ----------------------------------
    def run(self, cycle_id: int, created_at: str = "") -> dict:
        """Deterministic path. Returns {workOrder, narration, toolTrace}."""
        trace = []
        rules = self.tb.check_physics_rules(cycle_id);      trace.append("check_physics_rules")
        prob = self.tb.predict_failure_prob(cycle_id);      trace.append("predict_failure_prob")
        diag = self.tb.diagnose_root_cause(cycle_id);       trace.append("diagnose_root_cause")

        signal = self.tb._signal(cycle_id)
        decision, why = decide(signal["severity"], signal["confidence"])

        if diag["rootCause"]:
            trace.append("lookup_playbook")
            trace.append("estimate_cost")
        wo = self.tb.create_work_order(cycle_id, decision, created_at);  trace.append("create_work_order")

        narration = self._narrate(wo, why)
        return {"workOrder": wo, "narration": narration, "decisionReason": why, "toolTrace": trace}

    @staticmethod
    def _narrate(wo: dict, why: str) -> str:
        mid = wo["machineId"]
        if wo["rootCause"]:
            head = (f"{mid}: {wo['severity']} — {wo['rootCauseLabel']} "
                    f"({wo['confidence']} confidence, {wo['failureProbability']:.0%} model risk).")
            body = f" Recommend: {decision_verb(wo['decision'])}."
            if wo["costAvoidedUsd"]:
                body += (f" Acting now avoids ~{wo['downtimeAvoidedHours']:.1f} hrs of downtime "
                         f"(~${wo['costAvoidedUsd']:,.0f}).")
        elif wo["confidence"] == "CONFLICT":
            head = (f"{mid}: {wo['severity']} — model flags {wo['failureProbability']:.0%} risk but "
                    "no physical rule fired.")
            body = " No verified cause; escalating for human review rather than guessing."
        else:
            head = f"{mid}: NOMINAL — model and physics agree the cycle is healthy."
            body = " Continue monitoring."
        return head + body

    # ---- MODE B: Anthropic tool-calling loop ----------------------------
    def run_llm(self, cycle_id: int, model: str | None = None, max_turns: int = 8) -> dict:
        import anthropic

        model = model or os.environ.get("AGENT_MODEL", "claude-sonnet-4-5")
        client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY
        tools = anthropic_tool_schemas()
        messages = [{"role": "user",
                     "content": f"Analyse cycle {cycle_id} and reach a decision."}]
        trace, work_order = [], None

        for _ in range(max_turns):
            resp = client.messages.create(model=model, max_tokens=1024,
                                           system=SYSTEM_PROMPT, tools=tools, messages=messages)
            messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason != "tool_use":
                final_text = "".join(b.text for b in resp.content if b.type == "text")
                return {"workOrder": work_order, "narration": final_text, "toolTrace": trace}

            results = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                trace.append(block.name)
                fn = getattr(self.tb, block.name)
                out = fn(**block.input)
                if block.name == "create_work_order":
                    work_order = out
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": json.dumps(out)})
            messages.append({"role": "user", "content": results})

        return {"workOrder": work_order, "narration": "(max turns reached)", "toolTrace": trace}


def decision_verb(decision: str) -> str:
    return {"escalate-now": "stop the line and act now",
            "schedule": "schedule a fix at the next planned stop",
            "monitor": "continue monitoring"}.get(decision, decision)


# --------------------------------------------------------------------------
def _print_result(cycle_id: int, res: dict) -> None:
    wo = res["workOrder"]
    print(f"\n{'=' * 70}\nCYCLE {cycle_id}\n{'=' * 70}")
    print(f"DECISION: {wo['decision'].upper() if wo else '(none)'}")
    print(f"narration: {res['narration']}")
    print(f"tools called: {' -> '.join(res['toolTrace'])}")
    if wo:
        print("\nwork order:")
        print(json.dumps(wo, indent=2))


if __name__ == "__main__":
    args = sys.argv[1:]
    agent = MaintenanceAgent()

    if not args or args[0] == "--demo":
        print("Running pre-seeded demo cases (offline, deterministic)")
        for label, udis in DEMO_CASES.items():
            udi = udis[0]
            res = agent.run(udi)
            wo = res["workOrder"]
            print(f"\n[{label}] UDI {udi}: {wo['decision'].upper():<13} | "
                  f"{wo['severity']:<8} | {wo['confidence']:<8} | {wo['rootCauseLabel']}")
            print(f"   {res['narration']}")
    else:
        cycle_id = int(args[0])
        use_llm = "--llm" in args
        res = agent.run_llm(cycle_id) if use_llm else agent.run(cycle_id)
        _print_result(cycle_id, res)
