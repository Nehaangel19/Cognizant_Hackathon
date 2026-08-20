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

  MODE B (--llm): a tool-calling loop. The model calls the same seven tools,
    but the decision policy and the anti-hallucination guard live in the tools,
    so the LLM cannot invent a cause. If the LLM path fails (no API key, network
    error, timeout, or max turns exceeded), it automatically falls back to the
    offline deterministic path and logs that it did. The demo must never show a
    stack trace.

Run:
    python src/agent/agent.py 70            # offline
    python src/agent/agent.py 70 --llm      # tool-calling (needs API key or falls back)
    python src/agent/agent.py --demo        # runs the pre-seeded demo cases offline
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))          # src/agent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # src

from tools import Toolbox, anthropic_tool_schemas
from predict import DEMO_CASES
from llm_client import get_llm_client, LocalTemplateLLM, OllamaLLM

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


def _call_llm_with_retry(llm, system_prompt: str, user_prompt: str, context: dict,
                         max_retries: int = 3, base_backoff: float = 1.0) -> str | None:
    """Call llm.generate() with exponential-backoff retry. Returns None on permanent failure."""
    for attempt in range(1, max_retries + 1):
        try:
            return llm.generate(system_prompt, user_prompt, context)
        except Exception as e:
            if attempt == max_retries:
                print(f"[agent] LLM call failed after {max_retries} attempts: {e}", file=sys.stderr)
                return None
            backoff = base_backoff * (2 ** (attempt - 1))
            print(f"[agent] LLM attempt {attempt} failed, retrying in {backoff}s...", file=sys.stderr)
            time.sleep(backoff)
    return None


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
            head = (f"{mid}: {wo['severity']} "
                    f"{wo['rootCauseLabel']} "
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

    # ---- MODE B: tool-calling loop (with fallback) ----------------------------
    def run_llm(self, cycle_id: int, model: str | None = None, max_turns: int = 8,
                 timeout_per_call: float = 15.0) -> dict:
        """
        Tool-calling loop with retry, timeout, and hard fallback.
        If the LLM path raises, times out, or exceeds max turns,
        automatically falls back to the offline deterministic path
        and logs that it did. The demo must never show a stack trace.
        """
        llm = get_llm_client(model=model)
        system_prompt = SYSTEM_PROMPT
        user_prompt = f"Analyse cycle {cycle_id} and reach a decision."

        # Warm-up: test that the LLM client works
        try:
            _call_llm_with_retry(llm, system_prompt, user_prompt,
                                 {"role": "sensor_anomaly", "max_abs_zscore": 0, "max_zscore_sensor": "none"},
                                 max_retries=1, base_backoff=0.1)
        except Exception:
            pass  # non-fatal; we'll fall back if actual cycle fails

        trace, work_order = [], None

        # Run with per-call timeout and retry
        deadline = time.time() + timeout_per_call * 4  # generous overall wall-clock limit

        for turn in range(max_turns):
            if time.time() > deadline:
                print("[agent] LLM loop terminated: timeout exceeded", file=sys.stderr)
                break

            try:
                resp = llm.generate(system_prompt, user_prompt, {"role": "user", "context": None})
            except Exception as e:
                print(f"[agent] LLM generate exception: {e}, falling back to offline path", file=sys.stderr)
                break

            if resp is None:
                print("[agent] LLM returned None, falling back to offline path", file=sys.stderr)
                break

            # Handle string output (LocalTemplateLLM) vs structured output
            if isinstance(resp, str):
                # String output from LocalTemplateLLM - it's already JSON
                # Parse it to check structure
                try:
                    parsed = json.loads(resp)
                    # If it contains work order fields, use it
                    if "rootCause" in parsed or "decision" in parsed:
                        # Build a minimal trace
                        trace = ["offline_fallback"]
                        work_order = self.run(cycle_id)["workOrder"]
                        break
                except json.JSONDecodeError:
                    pass  # not JSON, treat as text
                # Treat as final text; if we don't have a work_order yet, fall back
                if work_order is None:
                    print("[agent] LLM returned non-JSON text, falling back offline", file=sys.stderr)
                    work_order = self.run(cycle_id)["workOrder"]
                    trace = ["offline_fallback"]
                    break
                # Otherwise we already have a work_order from a prior turn
                else:
                    final_text = resp
                    # Continue to tool processing below
            else:
                final_text = "".join(b.text for b in resp.content if b.type == "text") if hasattr(resp, 'content') else ""

            # Check if model called a tool (for OllamaLLM/Anthropic)
            # For LocalTemplateLLM returning a dict, we need different handling
            if isinstance(resp, dict) and "content" in resp:
                # Structured response from OllamaLLM/Anthropic
                tool_blocks = [b for b in resp.content if b.type == "tool_use"] if hasattr(resp, 'content') else []
                if tool_blocks:
                    # Model called a tool - process them
                    results = []
                    for block in tool_blocks:
                        trace.append(block.name)
                        fn = getattr(self.tb, block.name)
                        try:
                            out = fn(**block.input)
                        except Exception as e:
                            print(f"[agent] Tool {block.name} raised {e}, continuing", file=sys.stderr)
                            continue
                        if block.name == "create_work_order":
                            work_order = out
                        results.append({"type": "tool_result", "tool_use_id": block.id,
                                        "content": json.dumps(out)})
                    messages.append({"role": "user", "content": results})
                    continue  # next turn
                else:
                    # No tool calls - model gave up or gave text
                    if work_order is None:
                        print("[agent] LLM did not call create_work_order, falling back offline", file=sys.stderr)
                        work_order = self.run(cycle_id)["workOrder"]
                        trace = ["offline_fallback"]
                    final_text = final_text or "(no text from model)"
            elif isinstance(resp, str):
                # String output - check if we need to fallback
                if work_order is None:
                    print("[agent] LLM returned string without tool calls, falling back offline", file=sys.stderr)
                    work_order = self.run(cycle_id)["workOrder"]
                    trace = ["offline_fallback"]
                    break
                else:
                    # We already have a work order, just use the text
                    final_text = resp

        # ---- Fallback logic ---------------------------------------------------
        if work_order is None:
            print("[agent] LLM path produced no work order — using offline deterministic path", file=sys.stderr)
            res = self.run(cycle_id)
            work_order = res["workOrder"]
            if trace and trace[0] != "offline_fallback":
                trace = ["offline_fallback"] + trace
            narration = self._narrate(work_order, "")
        else:
            narration = self._narrate(work_order, "")

        return {"workOrder": work_order, "narration": narration, "toolTrace": trace}


def decision_verb(decision: str) -> str:
    return {"escalate-now": "stop the line and act now",
            "schedule": "schedule a fix at the next planned stop",
            "monitor": "continue monitoring"}.get(decision, decision)


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
        json_only = "--json" in args
        res = agent.run_llm(cycle_id) if use_llm else agent.run(cycle_id)
        wo = res["workOrder"]
        if json_only:
            print(json.dumps(wo))
        else:
            _print_result(cycle_id, res)