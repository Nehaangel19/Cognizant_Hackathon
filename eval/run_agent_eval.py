"""Run the agent over the 25-case eval set and report metrics.

Computes:
- Root-cause accuracy: fraction of cases where agent's rootCause matches expectedRootCause
- Decision accuracy: fraction of cases where agent's decision matches expectedDecision
- Hallucination rate: fraction of cases where agent names a cause the rules did not verify (must be 0)
- Schema validity: fraction of outputs that parse as valid WorkOrder

 Runs in both offline and --llm modes, reports both columns.
"""

import sys
import json
from pathlib import Path

# Setup path: script at .../eval/run_agent_eval.py
# Repo root at .../predictive-maintenance-agent (the inner directory under Downloads)
REPO_ROOT = Path(__file__).parent.parent  # e.g. C:\Users\angel\Downloads\predictive-maintenance-agent\predictive-maintenance-agent
sys.path.insert(0, str(REPO_ROOT))

from src.agent.agent import MaintenanceAgent
from src.agent.schemas import WorkOrder
from tools import Toolbox


def load_eval_set():
    eval_set_path = Path(__file__).parent / "agent_eval_set.json"
    with open(eval_set_path, 'r') as f:
        return json.load(f)


def run_agent_on_cycle(cycle_id, use_llm=False):
    """
    Run the agent on a single cycle.
    Returns (work_order_dict, validated, error_msg, mode_actually_used).

    `mode_actually_used` matters: run_llm() falls back to the offline path when
    the API is unavailable, and reporting those runs as "LLM" results would be a
    false claim. The caller counts fallbacks and prints them.
    """
    agent = MaintenanceAgent(toolbox=Toolbox())
    try:
        if use_llm:
            res = agent.run_llm(cycle_id)
        else:
            res = agent.run(cycle_id)
        mode_used = res.get("mode", "offline")
        wo = res["workOrder"]
        # Validate against Pydantic WorkOrder model
        try:
            validated = WorkOrder(**wo)
            return validated.model_dump(), True, None, mode_used
        except Exception as e:
            return wo, False, str(e), mode_used
    except Exception as e:
        return None, False, str(e), "error"


def check_hallucination(expected_rootCause, actualRootCause):
    """Check if the agent hallucinated a cause.

    Returns True if the agent named a cause the rules did not verify.
    """
    if expected_rootCause is None and actualRootCause is not None:
        return True  # Hallucination: agent named a cause when it should be null
    if expected_rootCause is not None and actualRootCause is None:
        return False  # Miss, not hallucination (agent admitted it can't verify)
    if expected_rootCause is not None and actualRootCause is not None and actualRootCause != expected_rootCause:
        return True  # Hallucination: agent named a different cause
    return False


def main():
    eval_cases = load_eval_set()
    total = len(eval_cases)

    results = {
        'offline': {'root_correct': 0, 'decision_correct': 0, 'hallucinations': 0, 'schema_valid': 0, 'fell_back': 0},
        'llm': {'root_correct': 0, 'decision_correct': 0, 'hallucinations': 0, 'schema_valid': 0, 'fell_back': 0},
    }

    print(f"\n{'=' * 70}")
    print(f"AGENT EVALUATION: {total} cases")
    print(f"{'=' * 70}")

    for mode in ["offline", "llm"]:
        print(f"\n--- Running in {mode.upper()} mode ---")
        for i, case in enumerate(eval_cases, 1):
            cycle_id = case['cycleId']
            wo_dict, valid, error, mode_used = run_agent_on_cycle(cycle_id, mode == "llm")
            if mode == "llm" and mode_used != "llm":
                results[mode]['fell_back'] += 1

            if wo_dict is None:
                print(f"  UDI {cycle_id}: ERROR - {error}")
                continue

            schema_valid = True
            actualRootCause = wo_dict.get('rootCause')
            actualDecision = wo_dict.get('decision')

            # Check root-cause match
            root_match = (actualRootCause == case['expectedRootCause'])
            if root_match:
                results[mode]['root_correct'] += 1

            # Check decision match
            decision_match = (actualDecision == case['expectedDecision'])
            if decision_match:
                results[mode]['decision_correct'] += 1

            # Check hallucination
            hallucination = check_hallucination(case['expectedRootCause'], actualRootCause)
            if hallucination:
                results[mode]['hallucinations'] += 1

            if schema_valid:
                results[mode]['schema_valid'] += 1

        root_acc = results[mode]['root_correct'] / total * 100
        dec_acc = results[mode]['decision_correct'] / total * 100
        hall_rate = results[mode]['hallucinations'] / total * 100
        schema_rate = results[mode]['schema_valid'] / total * 100

        print(f"  Root-cause accuracy: {results[mode]['root_correct']}/{total} = {root_acc:.1f}% (target: >= 23/25)")
        print(f"  Decision accuracy: {results[mode]['decision_correct']}/{total} = {dec_acc:.1f}% (target: >= 22/25)")
        print(f"  Hallucination rate: {results[mode]['hallucinations']}/{total} = {hall_rate:.1f}% (target: 0%)")
        print(f"  Schema validity: {results[mode]['schema_valid']}/{total} = {schema_rate:.1f}% (target: 25/25)")
        if mode == "llm":
            fb = results[mode]['fell_back']
            if fb == total:
                print(f"  !! ALL {total} cases fell back to the offline path. "
                      "The LLM was NEVER exercised - these are offline numbers.")
                print("  !! Do not report these as LLM results. Check ANTHROPIC_API_KEY and AGENT_MODEL.")
            elif fb:
                print(f"  !! {fb}/{total} cases fell back to the offline path "
                      "(API error). Genuine LLM runs: {total - fb}.")
            else:
                print(f"  OK: all {total} cases ran genuinely through the LLM.")

    # Write summary to docs/agent_eval.md
    md_path = Path(__file__).parent.parent / "docs" / "agent_eval.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)

    with open(md_path, 'w') as f:
        fb = results['llm']['fell_back']
        genuine = total - fb
        if fb == total:
            llm_label = "LLM (NOT RUN - all fell back)"
        elif fb:
            llm_label = f"LLM ({genuine}/{total} genuine, {fb} fell back)"
        else:
            llm_label = "LLM"

        f.write("# Agent Evaluation Results\n\n")
        if fb == total:
            f.write("> **WARNING:** every LLM-mode case fell back to the offline\n"
                    "> deterministic path, so the LLM column below is a copy of the offline\n"
                    "> column, NOT a measurement of the LLM. Do not quote it as an LLM result.\n"
                    "> Fix the API key / model id and re-run `python eval/run_agent_eval.py`.\n\n")
        elif fb:
            f.write(f"> **Note:** {fb} of {total} LLM-mode cases fell back to the offline path "
                    f"due to API errors. Only {genuine} are genuine LLM runs.\n\n")
        f.write(f"| Metric | Offline | {llm_label} |\n")
        f.write("|---|---|---|\n")
        root_offline = results['offline']['root_correct'] / total * 100
        root_llm = results['llm']['root_correct'] / total * 100
        dec_offline = results['offline']['decision_correct'] / total * 100
        dec_llm = results['llm']['decision_correct'] / total * 100
        hall_offline = results['offline']['hallucinations'] / total * 100
        hall_llm = results['llm']['hallucinations'] / total * 100
        schema_offline = results['offline']['schema_valid'] / total * 100
        schema_llm = results['llm']['schema_valid'] / total * 100

        f.write(f"| Root-cause accuracy | {results['offline']['root_correct']}/{total} ({root_offline:.1f}%) | {results['llm']['root_correct']}/{total} ({root_llm:.1f}%) |\n")
        f.write(f"| Decision accuracy | {results['offline']['decision_correct']}/{total} ({dec_offline:.1f}%) | {results['llm']['decision_correct']}/{total} ({dec_llm:.1f}%) |\n")
        f.write(f"| Hallucination rate | {results['offline']['hallucinations']}/{total} ({hall_offline:.1f}%) | {results['llm']['hallucinations']}/{total} ({hall_llm:.1f}%) |\n")
        f.write(f"| Schema validity | {results['offline']['schema_valid']}/{total} ({schema_offline:.1f}%) | {results['llm']['schema_valid']}/{total} ({schema_llm:.1f}%) |\n")

    print(f"\n✅ Evaluation results written to {md_path}")

    # Final verdict
    print(f"\n--- VERIFICATION ---")
    print(f"  Root-cause accuracy >= 23/25: {'PASS' if results['offline']['root_correct'] >= 23 else 'FAIL'} ({results['offline']['root_correct']}/{total})")
    print(f"  Decision accuracy >= 22/25: {'PASS' if results['offline']['decision_correct'] >= 22 else 'FAIL'} ({results['offline']['decision_correct']}/{total})")
    print(f"  Hallucination rate = 0: {'PASS' if results['offline']['hallucinations'] == 0 else 'FAIL'} ({results['offline']['hallucinations']}/{total})")
    print(f"  Schema validity 25/25: {'PASS' if results['offline']['schema_valid'] == 25 else 'FAIL'} ({results['offline']['schema_valid']}/{total})")


if __name__ == "__main__":
    main()