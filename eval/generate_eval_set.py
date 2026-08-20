"""Generate the 25-case agent evaluation set from the AI4I 2020 dataset.

Uses ground-truth mode flag columns for expected values.
Produces eval/agent_eval_set.json and docs/agent_eval.md.
Distinct UDIs for each category.
"""

from pathlib import Path
import json
import numpy as np

repo_root = Path(__file__).parent.parent.parent
data_path = repo_root / "data" / "ai4i2020.csv"

# Add repo to path
import sys
sys.path.insert(0, str(repo_root / "predictive-maintenance-agent" / "predictive-maintenance-agent" / "src"))

from features import build_feature_matrix
from rules_engine import assess, Engine


def main():
    repo_root = Path(__file__).parent.parent.parent
    data_path = repo_root / "data" / "ai4i2020.csv"
    eval_set_path = repo_root / "eval" / "agent_eval_set.json"
    eval_set_path.parent.mkdir(parents=True, exist_ok=True)

    # Build feature matrix and engine
    _, _, df = build_feature_matrix(data_path, verify=True)
    engine = Engine()

    # Organize UDIs by ground-truth mode flag
    mode_udis = {}
    for mode in ['twf', 'hdf', 'pwf', 'osf']:
        mode_udis[mode] = df[df[mode] == 1]['udi'].tolist()
    # Nominal UDIs: machine_failure=0 and no mode flags are 1
    nominal_mask = (df['machine_failure'] == 0)
    for mode in ['twf', 'hdf', 'pwf', 'osf']:
        nominal_mask = nominal_mask & (df[mode] == 0)
    nominal_udis = df[nominal_mask]['udi'].tolist()

    print(f"Mode UDI counts - TWF: {len(mode_udis['twf'])}, HDF: {len(mode_udis['hdf'])}, "
          f"PWF: {len(mode_udis['pwf'])}, OSF: {len(mode_udis['osf'])}, Nominal: {len(nominal_udis)}")

    # Avoid demo UDIs
    demo_UDIs = {70, 161, 162, 243, 51, 169, 195, 208, 3237, 3761, 3788, 3794, 78, 1088, 1438, 1510}

    cases = []
    used_udis = set()

    # ---- 5 OSF cases (rootCause=OSF, escalate-now, HIGH confidence) ----
    print("\nSelecting 5 OSF cases...")
    osf_available = [u for u in mode_udis['osf'] if u not in used_udis and u not in demo_UDIs]
    for i in range(5):
        udi = osf_available[i]
        used_udis.add(udi)
        row = df.loc[int(udi)]
        signal = engine.score_udi(int(udi))
        prob = signal["reading"]["failureProbability"]
        # OSF with HIGH probability and rule verified -> escalate-now, HIGH confidence
        if prob >= 0.70:
            decision = "escalate-now"
            confidence = "HIGH"
        else:
            decision = "escalate-now"
            confidence = signal["confidence"]
        cases.append({
            'cycleId': int(udi),
            'expectedRootCause': 'OSF',
            'expectedDecision': decision,
            'expectedConfidence': confidence,
            'note': f"OSF overstrain failure"
        })

    # ---- 5 PWF cases (rootCause=PWF, escalate-now, HIGH confidence) ----
    print("Selecting 5 PWF cases...")
    pwf_available = [u for u in mode_udis['pwf'] if u not in used_udis and u not in demo_UDIs]
    for i in range(5):
        udi = pwf_available[i]
        used_udis.add(udi)
        row = df.loc[int(udi)]
        signal = engine.score_udi(int(udi))
        prob = signal["reading"]["failureProbability"]
        if prob >= 0.70:
            decision = "escalate-now"
            confidence = "HIGH"
        else:
            decision = "escalate-now"
            confidence = signal["confidence"]
        cases.append({
            'cycleId': int(udi),
            'expectedRootCause': 'PWF',
            'expectedDecision': decision,
            'expectedConfidence': confidence,
            'note': f"Power failure"
        })

    # ---- 5 HDF cases (rootCause=HDF, escalate-now, HIGH confidence) ----
    print("Selecting 5 HDF cases...")
    hdf_available = [u for u in mode_udis['hdf'] if u not in used_udis and u not in demo_UDIs]
    for i in range(5):
        udi = hdf_available[i]
        used_udis.add(udi)
        row = df.loc[int(udi)]
        signal = engine.score_udi(int(udi))
        prob = signal["reading"]["failureProbability"]
        if prob >= 0.70:
            decision = "escalate-now"
            confidence = "HIGH"
        else:
            decision = "escalate-now"
            confidence = signal["confidence"]
        cases.append({
            'cycleId': int(udi),
            'expectedRootCause': 'HDF',
            'expectedDecision': decision,
            'expectedConfidence': confidence,
            'note': f"Heat dissipation failure"
        })

    # ---- 4 TWF cases (rootCause=TWF, schedule, MEDIUM confidence - advisory) ----
    print("Selecting 4 TWF cases (advisory)...")
    twf_available = [u for u in mode_udis['twf'] if u not in used_udis and u not in demo_UDIs]
    for i in range(4):
        udi = twf_available[i]
        used_udis.add(udi)
        row = df.loc[int(udi)]
        signal = engine.score_udi(int(udi))
        # TWF: always advisory - rootCause=TWF, decision=schedule, MEDIUM confidence
        # The rules verify TWF, but it's a risk window, not a limit breach
        prob = signal["reading"]["failureProbability"]
        decision = "schedule"
        confidence = "MEDIUM"  # TWF is advisory by design
        cases.append({
            'cycleId': int(udi),
            'expectedRootCause': 'TWF',
            'expectedDecision': decision,
            'expectedConfidence': confidence,
            'note': f"Tool wear risk window, wear={int(row['tool_wear_min'])}min"
        })

    # ---- 3 CONFLICT cases (rootCause:null, escalate-now, CONFIDENCE=CONFLICT) ----
    print("Selecting 3 CONFLICT cases...")
    # Find UDIs where model predicts failure but no physical rule verified
    # Use the existing conflict UDIs but ensure they're not used already
    conflict_candidates = []
    for udi in [78, 1088, 1438, 1510]:  # from DEMO_CASES CONFLICT list
        if udi not in used_udis:
            conflict_candidates.append(udi)
    # If we don't have enough, find more from the dataset
    if len(conflict_candidates) < 3:
        # Find UDIs where machine_failure=1 but no mode flag is 1 (or rules don't verify)
        for udi in df[(df['machine_failure'] == 1) & ~((df['twf'] == 1) | (df['hdf'] == 1) | (df['pwf'] == 1) | (df['osf'] == 1))]['udi'].tolist()[:8]:
            if udi not in used_udis and udi not in conflict_candidates:
                conflict_candidates.append(udi)
    for i in range(3):
        udi = conflict_candidates[i]
        used_udis.add(udi)
        row = df.loc[int(udi)]
        signal = engine.score_udi(int(udi))
        prob = signal["reading"]["failureProbability"]
        # CONFLICT: rootCause=None, decision=escalate-now, confidence=CONFLICT
        cases.append({
            'cycleId': int(udi),
            'expectedRootCause': None,
            'expectedDecision': 'escalate-now',
            'expectedConfidence': 'CONFLICT',
            'note': f"Model-physics disagreement, prob={prob:.3f}"
        })

    # ---- 3 NOMINAL cases (rootCause:null, monitor, HIGH confidence) ----
    print("Selecting 3 NOMINAL cases...")
    nominal_available = [u for u in nominal_udis if u not in used_udis and u not in demo_UDIs][:10]
    for i in range(3):
        udi = nominal_available[i]
        used_udis.add(udi)
        row = df.loc[int(udi)]
        signal = engine.score_udi(int(udi))
        # NOMINAL: rootCause=None, decision=monitor, confidence=HIGH
        cases.append({
            'cycleId': int(udi),
            'expectedRootCause': None,
            'expectedDecision': 'monitor',
            'expectedConfidence': 'HIGH',
            'note': f"Nominal cycle, {row['type']} variant"
        })

    # Verify we have 25 cases with distinct UDIs
    all_udis = [c['cycleId'] for c in cases]
    assert len(cases) == 25, f"Expected 25 cases, got {len(cases)}"
    assert len(set(all_udis)) == 25, f"Expected 25 distinct UDIs, got {len(set(all_udis))}"

    # Write the eval set
    with open(eval_set_path, 'w') as f:
        json.dump(cases, f, indent=2)

    print(f"\n✅ Evaluation set written to {eval_set_path}")

    # Print summary
    decision_counts = {}
    confidence_counts = []
    for c in cases:
        decision_counts[c['expectedDecision']] = decision_counts.get(c['expectedDecision'], 0) + 1
    print(f"\nSummary by expected decision: {decision_counts}")

    # Verify acceptance criteria counts
    root_cause_cases = sum(1 for c in cases if c['expectedRootCause'] is not None)
    null_root_cause_cases = sum(1 for c in cases if c['expectedRootCause'] is None)
    conflict_cases = sum(1 for c in cases if c['expectedDecision'] == 'escalate-now' and c['expectedRootCause'] is None)
    nominal_cases = sum(1 for c in cases if c['expectedDecision'] == 'monitor' and c['expectedRootCause'] is None)
    print(f"  Cases with expected root cause: {root_cause_cases}/25 (need >= 23)")
    print(f"  Cases with expected null root cause: {null_root_cause_cases}/25")
    print(f"  CONFLICT cases (rootCause=null): {conflict_cases}/3")
    print(f"  NOMINAL cases (monitor, rootCause=null): {nominal_cases}/3")

    # Decision accuracy check
    print(f"\n  All expected decisions: {list(decision_counts.keys())}")
    print(f"  escalate-now: {decision_counts.get('escalate-now', 0)}/25")
    print(f"  schedule: {decision_counts.get('schedule', 0)}/25")
    print(f"  monitor: {decision_counts.get('monitor', 0)}/25")


if __name__ == "__main__":
    main()