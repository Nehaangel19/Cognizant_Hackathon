"""
Precompute demo data cache — scores all 10,000 rows once and writes static JSON.

OWNER: Dev 4 (Platform, Verification & QA)

WHY THIS EXISTS
---------------
Scoring all 10,000 rows live takes minutes because SHAP runs per row. Nobody
wants that lag mid-pitch. This script precomputes everything once and writes
static files that any downstream consumer (frontend, demo script, eval) can
read without recomputing.

Produces
--------
data/demo_cache.json     Full signal per cycle (10,000 entries)
data/work_orders.json    Agent output for every non-NOMINAL cycle

Run: python scripts/build_demo_cache.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

# Ensure src/ is on the path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from predict import Engine, DEMO_CASES  # noqa: E402


def _summary_row(signal: dict) -> dict:
    """Extract the summary fields we care about for counting."""
    return {
        "severity": signal["severity"],
        "confidence": signal["confidence"],
        "rootCause": signal["rootCause"],
    }


def main() -> None:
    t0 = time.time()
    data_dir = REPO_ROOT / "data"
    data_dir.mkdir(exist_ok=True)

    print("=" * 68)
    print("BUILDING DEMO CACHE")
    print("=" * 68)

    # --- Load engine ---
    print("\nLoading engine (models + dataset)...")
    t_load = time.time()
    engine = Engine()
    print(f"  Engine ready in {time.time() - t_load:.1f}s")

    # --- Score all 10,000 rows ---
    print(f"\nScoring all {len(engine.df):,} rows...")
    t_score = time.time()
    demo_cache = []
    work_orders = []

    for udi in engine.df.index:
        signal = engine.score_udi(int(udi))
        demo_cache.append(signal)

        if signal["severity"] != "NOMINAL":
            # Run the agent to get the full work order
            from agent.agent import MaintenanceAgent
            agent = MaintenanceAgent()
            result = agent.run(int(udi))
            work_orders.append(result["workOrder"])

    elapsed = time.time() - t_score
    print(f"  Scored {len(demo_cache):,} rows in {elapsed:.1f}s "
          f"({len(demo_cache) / max(elapsed, 0.01):.0f} rows/sec)")

    # --- Write demo_cache.json ---
    cache_path = data_dir / "demo_cache.json"
    with open(cache_path, "w") as f:
        json.dump(demo_cache, f)
    cache_size = cache_path.stat().st_size
    print(f"\n  Wrote {cache_path.name}: {cache_size / 1024 / 1024:.1f} MB "
          f"({len(demo_cache):,} entries)")

    # --- Write work_orders.json ---
    wo_path = data_dir / "work_orders.json"
    with open(wo_path, "w") as f:
        json.dump(work_orders, f)
    wo_size = wo_path.stat().st_size
    print(f"  Wrote {wo_path.name}: {wo_size / 1024:.0f} KB "
          f"({len(work_orders):,} non-NOMINAL cycles)")

    # --- Summary counts ---
    print(f"\n{'=' * 68}")
    print("SUMMARY")
    print(f"{'=' * 68}")

    severity_counts = Counter(r["severity"] for r in demo_cache)
    confidence_counts = Counter(r["confidence"] for r in demo_cache)
    cause_counts = Counter(r["rootCause"] for r in demo_cache)

    print("\n  By severity:")
    for sev in ["NOMINAL", "ADVISORY", "WARNING", "CRITICAL"]:
        print(f"    {sev:<12} {severity_counts.get(sev, 0):>6}")

    print("\n  By confidence:")
    for conf in ["HIGH", "MEDIUM", "CONFLICT"]:
        print(f"    {conf:<12} {confidence_counts.get(conf, 0):>6}")

    print("\n  By root cause:")
    for cause in ["OSF", "PWF", "HDF", "TWF", None]:
        label = cause or "None"
        print(f"    {label:<12} {cause_counts.get(cause, 0):>6}")

    # --- Diff against live output ---
    print(f"\n{'=' * 68}")
    print("CROSS-CHECK vs LIVE OUTPUT")
    print(f"{'=' * 68}")

    # Re-run validate_rules logic inline to compare
    from features import build_feature_matrix
    from rules_engine import check_hdf, check_osf, check_pwf, check_twf

    _, _, df = build_feature_matrix()
    CHECKS = {"TWF": check_twf, "HDF": check_hdf, "PWF": check_pwf, "OSF": check_osf}

    all_pass = True
    for mode, fn in CHECKS.items():
        predicted = df.apply(lambda r: fn(r).triggered, axis=1)
        actual = df[mode.lower()].astype(bool)
        tp = int((predicted & actual).sum())
        fp = int((predicted & ~actual).sum())
        fn_ = int((~predicted & actual).sum())
        flagged = int(predicted.sum())
        actual_count = int(actual.sum())

        # Count from our cache
        cache_flagged = sum(1 for r in demo_cache
                           if any(rc["mode"] == mode and rc["triggered"]
                                  for rc in r["ruleChecks"]))

        match = "MATCH" if cache_flagged == flagged else "MISMATCH"
        if match == "MISMATCH":
            all_pass = False
        print(f"  {mode}: live flagged={flagged}, cache flagged={cache_flagged} "
              f"actual={actual_count} -> {match}")

    # Check demo case UDI 70
    udi70 = next(r for r in demo_cache if r["reading"]["cycleId"] == 70)
    assert udi70["rootCause"] == "OSF", f"UDI 70 expected OSF, got {udi70['rootCause']}"
    assert udi70["severity"] == "CRITICAL", f"UDI 70 expected CRITICAL, got {udi70['severity']}"
    print(f"\n  UDI 70: severity={udi70['severity']}, cause={udi70['rootCause']} -> MATCH")

    # Check demo case UDI 78
    udi78 = next(r for r in demo_cache if r["reading"]["cycleId"] == 78)
    assert udi78["rootCause"] is None, f"UDI 78 expected None, got {udi78['rootCause']}"
    assert udi78["confidence"] == "CONFLICT", f"UDI 78 expected CONFLICT, got {udi78['confidence']}"
    print(f"  UDI 78: confidence={udi78['confidence']}, cause={udi78['rootCause']} -> MATCH")

    print(f"\n{'=' * 68}")
    if all_pass:
        print("ALL CROSS-CHECKS PASSED")
    else:
        print("WARNING: Some cross-checks failed — investigate before demo")
    print(f"{'=' * 68}")
    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
