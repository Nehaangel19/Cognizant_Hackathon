"""
Score the physics rules engine against the dataset's ground-truth mode flags.

OWNER: Dev 1
RUN:   python src/validate_rules.py

WHY THIS EXISTS
---------------
Claiming "our rules encode the real physics" is worth nothing on its own. This
script turns it into a number: for each mode, how many of the true failures did
the rule catch, and how many rows did it flag that were not failures.

That single table is probably the strongest evidence slide in the deck. It is
also how you find out your threshold is wrong at hour 9 instead of hour 34.

WHAT GOOD LOOKS LIKE
--------------------
    HDF   precision 1.00   recall 1.00     (documented deterministic rule)
    PWF   precision 1.00   recall 1.00     (documented deterministic rule)
    OSF   precision 1.00   recall 1.00     (documented deterministic rule)
    TWF   precision ~0.05  recall ~0.93    (EXPECTED - failure point is random
                                            inside the 200-240 min window, so no
                                            deterministic rule can be precise)

If HDF, PWF or OSF come in below 1.00, your threshold or your AND/OR is wrong.
Anything less than perfect on those three is a bug, not a modelling limitation.
Skipping a mode you have not implemented yet is handled - it just prints TODO.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from features import build_feature_matrix
from rules_engine import check_hdf, check_osf, check_pwf, check_twf

CHECKS = {
    "TWF": check_twf,
    "HDF": check_hdf,
    "PWF": check_pwf,
    "OSF": check_osf,
}


def score_mode(df: pd.DataFrame, mode: str) -> dict | None:
    """Run one rule across all 10,000 rows. Returns None if not implemented."""
    fn = CHECKS[mode]
    try:
        fn(df.iloc[0])
    except NotImplementedError:
        return None

    predicted = df.apply(lambda r: fn(r).triggered, axis=1)
    actual = df[mode.lower()].astype(bool)

    tp = int((predicted & actual).sum())
    fp = int((predicted & ~actual).sum())
    fn_ = int((~predicted & actual).sum())
    tn = int((~predicted & ~actual).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn_) if (tp + fn_) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {"mode": mode, "actual": int(actual.sum()), "flagged": int(predicted.sum()),
            "tp": tp, "fp": fp, "fn": fn_, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1}


def main() -> int:
    _, _, df = build_feature_matrix()
    print(f"Scoring physics rules over {len(df):,} rows\n")

    header = f"{'MODE':<6}{'TRUE':>6}{'FLAG':>6}{'TP':>5}{'FP':>6}{'FN':>5}{'PREC':>8}{'REC':>8}{'F1':>8}"
    print(header)
    print("-" * len(header))

    results, todo = [], []
    for mode in ["TWF", "HDF", "PWF", "OSF"]:
        r = score_mode(df, mode)
        if r is None:
            todo.append(mode)
            print(f"{mode:<6}{'':>6}{'':>6}{'':>5}{'':>6}{'':>5}{'  NOT IMPLEMENTED':>30}")
            continue
        results.append(r)
        print(f"{r['mode']:<6}{r['actual']:>6}{r['flagged']:>6}{r['tp']:>5}"
              f"{r['fp']:>6}{r['fn']:>5}{r['precision']:>8.3f}{r['recall']:>8.3f}{r['f1']:>8.3f}")

    print()
    exit_code = 0
    for r in results:
        if r["mode"] == "TWF":
            print(f"TWF: precision {r['precision']:.2f} is EXPECTED to be low - the failure "
                  f"point is drawn at random inside the {r['flagged']}-row wear window. "
                  "Report this honestly; do not tune it away.")
        elif r["precision"] < 1.0 or r["recall"] < 1.0:
            print(f"FAIL {r['mode']}: deterministic rule should be perfect. "
                  f"{r['fp']} false positives, {r['fn']} misses. Check the threshold "
                  "and whether the conditions are AND or OR.")
            exit_code = 1
        else:
            print(f"PASS {r['mode']}: recovers all {r['actual']} true cases with zero "
                  "false positives. Quotable in the pitch.")

    if todo:
        print(f"\nStill to implement (Dev 1): {', '.join(todo)} - see src/rules_engine.py")

    # Coverage: what share of ALL machine failures do the verified rules explain?
    if len(results) == 4:
        pred_any = df.apply(
            lambda r: any(CHECKS[m](r).triggered for m in ["HDF", "PWF", "OSF"]), axis=1)
        failures = df["machine_failure"].astype(bool)
        covered = int((pred_any & failures).sum())
        print(f"\nCoverage: the deterministic rules (HDF/PWF/OSF) explain "
              f"{covered}/{int(failures.sum())} of all machine failures "
              f"({covered / failures.sum():.1%}). The remainder are tool-wear and "
              f"random (RNF) events - {int(df['rnf'].sum())} RNF rows are irreducible "
              "by construction and belong on the limitations slide.")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
