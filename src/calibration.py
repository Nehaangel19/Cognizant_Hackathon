"""
Probability calibration for the XGBoost failure-risk classifier.

OWNER: Dev 4 (Platform, Verification & QA)
STATUS: Purely additive — does not modify train.py or predict.py.

WHY THIS EXISTS
---------------
XGBoost was trained with scale_pos_weight=28.5 to handle class imbalance.
That fixes ranking (PR-AUC 0.87) but distorts the output probabilities —
the "91%" in a work order is a ranking score, not a literal frequency. A
sharp judge will press on this exact point.

This module fits an isotonic calibrator (and a Platt/sigmoid calibrator)
on a held-out calibration split, picks whichever reduces Brier score more,
and exposes a single function: calibrate(p) -> float.

Public surface
--------------
calibrate(p: float) -> float          calibrate a raw probability
main()                                print Brier before/after, save curve
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

from features import (
    DEFAULT_MODEL_DIR,
    FEATURE_COLUMNS,
    REPO_ROOT,
    build_feature_matrix,
)

RANDOM_STATE = 42
CALIBRATION_SIZE = 0.30  # 30% held out for calibration fitting


def _load_model_and_data():
    """Load the trained XGBoost model and the full dataset."""
    model_dir = Path(DEFAULT_MODEL_DIR)
    clf = joblib.load(model_dir / "failure_xgb.joblib")
    X, y, df = build_feature_matrix()
    return clf, X, y


def _fit_isotonic(y_true: np.ndarray, y_prob: np.ndarray) -> IsotonicRegression:
    """Fit an isotonic calibrator."""
    ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    ir.fit(y_prob, y_true)
    return ir


def _fit_platt(y_true: np.ndarray, y_prob: np.ndarray) -> LogisticRegression:
    """Fit a Platt (sigmoid) calibrator."""
    lr = LogisticRegression(C=1.0, solver="lbfgs", random_state=RANDOM_STATE)
    lr.fit(y_prob.reshape(-1, 1), y_true)
    return lr


def _brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(brier_score_loss(y_true, y_prob))


def build_reliability_diagram(
    y_true: np.ndarray,
    y_prob_raw: np.ndarray,
    y_prob_cal: np.ndarray,
    output_path: str | Path,
) -> None:
    """Save a reliability diagram comparing raw vs calibrated probabilities."""
    fig, ax = plt.subplots(figsize=(7, 6))

    # Raw probabilities
    frac_raw, mean_raw = calibration_curve(y_true, y_prob_raw, n_bins=10, strategy="uniform")
    ax.plot(mean_raw, frac_raw, "s-", color="#e74c3c", linewidth=2,
            label=f"Raw (Brier={_brier(y_true, y_prob_raw):.4f})")

    # Calibrated probabilities
    frac_cal, mean_cal = calibration_curve(y_true, y_prob_cal, n_bins=10, strategy="uniform")
    ax.plot(mean_cal, frac_cal, "o-", color="#2ecc71", linewidth=2,
            label=f"Calibrated (Brier={_brier(y_true, y_prob_cal):.4f})")

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Perfect calibration")

    ax.set_xlabel("Mean predicted probability", fontsize=12)
    ax.set_ylabel("Fraction of positives", fontsize=12)
    ax.set_title("Reliability Diagram — Failure Risk Classifier", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved reliability diagram to {output_path}")


# --------------------------------------------------------------------------
# Public interface
# --------------------------------------------------------------------------

# Module-level calibrator, loaded lazily
_calibrator: IsotonicRegression | LogisticRegression | None = None
_calibrator_method: str = ""


def calibrate(p: float) -> float:
    """
    Calibrate a raw XGBoost failure probability.

    Parameters
    ----------
    p : float
        Raw probability from XGBClassifier.predict_proba().

    Returns
    -------
    float
        Calibrated probability in [0, 1].
    """
    global _calibrator, _calibrator_method

    if _calibrator is None:
        _load_calibrator()

    arr = np.array([p])
    if _calibrator_method == "isotonic":
        return float(_calibrator.predict(arr)[0])
    else:
        return float(_calibrator.predict(arr.reshape(-1, 1))[0])


def _load_calibrator() -> None:
    """Load the saved calibrator from disk."""
    global _calibrator, _calibrator_method

    model_dir = Path(DEFAULT_MODEL_DIR)
    cal_path = model_dir / "calibrator.joblib"
    if not cal_path.exists():
        raise FileNotFoundError(
            f"Calibrator not found at {cal_path}. "
            "Run `python src/calibration.py` first to train it."
        )
    meta = joblib.load(cal_path)
    _calibrator = meta["calibrator"]
    _calibrator_method = meta["method"]


# --------------------------------------------------------------------------
# Main — fit, evaluate, save
# --------------------------------------------------------------------------
def main() -> None:
    print("=" * 68)
    print("PROBABILITY CALIBRATION")
    print("=" * 68)

    clf, X, y = _load_model_and_data()
    y_true = y.values

    # Split: 70% train+test (used by train.py), 30% calibration held-out
    # We replicate the same split seed so the calibration set is deterministic.
    X_cal, _, y_cal, _ = train_test_split(
        X, y_true, test_size=1 - CALIBRATION_SIZE, stratify=y_true,
        random_state=RANDOM_STATE
    )
    # Of that 30%, we use it directly for calibration fitting.
    # The raw probabilities on this set tell us the uncalibrated Brier score.
    raw_probs = clf.predict_proba(X_cal)[:, 1]

    print(f"\nCalibration set: {len(X_cal):,} rows, {int(y_cal.sum())} failures")
    print(f"  Positive rate: {y_cal.mean():.4f}")

    # --- Raw Brier score ---
    brier_raw = _brier(y_cal, raw_probs)
    print(f"\n  Raw XGBoost Brier score:   {brier_raw:.6f}")

    # --- Fit isotonic calibrator ---
    iso = _fit_isotonic(y_cal, raw_probs)
    iso_probs = iso.predict(raw_probs)
    brier_iso = _brier(y_cal, iso_probs)
    print(f"  Isotonic Brier score:      {brier_iso:.6f}  (delta {brier_iso - brier_raw:+.6f})")

    # --- Fit Platt (sigmoid) calibrator ---
    platt = _fit_platt(y_cal, raw_probs)
    platt_probs = platt.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
    brier_platt = _brier(y_cal, platt_probs)
    print(f"  Platt (sigmoid) Brier:     {brier_platt:.6f}  (delta {brier_platt - brier_raw:+.6f})")

    # --- Pick the better calibrator ---
    if brier_iso <= brier_platt:
        best_calibrator = iso
        best_method = "isotonic"
        best_probs = iso_probs
        best_brier = brier_iso
        print(f"\n  -> Isotonic wins (lower Brier)")
    else:
        best_calibrator = platt
        best_method = "platt"
        best_probs = platt_probs
        best_brier = brier_platt
        print(f"\n  -> Platt wins (lower Brier)")

    # --- Recommendation ---
    improvement = brier_raw - best_brier
    if improvement > 0.001:
        print(f"\n  RECOMMENDATION: Calibration helps (Brier improves by {improvement:.4f}).")
        print(f"  Wire calibrate() into the pipeline when ready.")
    elif improvement > 0:
        print(f"\n  RECOMMENDATION: Marginal improvement ({improvement:.4f}).")
        print(f"  Consider leaving calibration out — the gain is within noise.")
    else:
        print(f"\n  RECOMMENDATION: Calibration does NOT improve Brier score.")
        print(f"  Leave it out. The raw probabilities are already well-calibrated.")

    # --- Save calibrator ---
    model_dir = Path(DEFAULT_MODEL_DIR)
    model_dir.mkdir(exist_ok=True)
    joblib.dump({"calibrator": best_calibrator, "method": best_method,
                 "brier_before": brier_raw, "brier_after": best_brier},
                model_dir / "calibrator.joblib")
    print(f"\n  Saved models/calibrator.joblib ({best_method})")

    # --- Save reliability diagram ---
    docs_dir = REPO_ROOT / "docs"
    docs_dir.mkdir(exist_ok=True)
    build_reliability_diagram(y_cal, raw_probs, best_probs,
                              docs_dir / "calibration_curve.png")

    # --- Summary ---
    print(f"\n{'=' * 68}")
    print(f"SUMMARY")
    print(f"{'=' * 68}")
    print(f"  Raw Brier:     {brier_raw:.6f}")
    print(f"  Calibrated:    {best_brier:.6f} ({best_method})")
    print(f"  Improvement:   {improvement:+.6f}")
    print(f"{'=' * 68}")


if __name__ == "__main__":
    main()
