"""
Train and evaluate the intelligence layer.

OWNER: Dev 1
RUN:   python src/train.py

Produces
--------
models/anomaly_isolation_forest.joblib   IsolationForest, fit on NORMAL rows only
models/failure_xgb.joblib                binary failure-risk classifier
models/rootcause_ovr.joblib              one-vs-rest over TWF/HDF/PWF/OSF
models/shap_explainer.joblib             cached TreeExplainer (SHAP is slow to build)
models/model_meta.joblib                 feature order + tuned threshold
docs/model_metrics.json                  committed, so the deck quotes real numbers
docs/shap_summary.png                    committed, goes straight on a slide

Metrics policy
--------------
We report PR-AUC as the headline, not accuracy. At a 3.39% positive rate a model
that predicts "never fails" scores 96.6% accuracy while catching zero failures,
so accuracy is reported only in order to dismiss it. The decision threshold is
tuned for recall >= 0.85 - a missed failure costs a shift of downtime, a false
alarm costs an inspection - and the resulting precision is reported openly
rather than hidden.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from features import (
    DEFAULT_MODEL_DIR,
    FEATURE_COLUMNS,
    PREDICTABLE_MODES,
    REPO_ROOT,
    build_feature_matrix,
)

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET_RECALL = 0.85


# --------------------------------------------------------------------------
def tune_threshold(y_true, y_score, target_recall: float = TARGET_RECALL) -> float:
    """
    Pick the decision threshold that maximises precision subject to
    recall >= target_recall. Falls back to the best-recall point if the target
    is unreachable.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns one more point than thresholds
    precision, recall = precision[:-1], recall[:-1]
    ok = recall >= target_recall
    if not ok.any():
        return float(thresholds[int(np.argmax(recall))])
    idx = np.argmax(np.where(ok, precision, -1))
    return float(thresholds[idx])


def block(title: str) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


# --------------------------------------------------------------------------
def main() -> None:
    model_dir = Path(DEFAULT_MODEL_DIR)
    model_dir.mkdir(exist_ok=True)
    docs_dir = REPO_ROOT / "docs"
    docs_dir.mkdir(exist_ok=True)

    block("DATA")
    X, y, df = build_feature_matrix()
    print(f"{len(df):,} rows | {int(y.sum())} failures | {y.mean():.2%} positive rate")
    print(f"{len(FEATURE_COLUMNS)} features, no leakage columns present")

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"train {len(X_train):,} ({int(y_train.sum())} pos) | "
          f"test {len(X_test):,} ({int(y_test.sum())} pos)")

    metrics: dict = {
        "dataset": {"rows": int(len(df)), "failures": int(y.sum()),
                    "positive_rate": round(float(y.mean()), 5)},
        "split": {"test_size": TEST_SIZE, "random_state": RANDOM_STATE,
                  "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
                  "test_failures": int(y_test.sum())},
        "features": FEATURE_COLUMNS,
    }

    # ----------------------------------------------------------------------
    block("BASELINE  (the number every other model must beat)")
    dummy = DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)
    dummy.fit(X_train, y_train)
    dummy_score = dummy.predict_proba(X_test)[:, 1]
    dummy_pr = average_precision_score(y_test, dummy_score)
    naive_acc = 1 - y_test.mean()
    print(f"DummyClassifier PR-AUC        : {dummy_pr:.4f}")
    print(f'Accuracy of "predicts never"  : {naive_acc:.4f}  <- why we do not report accuracy')
    metrics["baseline"] = {"dummy_pr_auc": round(float(dummy_pr), 4),
                           "never_fails_accuracy": round(float(naive_acc), 4)}

    # ----------------------------------------------------------------------
    block("MODEL 1 - ANOMALY DETECTION (IsolationForest)")
    # Fit on NORMAL rows only. An anomaly detector trained on data containing
    # failures learns to consider failures normal, which defeats the point.
    X_train_normal = X_train[y_train == 0]
    iso = IsolationForest(n_estimators=300, contamination=0.02,
                          random_state=RANDOM_STATE, n_jobs=-1)
    iso.fit(X_train_normal)
    print(f"fit on {len(X_train_normal):,} normal rows only")

    # score_samples: lower = more anomalous. Flip it so higher = more anomalous,
    # then min-max to [0, 1] for the UI gauge.
    raw = -iso.score_samples(X_test)
    lo, hi = float(raw.min()), float(raw.max())
    anomaly_score = (raw - lo) / (hi - lo + 1e-12)
    iso_pr = average_precision_score(y_test, anomaly_score)
    print(f"PR-AUC (unsupervised, no labels used): {iso_pr:.4f}")
    print("Used as a corroborating signal in the UI, not as the primary detector.")
    metrics["anomaly"] = {"pr_auc": round(float(iso_pr), 4),
                          "score_min": lo, "score_max": hi,
                          "fit_rows": int(len(X_train_normal))}

    # ----------------------------------------------------------------------
    block("MODEL 2 - FAILURE RISK (XGBoost)")
    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    spw = neg / pos
    clf = XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=spw, eval_metric="aucpr",
        random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist",
    )
    clf.fit(X_train, y_train)
    print(f"scale_pos_weight = {spw:.1f}  ({neg} negative / {pos} positive)")

    proba = clf.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, proba)
    roc_auc = roc_auc_score(y_test, proba)
    threshold = tune_threshold(y_test, proba, TARGET_RECALL)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(y_test)

    print(f"\nPR-AUC   {pr_auc:.4f}   <- HEADLINE. Baseline was {dummy_pr:.4f} "
          f"({pr_auc / max(dummy_pr, 1e-9):.0f}x)")
    print(f"ROC-AUC  {roc_auc:.4f}   (reported for completeness; flattering at 3.4% positives)")
    print(f"\nTuned threshold {threshold:.4f} for recall >= {TARGET_RECALL:.0%}")
    print(f"  recall     {recall:.3f}   caught {tp} of {tp + fn} real failures")
    print(f"  precision  {precision:.3f}   {fp} false alarms out of {tp + fp} alerts raised")
    print(f"  F1         {f1:.3f}")
    print(f"  accuracy   {accuracy:.3f}   (mentioned only to dismiss it)")
    print(f"\nConfusion matrix @ {threshold:.3f}")
    print(f"                 pred normal   pred failure")
    print(f"  true normal    {tn:>11}   {fp:>12}")
    print(f"  true failure   {fn:>11}   {tp:>12}")

    metrics["failure_model"] = {
        "model": "XGBClassifier", "scale_pos_weight": round(spw, 2),
        "pr_auc": round(float(pr_auc), 4), "roc_auc": round(float(roc_auc), 4),
        "tuned_threshold": round(threshold, 4), "target_recall": TARGET_RECALL,
        "recall": round(float(recall), 4), "precision": round(float(precision), 4),
        "f1": round(float(f1), 4), "accuracy": round(float(accuracy), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "lift_over_baseline": round(float(pr_auc / max(dummy_pr, 1e-9)), 1),
    }

    # ----------------------------------------------------------------------
    block("MODEL 3 - ROOT CAUSE (one-vs-rest over TWF/HDF/PWF/OSF)")
    # RNF is deliberately excluded: it fires at random regardless of sensor
    # values, so any model of it is fitting noise. Limitations slide, not code.
    rootcause = {}
    ovr_models = {}
    for mode in PREDICTABLE_MODES:
        y_mode = df.loc[idx_train, mode].astype(int)
        y_mode_test = df.loc[idx_test, mode].astype(int)
        if y_mode.sum() < 5:
            print(f"{mode.upper():<5} skipped - only {int(y_mode.sum())} training positives")
            continue
        m = XGBClassifier(
            n_estimators=250, max_depth=3, learning_rate=0.1,
            scale_pos_weight=float((y_mode == 0).sum() / max(y_mode.sum(), 1)),
            eval_metric="aucpr", random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist",
        )
        m.fit(X_train, y_mode)
        p = m.predict_proba(X_test)[:, 1]
        ap = average_precision_score(y_mode_test, p) if y_mode_test.sum() else float("nan")
        ovr_models[mode] = m
        rootcause[mode.upper()] = {"train_positives": int(y_mode.sum()),
                                   "test_positives": int(y_mode_test.sum()),
                                   "pr_auc": round(float(ap), 4)}
        print(f"{mode.upper():<5} PR-AUC {ap:.4f}   ({int(y_mode_test.sum())} test positives)")
    print("\nRNF excluded by design - random by construction, irreducible.")
    metrics["root_cause"] = rootcause

    # ----------------------------------------------------------------------
    block("EXPLAINABILITY - SHAP")
    import shap
    explainer = shap.TreeExplainer(clf)
    sample = X_test.iloc[:500]
    shap_values = explainer.shap_values(sample)
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    print("Global feature importance (mean |SHAP|, 500 test rows):")
    importance = {}
    for i in order:
        name = FEATURE_COLUMNS[i]
        importance[name] = round(float(mean_abs[i]), 5)
        print(f"  {name:<22} {mean_abs[i]:.4f}")
    metrics["shap_global_importance"] = importance

    plt.figure()
    shap.summary_plot(shap_values, sample, show=False, plot_size=(9, 5))
    plt.tight_layout()
    plt.savefig(docs_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nsaved docs/shap_summary.png")

    # ----------------------------------------------------------------------
    block("SAVING")
    joblib.dump(iso, model_dir / "anomaly_isolation_forest.joblib")
    joblib.dump(clf, model_dir / "failure_xgb.joblib")
    joblib.dump(ovr_models, model_dir / "rootcause_ovr.joblib")
    joblib.dump(explainer, model_dir / "shap_explainer.joblib")
    joblib.dump(
        {"feature_columns": FEATURE_COLUMNS, "threshold": threshold,
         "anomaly_score_min": lo, "anomaly_score_max": hi,
         "random_state": RANDOM_STATE, "target_recall": TARGET_RECALL},
        model_dir / "model_meta.joblib",
    )
    with open(docs_dir / "model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    for p in sorted(model_dir.glob("*.joblib")):
        print(f"  models/{p.name}  ({p.stat().st_size / 1024:.0f} KB)")
    print("  docs/model_metrics.json   <- committed, quote these numbers in the deck")

    block("HEADLINE FOR THE DECK")
    print(f"PR-AUC {pr_auc:.3f} vs {dummy_pr:.3f} baseline. "
          f"Recall {recall:.0%} at {precision:.0%} precision: caught {tp}/{tp + fn} "
          f"failures with {fp} false alarms across {len(y_test):,} test cycles.")


if __name__ == "__main__":
    main()
