"""
Inference wrapper - the single entry point between the models and everything else.

OWNER: Dev 1
CONSUMED BY: Dev 2 (src/agent/tools.py), Dev 3 (replay simulator / dashboard)

WHY THIS EXISTS
---------------
Without it, Dev 2 and Dev 3 each have to load five joblib artefacts, remember the
tuned threshold, normalise the anomaly score the same way training did, and drive
SHAP by hand. Three copies of that logic drift apart by hour 20 and the demo shows
one number while the eval slide shows another.

Instead: one object, one call, one dict that already matches the frontend data
contract.

    from predict import Engine
    engine = Engine()                       # loads models once, ~1s
    signal = engine.score_udi(2153)         # everything about one reading

Dev 2 - your agent tools map onto this directly:
    detect_anomaly()       -> signal["reading"]["anomalyScore"]
    predict_failure_prob() -> signal["reading"]["failureProbability"]
    diagnose_root_cause()  -> signal["rootCause"], signal["shapValues"]
    check_physics_rules()  -> signal["ruleChecks"], signal["confidence"]

The two fields this file deliberately does NOT fill are `recommendedActions` and
`partsRequired`. Those come from the maintenance playbook, which is Non-tech A's
deliverable and Dev 2's lookup. Inventing them here would be exactly the
hallucination the rules engine exists to prevent.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd

from features import (
    DEFAULT_MODEL_DIR,
    FEATURE_COLUMNS,
    build_feature_matrix,
    row_to_reading,
)
from rules_engine import assess

# Severity ladder, mirroring the `Severity` type in the frontend data contract.
# Driven by BOTH the model and the rules, never by probability alone - a verified
# limit breach outranks a merely high score, and a high score with no verified
# cause is a WARNING to be investigated, not a CRITICAL to stop the line for.
CRITICAL_PROBABILITY = 0.70

# Pre-seeded demo cases, found by scanning the full dataset once. Non-tech B:
# these are the UDIs to hard-code into the demo script so the escalation beat
# lands on cue instead of being hunted for live on stage. Every one is a real
# failure with a matching ground-truth flag - nothing here is fabricated.
DEMO_CASES = {
    # Overstrain - the beat the 3-minute script escalates on.
    "OSF": [70, 161, 162, 243],
    # Power failure - UDI 51 is the cleanest: SHAP's top feature is power_w and
    # the PWF rule fires on the same quantity. Model and physics point at one thing.
    "PWF": [51, 169, 195, 208],
    "HDF": [3237, 3761, 3788, 3794],
    # Model is >96% sure, no physical limit breached. The agent refuses to name a
    # cause and escalates instead. This is the "it does not bluff" beat.
    "CONFLICT": [78, 1088, 1438, 1510],
}


class Engine:
    """Loads the trained artefacts once and scores readings."""

    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR, data_path=None):
        model_dir = Path(model_dir)
        missing = [n for n in ("failure_xgb", "anomaly_isolation_forest",
                               "model_meta", "shap_explainer")
                   if not (model_dir / f"{n}.joblib").exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing model artefacts {missing} in {model_dir}. "
                "Run `python src/train.py` first (~30s)."
            )

        self.clf = joblib.load(model_dir / "failure_xgb.joblib")
        self.iso = joblib.load(model_dir / "anomaly_isolation_forest.joblib")
        self.explainer = joblib.load(model_dir / "shap_explainer.joblib")
        self.meta = joblib.load(model_dir / "model_meta.joblib")
        self.threshold: float = self.meta["threshold"]
        self._anom_lo: float = self.meta["anomaly_score_min"]
        self._anom_hi: float = self.meta["anomaly_score_max"]

        # The engineered frame, kept in memory for replay and UDI lookup.
        _, _, self.df = build_feature_matrix(data_path) if data_path else build_feature_matrix()
        self.df = self.df.set_index("udi", drop=False)

    # ------------------------------------------------------------------
    def _anomaly_score(self, X: pd.DataFrame) -> float:
        raw = float(-self.iso.score_samples(X)[0])
        scaled = (raw - self._anom_lo) / (self._anom_hi - self._anom_lo + 1e-12)
        return float(np.clip(scaled, 0.0, 1.0))

    def _shap_top(self, X: pd.DataFrame, k: int = 5) -> list[dict]:
        """Top-k features pushing this specific reading's risk, signed."""
        values = self.explainer.shap_values(X)[0]
        order = np.argsort(np.abs(values))[::-1][:k]
        return [
            {
                "feature": FEATURE_COLUMNS[i],
                "value": round(float(X.iloc[0, i]), 3),
                "contribution": round(float(values[i]), 4),
            }
            for i in order
        ]

    @staticmethod
    def _severity(probability: float, confidence: str, root_cause: str | None) -> str:
        if confidence == "HIGH" and root_cause and probability >= CRITICAL_PROBABILITY:
            return "CRITICAL"
        if confidence == "HIGH" and root_cause:
            return "WARNING"
        if confidence == "CONFLICT":
            return "WARNING"      # unexplained, so a human looks at it
        if confidence == "MEDIUM":
            return "ADVISORY"     # tool-wear window: schedule, do not stop
        return "NOMINAL"

    # ------------------------------------------------------------------
    def score(self, row: pd.Series) -> dict:
        """
        Score one engineered row. Returns a dict the frontend can render and the
        agent can reason over, with no further post-processing.
        """
        X = row[FEATURE_COLUMNS].to_frame().T.astype(float)

        probability = float(self.clf.predict_proba(X)[0, 1])
        anomaly = self._anomaly_score(X)
        verdict = assess(row, probability, self.threshold)
        severity = self._severity(probability, verdict["confidence"], verdict["rootCause"])

        reading = row_to_reading(row)
        reading["anomalyScore"] = round(anomaly, 4)
        reading["failureProbability"] = round(probability, 4)

        # Evidence: the verified physical breach first, then the model's own
        # top attribution. Ordered this way on purpose - physics leads, the
        # statistical model corroborates.
        shap_values = self._shap_top(X)
        evidence: list[str] = []
        for check in verdict["ruleChecks"]:
            if check["triggered"]:
                evidence.append(f"{check['ruleName']}: {check['measuredValue']}")
        if shap_values:
            top = shap_values[0]
            direction = "raises" if top["contribution"] > 0 else "lowers"
            evidence.append(
                f"SHAP: {top['feature']} = {top['value']} {direction} risk by "
                f"{abs(top['contribution']):.2f}"
            )

        return {
            "reading": reading,
            "severity": severity,
            "confidence": verdict["confidence"],
            "rootCause": verdict["rootCause"],
            "rootCauseLabel": verdict["rootCauseLabel"],
            "rationale": verdict["rationale"],
            "ruleChecks": verdict["ruleChecks"],
            "shapValues": shap_values,
            "evidence": evidence,
            "modelThreshold": round(self.threshold, 4),
            # Filled by Dev 2 from the maintenance playbook + cost model:
            "recommendedActions": [],
            "partsRequired": [],
        }

    def score_udi(self, udi: int) -> dict:
        """Score one row by its UDI (the replay index)."""
        if udi not in self.df.index:
            raise KeyError(f"UDI {udi} not in dataset (valid range 1-{len(self.df)}).")
        return self.score(self.df.loc[udi])

    def replay(self, start: int = 1, end: int | None = None):
        """
        Generator over readings in UDI order - the simulated live stream.

        Dev 3: drive the dashboard from this and sleep between yields for the
        ~2 rows/sec replay rate. No timestamps exist in the data; UDI is the
        stand-in, and we say so in the pitch.
        """
        end = end or len(self.df)
        for udi in range(start, end + 1):
            yield self.score_udi(udi)

    def find_demo_cases(self, severity: str = "CRITICAL", limit: int = 5,
                        scan_limit: int = 1000) -> list[int]:
        """
        UDIs that produce a given severity.

        Scoring builds a SHAP explanation per row, so a full 10,000-row sweep
        takes minutes. `scan_limit` caps how many rows are examined. For the
        demo you want the pre-computed DEMO_CASES constant above, not this.
        """
        hits = []
        for udi in list(self.df.index)[:scan_limit]:
            if self.score_udi(int(udi))["severity"] == severity:
                hits.append(int(udi))
                if len(hits) >= limit:
                    break
        return hits


if __name__ == "__main__":
    import json

    engine = Engine()
    print(f"Engine ready. Threshold {engine.threshold:.4f}\n")

    print("Pre-seeded demo cases:")
    for mode, udis in DEMO_CASES.items():
        print(f"  {mode:<9} {udis}")

    print(f"\n--- UDI {DEMO_CASES['OSF'][0]}: the overstrain escalation beat ---")
    signal = engine.score_udi(DEMO_CASES["OSF"][0])
    print(f"severity   {signal['severity']}")
    print(f"confidence {signal['confidence']}")
    print(f"cause      {signal['rootCauseLabel']}")
    print("evidence:")
    for line in signal["evidence"]:
        print(f"  - {line}")

    print(f"\n--- UDI {DEMO_CASES['CONFLICT'][0]}: the agent refuses to bluff ---")
    c = engine.score_udi(DEMO_CASES["CONFLICT"][0])
    print(f"severity {c['severity']} | confidence {c['confidence']} | cause {c['rootCause']}")
    print(f"  {c['rationale']}")

    print("\nFull signal payload for the OSF case:")
    print(json.dumps(signal, indent=2)[:900] + "\n  ...")
