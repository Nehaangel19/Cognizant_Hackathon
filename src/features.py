"""
Feature engineering for the AI4I 2020 predictive maintenance dataset.

OWNER: Dev 1
CONSUMED BY: src/train.py, src/rules_engine.py, src/agent/tools.py (Dev 2),
             src/app/ (Dev 3)

Everything downstream imports from here. If you need a new column, add it in
`add_engineered_features` and register it in FEATURE_COLUMNS - do not compute it
ad hoc in a notebook, or the model and the live demo will silently disagree.

The four engineered columns are not decoration. Each one is the exact quantity
that appears in the dataset's documented physics rule for a failure mode, which
is what lets SHAP attributions line up with `rules_engine.py` instead of
contradicting it.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root, resolved from this file's location, so every script works no matter
# which directory you launched it from (src/, notebooks/, repo root - all fine).
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "ai4i2020.csv"
DEFAULT_MODEL_DIR = REPO_ROOT / "models"

# --------------------------------------------------------------------------
# Column naming
# --------------------------------------------------------------------------
# The raw CSV uses names with units and spaces ("Air temperature [K]"), which
# are painful in code. We rename once, here, and never touch the raw names again.

RAW_TO_CLEAN = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf",
}

TARGET = "machine_failure"
MODE_COLUMNS = ["twf", "hdf", "pwf", "osf", "rnf"]

# Modes the model is allowed to predict. RNF is excluded deliberately: it fires
# with 0.1% probability regardless of sensor values, so it is irreducible noise.
# Trying to predict it adds false positives and nothing else. This goes on the
# limitations slide, not in the model.
PREDICTABLE_MODES = ["twf", "hdf", "pwf", "osf"]

# NEVER feed these to a model.
#   - mode flags: they ARE the target (machine_failure = OR of the five)
#   - udi: row index, pure leakage of dataset ordering
#   - product_id: encodes Type plus a serial number
LEAKAGE_COLUMNS = MODE_COLUMNS + [TARGET, "udi", "product_id"]

# --------------------------------------------------------------------------
# Physics constants - shared with rules_engine.py so there is ONE source of truth
# --------------------------------------------------------------------------
OSF_LIMITS_MIN_NM = {"L": 11_000, "M": 12_000, "H": 13_000}
TYPE_ENCODING = {"L": 0, "M": 1, "H": 2}

HDF_TEMP_DIFF_K = 8.6        # heat dissipation fails below this delta ...
HDF_SPEED_RPM = 1380         # ... AND below this rotational speed
PWF_POWER_MIN_W = 3_500      # power outside [min, max] -> power failure
PWF_POWER_MAX_W = 9_000
TWF_WEAR_MIN = 200           # tool wear failure window, minutes
TWF_WEAR_MAX = 240

# --------------------------------------------------------------------------
# Feature matrix definition
# --------------------------------------------------------------------------
BASE_FEATURES = [
    "air_temp_k",
    "process_temp_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]

ENGINEERED_FEATURES = [
    "temp_diff_k",       # HDF signal
    "power_w",           # PWF signal
    "wear_torque_min_nm",  # OSF signal
    "osf_margin",        # OSF signal, normalised by type-specific limit
    "type_encoded",      # L/M/H
]

FEATURE_COLUMNS = BASE_FEATURES + ENGINEERED_FEATURES


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_raw(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Read the CSV and apply clean column names. No feature engineering yet."""
    df = pd.read_csv(path, encoding="utf-8-sig")  # utf-8-sig eats a stray BOM
    missing = set(RAW_TO_CLEAN) - set(df.columns)
    if missing:
        raise ValueError(
            f"Unexpected schema in {path}. Missing columns: {sorted(missing)}. "
            "Did you download a different AI4I variant? See data/README.md."
        )
    return df.rename(columns=RAW_TO_CLEAN)


def verify_dataset(df: pd.DataFrame) -> dict[str, int]:
    """
    Fail loudly if the data is not the canonical AI4I 2020 file.

    Cheap insurance: every metric we quote to a judge depends on this being the
    file we think it is.
    """
    counts = {c: int(df[c].sum()) for c in [TARGET] + MODE_COLUMNS}
    expected = {"machine_failure": 339, "twf": 46, "hdf": 115,
                "pwf": 95, "osf": 98, "rnf": 19}
    if len(df) != 10_000:
        raise ValueError(f"Expected 10,000 rows, got {len(df):,}.")
    if counts != expected:
        raise ValueError(f"Label counts differ from canonical.\n"
                         f"  expected: {expected}\n  got:      {counts}")
    return counts


# --------------------------------------------------------------------------
# Engineering
# --------------------------------------------------------------------------
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the five derived columns. Returns a copy; the input is not mutated.

    temp_diff_k        process temp - air temp, in K.
                       The HDF rule triggers below 8.6 K.
    power_w            torque [Nm] x angular velocity [rad/s], in watts.
                       rpm -> rad/s is x(2*pi/60). The PWF rule triggers
                       outside [3500, 9000] W.
    wear_torque_min_nm tool wear [min] x torque [Nm].
                       The OSF rule triggers above a type-dependent limit.
    osf_margin         wear_torque / that limit. Dimensionless, so >= 1.0 means
                       "over the line" for L, M and H alike. This is the single
                       most interpretable number in the whole feature set - it
                       is what the agent quotes in its evidence string.
    type_encoded       L=0, M=1, H=2. Ordinal on purpose: the product variants
                       are ordered by tolerance, and the OSF limit rises
                       monotonically with them.
    """
    out = df.copy()

    out["temp_diff_k"] = out["process_temp_k"] - out["air_temp_k"]
    out["power_w"] = out["torque_nm"] * out["rotational_speed_rpm"] * (2 * math.pi / 60)
    out["wear_torque_min_nm"] = out["tool_wear_min"] * out["torque_nm"]

    limits = out["type"].map(OSF_LIMITS_MIN_NM)
    if limits.isna().any():
        bad = sorted(out.loc[limits.isna(), "type"].unique())
        raise ValueError(f"Unknown product Type(s): {bad}. Expected L, M or H.")
    out["osf_limit_min_nm"] = limits.astype(float)
    out["osf_margin"] = out["wear_torque_min_nm"] / out["osf_limit_min_nm"]

    out["type_encoded"] = out["type"].map(TYPE_ENCODING).astype(int)
    return out


# --------------------------------------------------------------------------
# Leakage guard
# --------------------------------------------------------------------------
def assert_no_leakage(columns) -> None:
    """
    Raise if any leakage column made it into a feature list.

    This is the single highest-value assertion in the repo. `machine_failure` is
    literally the OR of the five mode flags, so including any of them produces a
    model that looks perfect and means nothing. Called automatically by
    `build_feature_matrix`; call it yourself in notebooks too.
    """
    leaked = sorted(set(columns) & set(LEAKAGE_COLUMNS))
    if leaked:
        raise ValueError(
            f"LEAKAGE: {leaked} cannot be used as features.\n"
            "machine_failure = TWF OR HDF OR PWF OR OSF OR RNF, so the mode "
            "flags are the answer sheet. See data/README.md."
        )


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def build_feature_matrix(
    path: str | Path = DEFAULT_DATA_PATH,
    verify: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    One call, everything downstream needs.

    Returns
    -------
    X   : DataFrame of FEATURE_COLUMNS only, leakage-checked.
    y   : Series, binary machine_failure.
    df  : the full engineered frame, including mode flags and udi. Use it for
          EDA, for the replay simulator, and for scoring the rules engine
          against ground truth - just never hand it to `.fit()`.
    """
    df = load_raw(path)
    if verify:
        verify_dataset(df)
    df = add_engineered_features(df)

    assert_no_leakage(FEATURE_COLUMNS)
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET].astype(int).copy()
    return X, y, df


def row_to_reading(row: pd.Series) -> dict:
    """
    Convert one engineered row into the camelCase shape of the `SensorReading`
    interface in the frontend data contract (handoff section 7).

    Dev 2 and Dev 3: this is the boundary. `anomalyScore` and `failureProbability`
    are left out here because they come from the models, not the data - fill them
    in after scoring.
    """
    return {
        "cycleId": int(row["udi"]),
        "machineId": f"M-{int(row['udi']):05d}",
        "productType": str(row["type"]),
        "airTemperatureK": float(row["air_temp_k"]),
        "processTemperatureK": float(row["process_temp_k"]),
        "rotationalSpeedRpm": int(row["rotational_speed_rpm"]),
        "torqueNm": float(row["torque_nm"]),
        "toolWearMin": int(row["tool_wear_min"]),
        "tempDiffK": round(float(row["temp_diff_k"]), 2),
        "powerW": round(float(row["power_w"]), 1),
        "wearTorqueMinNm": round(float(row["wear_torque_min_nm"]), 1),
    }


if __name__ == "__main__":
    X, y, df = build_feature_matrix()
    print(f"Dataset OK: {len(df):,} rows, {int(y.sum())} failures "
          f"({y.mean():.2%} positive rate)")
    print(f"\nFeature matrix: {X.shape}")
    print(f"Features ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")
    print("\nEngineered feature summary:")
    print(df[ENGINEERED_FEATURES].describe().T[["mean", "std", "min", "max"]].round(2))
    print("\nSample reading for the frontend contract:")
    import json
    print(json.dumps(row_to_reading(df.iloc[0]), indent=2))
