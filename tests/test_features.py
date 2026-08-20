"""
Tests for src/features.py — dataset integrity, leakage guard, feature engineering.

OWNER: Dev 4
"""

import math

import numpy as np
import pandas as pd
import pytest


class TestDatasetVerification:
    """Verify the committed dataset matches the canonical AI4I 2020 file."""

    def test_dataset_shape(self, feature_data):
        _, _, df = feature_data
        assert len(df) == 10_000, f"Expected 10,000 rows, got {len(df):,}"

    def test_failure_count(self, feature_data):
        _, y, _ = feature_data
        assert int(y.sum()) == 339, f"Expected 339 failures, got {int(y.sum())}"

    def test_positive_rate(self, feature_data):
        _, y, _ = feature_data
        assert abs(y.mean() - 0.0339) < 0.001, (
            f"Expected ~3.39% positive rate, got {y.mean():.4f}"
        )

    def test_mode_counts(self, feature_data):
        _, _, df = feature_data
        expected = {"twf": 46, "hdf": 115, "pwf": 95, "osf": 98, "rnf": 19}
        for mode, count in expected.items():
            actual = int(df[mode].sum())
            assert actual == count, f"{mode.upper()}: expected {count}, got {actual}"


class TestLeakageGuard:
    """The leakage guard must catch forbidden columns and allow valid ones."""

    def test_leakage_guard_catches_mode_flags(self):
        from features import assert_no_leakage, LEAKAGE_COLUMNS
        with pytest.raises(ValueError, match="LEAKAGE"):
            assert_no_leakage(["twf", "hdf", "torque_nm"])

    def test_leakage_guard_catches_target(self):
        from features import assert_no_leakage
        with pytest.raises(ValueError, match="LEAKAGE"):
            assert_no_leakage(["machine_failure"])

    def test_leakage_guard_catches_udi(self):
        from features import assert_no_leakage
        with pytest.raises(ValueError, match="LEAKAGE"):
            assert_no_leakage(["udi", "air_temp_k"])

    def test_leakage_guard_catches_product_id(self):
        from features import assert_no_leakage
        with pytest.raises(ValueError, match="LEAKAGE"):
            assert_no_leakage(["product_id"])

    def test_no_leakage_on_real_features(self):
        from features import assert_no_leakage, FEATURE_COLUMNS
        assert_no_leakage(FEATURE_COLUMNS)  # should not raise

    def test_all_leakage_columns_identified(self):
        from features import LEAKAGE_COLUMNS
        expected = {"twf", "hdf", "pwf", "osf", "rnf", "machine_failure", "udi", "product_id"}
        assert set(LEAKAGE_COLUMNS) == expected


class TestFeatureEngineering:
    """Spot-check engineered features on known rows."""

    def test_power_w_formula(self, feature_data):
        _, _, df = feature_data
        row = df.iloc[0]
        expected = row["torque_nm"] * row["rotational_speed_rpm"] * (2 * math.pi / 60)
        assert abs(row["power_w"] - expected) < 0.1, (
            f"power_w mismatch: {row['power_w']} vs {expected}"
        )

    def test_osf_margin_formula(self, feature_data):
        _, _, df = feature_data
        row = df.iloc[0]
        from features import OSF_LIMITS_MIN_NM
        limit = OSF_LIMITS_MIN_NM[row["type"]]
        expected = row["wear_torque_min_nm"] / limit
        assert abs(row["osf_margin"] - expected) < 0.001, (
            f"osf_margin mismatch: {row['osf_margin']} vs {expected}"
        )

    def test_temp_diff_k(self, feature_data):
        _, _, df = feature_data
        row = df.iloc[0]
        expected = row["process_temp_k"] - row["air_temp_k"]
        assert abs(row["temp_diff_k"] - expected) < 0.01

    def test_type_encoded(self, feature_data):
        _, _, df = feature_data
        from features import TYPE_ENCODING
        for t, enc in TYPE_ENCODING.items():
            subset = df[df["type"] == t]
            assert (subset["type_encoded"] == enc).all(), (
                f"type_encoded for {t} should be {enc}"
            )

    def test_spot_check_udi_70(self, feature_data):
        """UDI 70 is the overstrain demo case — spot-check its features."""
        _, _, df = feature_data
        row = df[df["udi"] == 70].iloc[0]
        assert row["type"] == "L"
        assert row["tool_wear_min"] == 191
        assert row["torque_nm"] == 65.7
        # wear_torque = 191 * 65.7 = 12548.7
        assert abs(row["wear_torque_min_nm"] - 12548.7) < 1.0
        # osf_margin = 12548.7 / 11000 = 1.141
        assert abs(row["osf_margin"] - 1.141) < 0.01
