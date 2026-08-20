"""
Tests for src/rules_engine.py — rule precision/recall, anti-hallucination guard.

OWNER: Dev 4
"""

import pytest


class TestExactRulePerformance:
    """HDF, PWF, OSF must be perfect: 1.000 precision and 1.000 recall."""

    def test_hdf_precision_and_recall(self, feature_data):
        from validate_rules import score_mode
        _, _, df = feature_data
        r = score_mode(df, "HDF")
        assert r is not None, "HDF rule not implemented"
        assert r["precision"] == 1.0, f"HDF precision {r['precision']} != 1.0"
        assert r["recall"] == 1.0, f"HDF recall {r['recall']} != 1.0"

    def test_pwf_precision_and_recall(self, feature_data):
        from validate_rules import score_mode
        _, _, df = feature_data
        r = score_mode(df, "PWF")
        assert r is not None, "PWF rule not implemented"
        assert r["precision"] == 1.0, f"PWF precision {r['precision']} != 1.0"
        assert r["recall"] == 1.0, f"PWF recall {r['recall']} != 1.0"

    def test_osf_precision_and_recall(self, feature_data):
        from validate_rules import score_mode
        _, _, df = feature_data
        r = score_mode(df, "OSF")
        assert r is not None, "OSF rule not implemented"
        assert r["precision"] == 1.0, f"OSF precision {r['precision']} != 1.0"
        assert r["recall"] == 1.0, f"OSF recall {r['recall']} != 1.0"

    def test_hdf_zero_false_positives(self, feature_data):
        from validate_rules import score_mode
        _, _, df = feature_data
        r = score_mode(df, "HDF")
        assert r["fp"] == 0, f"HDF has {r['fp']} false positives"

    def test_pwf_zero_false_positives(self, feature_data):
        from validate_rules import score_mode
        _, _, df = feature_data
        r = score_mode(df, "PWF")
        assert r["fp"] == 0, f"PWF has {r['fp']} false positives"

    def test_osf_zero_false_positives(self, feature_data):
        from validate_rules import score_mode
        _, _, df = feature_data
        r = score_mode(df, "OSF")
        assert r["fp"] == 0, f"OSF has {r['fp']} false positives"


class TestTWF:
    """TWF is a risk window, not an exact rule — precision is expected to be low."""

    def test_twf_recall_high(self, feature_data):
        from validate_rules import score_mode
        _, _, df = feature_data
        r = score_mode(df, "TWF")
        assert r is not None
        assert r["recall"] >= 0.90, f"TWF recall {r['recall']} below 0.90"

    def test_twf_precision_low_is_expected(self, feature_data):
        from validate_rules import score_mode
        _, _, df = feature_data
        r = score_mode(df, "TWF")
        assert r["precision"] < 0.10, (
            f"TWF precision {r['precision']} is suspiciously high — "
            "expected ~0.05 because failure point is random in 200-240 min window"
        )


class TestAntiHallucination:
    """The agent must NEVER name a root cause when confidence is CONFLICT."""

    def test_verified_root_cause_never_returns_twf(self, feature_data):
        from rules_engine import verified_root_cause, check_all
        _, _, df = feature_data
        # Check across all 10,000 rows: exact_only=True should never return TWF
        for _, row in df.iterrows():
            checks = check_all(row)
            cause = verified_root_cause(checks, exact_only=True)
            assert cause != "TWF", (
                f"UDI {int(row['udi'])}: verified_root_cause(exact_only=True) "
                "returned TWF — TWF is not an exact rule"
            )

    def test_conflict_never_invents_cause(self, feature_data):
        """When confidence is CONFLICT, rootCause must be None."""
        from rules_engine import assess
        _, _, df = feature_data

        # Find rows that produce CONFLICT and verify no cause is named
        for _, row in df.iterrows():
            # Force CONFLICT by making ML say failure but no rule fires
            verdict = assess(row, failure_probability=0.99, threshold=0.5)
            if verdict["confidence"] == "CONFLICT":
                assert verdict["rootCause"] is None, (
                    f"UDI {int(row['udi'])}: CONFLICT confidence but "
                    f"rootCause={verdict['rootCause']} — must be None"
                )

    def test_high_confidence_has_verified_cause(self, feature_data):
        """When confidence is HIGH and rootCause is set, it must be an exact mode."""
        from rules_engine import assess, EXACT_MODES
        _, _, df = feature_data

        for _, row in df.iterrows():
            prob = float(row.get("failure_probability", 0.5)) if "failure_probability" in row else 0.5
            verdict = assess(row, failure_probability=prob, threshold=0.5)
            if verdict["confidence"] == "HIGH" and verdict["rootCause"] is not None:
                assert verdict["rootCause"] in EXACT_MODES, (
                    f"UDI {int(row['udi'])}: HIGH confidence with "
                    f"rootCause={verdict['rootCause']} not in EXACT_MODES"
                )
