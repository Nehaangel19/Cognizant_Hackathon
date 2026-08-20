"""
Tests for src/predict.py — Engine scoring, demo cases, contract shape.

OWNER: Dev 4
"""

import pytest


class TestEngineDemoCases:
    """Verify the pre-seeded demo cases return expected results."""

    def test_udi_70_osf_critical(self, engine):
        signal = engine.score_udi(70)
        assert signal["rootCause"] == "OSF", f"UDI 70 expected OSF, got {signal['rootCause']}"
        assert signal["severity"] == "CRITICAL", (
            f"UDI 70 expected CRITICAL, got {signal['severity']}"
        )

    def test_udi_70_high_confidence(self, engine):
        signal = engine.score_udi(70)
        assert signal["confidence"] == "HIGH", (
            f"UDI 70 expected HIGH confidence, got {signal['confidence']}"
        )

    def test_udi_78_conflict_no_cause(self, engine):
        signal = engine.score_udi(78)
        assert signal["confidence"] == "CONFLICT", (
            f"UDI 78 expected CONFLICT, got {signal['confidence']}"
        )
        assert signal["rootCause"] is None, (
            f"UDI 78 expected rootCause=None, got {signal['rootCause']}"
        )

    def test_udi_51_pwf(self, engine):
        signal = engine.score_udi(51)
        assert signal["rootCause"] == "PWF", f"UDI 51 expected PWF, got {signal['rootCause']}"
        assert signal["severity"] == "CRITICAL"

    def test_udi_3237_hdf(self, engine):
        signal = engine.score_udi(3237)
        assert signal["rootCause"] == "HDF", f"UDI 3237 expected HDF, got {signal['rootCause']}"


class TestReadingContract:
    """Verify the SensorReading shape matches the frontend data contract."""

    def test_reading_has_all_required_fields(self, engine):
        signal = engine.score_udi(70)
        reading = signal["reading"]

        required_fields = [
            "cycleId", "machineId", "productType",
            "airTemperatureK", "processTemperatureK",
            "rotationalSpeedRpm", "torqueNm", "toolWearMin",
            "tempDiffK", "powerW", "wearTorqueMinNm",
            "anomalyScore", "failureProbability",
        ]
        for field in required_fields:
            assert field in reading, f"Missing field: {field}"

    def test_reading_no_none_for_numerics(self, engine):
        signal = engine.score_udi(70)
        reading = signal["reading"]
        numeric_fields = [
            "airTemperatureK", "processTemperatureK",
            "rotationalSpeedRpm", "torqueNm", "toolWearMin",
            "tempDiffK", "powerW", "wearTorqueMinNm",
            "anomalyScore", "failureProbability",
        ]
        for field in numeric_fields:
            assert reading[field] is not None, f"{field} is None"
            assert isinstance(reading[field], (int, float)), (
                f"{field} is {type(reading[field])}, expected numeric"
            )

    def test_reading_types(self, engine):
        signal = engine.score_udi(70)
        reading = signal["reading"]
        assert isinstance(reading["cycleId"], int)
        assert isinstance(reading["machineId"], str)
        assert isinstance(reading["productType"], str)
        assert isinstance(reading["rotationalSpeedRpm"], int)

    def test_anomaly_score_in_range(self, engine):
        signal = engine.score_udi(70)
        score = signal["reading"]["anomalyScore"]
        assert 0.0 <= score <= 1.0, f"anomalyScore {score} out of [0,1]"

    def test_failure_probability_in_range(self, engine):
        signal = engine.score_udi(70)
        prob = signal["reading"]["failureProbability"]
        assert 0.0 <= prob <= 1.0, f"failureProbability {prob} out of [0,1]"


class TestScoreOutput:
    """Verify the full score output structure."""

    def test_score_has_all_top_level_keys(self, engine):
        signal = engine.score_udi(70)
        expected_keys = [
            "reading", "severity", "confidence", "rootCause", "rootCauseLabel",
            "rationale", "ruleChecks", "shapValues", "evidence",
            "modelThreshold", "recommendedActions", "partsRequired",
        ]
        for key in expected_keys:
            assert key in signal, f"Missing top-level key: {key}"

    def test_rule_checks_have_correct_shape(self, engine):
        signal = engine.score_udi(70)
        assert len(signal["ruleChecks"]) == 4  # TWF, HDF, PWF, OSF
        for rc in signal["ruleChecks"]:
            assert "mode" in rc
            assert "ruleName" in rc
            assert "condition" in rc
            assert "measuredValue" in rc
            assert "triggered" in rc

    def test_evidence_is_non_empty_for_failure(self, engine):
        signal = engine.score_udi(70)
        assert len(signal["evidence"]) > 0, "UDI 70 should have evidence"

    def test_invalid_udi_raises(self, engine):
        with pytest.raises(KeyError):
            engine.score_udi(99999)
