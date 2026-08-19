"""
Deterministic physics rules for the AI4I 2020 failure modes.

OWNER: Dev 1  (Dominic)
STATUS: SCAFFOLD - HDF is implemented as a worked example.
        TWF, PWF and OSF are yours to fill in. Search this file for "TODO".

WHY THIS FILE IS THE PROJECT'S DIFFERENTIATOR
---------------------------------------------
An XGBoost model can tell a judge "91% failure probability". It cannot tell them
why, and it cannot be audited. The AI4I dataset was generated from documented
physical conditions, so we can re-derive those conditions exactly and check them
against every reading.

That gives the agent a second, independent opinion:

    ML says failure + rules say OSF   -> confidence HIGH, cause is OSF
    ML says failure + rules say nothing -> confidence CONFLICT, escalate to a human
    ML says fine    + rules say OSF   -> confidence CONFLICT, do not dismiss

The agent is forbidden from naming a root cause the rules engine did not verify.
That is the whole answer to "how do you stop the LLM hallucinating a diagnosis".

HOW TO FILL IN THE TODOs
------------------------
1. Read `check_hdf` below. Copy its shape exactly.
2. Every check returns a RuleCheck with a HUMAN-READABLE `condition` and
   `measured_value`. Those two strings get rendered verbatim in the UI rule
   table (Dev 3) and quoted in the agent's evidence array (Dev 2). Write them
   for a maintenance engineer, not for yourself.
3. Run `python src/validate_rules.py` after each one. It scores your rule
   against the ground-truth flag column and prints precision and recall.
   Target: HDF, PWF and OSF should hit ~1.00 on both. TWF will not - see below.
4. Never import the mode flag columns here. The rules must work from sensor
   values alone, exactly as they would on a live machine.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

# Let this module be imported from anywhere (src/agent/, notebooks/, app/)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import (  # noqa: F401  (constants are the shared source of truth)
    HDF_SPEED_RPM,
    HDF_TEMP_DIFF_K,
    OSF_LIMITS_MIN_NM,
    PWF_POWER_MAX_W,
    PWF_POWER_MIN_W,
    TWF_WEAR_MAX,
    TWF_WEAR_MIN,
)


@dataclass
class RuleCheck:
    """Mirrors the `RuleCheck` TypeScript interface in handoff section 7."""
    mode: str            # 'TWF' | 'HDF' | 'PWF' | 'OSF'
    rule_name: str       # short label for the UI table
    condition: str       # the threshold, in words
    measured_value: str  # what this row actually was
    triggered: bool

    def to_dict(self) -> dict:
        d = asdict(self)
        return {
            "mode": d["mode"],
            "ruleName": d["rule_name"],
            "condition": d["condition"],
            "measuredValue": d["measured_value"],
            "triggered": d["triggered"],
        }


# --------------------------------------------------------------------------
# Small helpers so the checks work on a plain dict OR a pandas row
# --------------------------------------------------------------------------
def _f(row, key: str) -> float:
    return float(row[key])


def _temp_diff(row) -> float:
    if "temp_diff_k" in row:
        return _f(row, "temp_diff_k")
    return _f(row, "process_temp_k") - _f(row, "air_temp_k")


def _power_w(row) -> float:
    if "power_w" in row:
        return _f(row, "power_w")
    return _f(row, "torque_nm") * _f(row, "rotational_speed_rpm") * (2 * math.pi / 60)


def _wear_torque(row) -> float:
    if "wear_torque_min_nm" in row:
        return _f(row, "wear_torque_min_nm")
    return _f(row, "tool_wear_min") * _f(row, "torque_nm")


# ==========================================================================
# HDF - Heat Dissipation Failure          [ IMPLEMENTED - use as the template ]
# ==========================================================================
def check_hdf(row) -> RuleCheck:
    """
    Ground truth: heat dissipation fails if
        (process temperature - air temperature) < 8.6 K  AND
        rotational speed < 1380 rpm

    Physically: the process is not shedding heat into the surrounding air fast
    enough, and the spindle is turning too slowly to help. 115 rows in the
    dataset. Both conditions must hold - AND, not OR.
    """
    dt = _temp_diff(row)
    rpm = _f(row, "rotational_speed_rpm")
    triggered = (dt < HDF_TEMP_DIFF_K) and (rpm < HDF_SPEED_RPM)

    return RuleCheck(
        mode="HDF",
        rule_name="Heat dissipation",
        condition=f"temp difference < {HDF_TEMP_DIFF_K} K AND speed < {HDF_SPEED_RPM} rpm",
        measured_value=f"temp difference = {dt:.1f} K, speed = {rpm:.0f} rpm",
        triggered=triggered,
    )


# ==========================================================================
# PWF - Power Failure                                            [ TODO Dev 1 ]
# ==========================================================================
def check_pwf(row) -> RuleCheck:
    """
    Ground truth: power = torque [Nm] x rotational speed [rad/s].
    The process fails if power is BELOW 3500 W or ABOVE 9000 W. 95 rows.

    Note the OR - this is a two-sided band, unlike HDF. Under-power means the
    tool stalls in the cut; over-power means the drive is being overdriven.
    Your `measured_value` string should say which side was breached, because a
    stall and an overdrive get completely different maintenance actions.

    Helpers available: _power_w(row) already does the rad/s conversion.
    Constants: PWF_POWER_MIN_W = 3500, PWF_POWER_MAX_W = 9000.
    """
    # TODO(Dev 1): implement. Delete the raise when done.
    raise NotImplementedError("check_pwf - see docstring above")


# ==========================================================================
# OSF - Overstrain Failure                                       [ TODO Dev 1 ]
# ==========================================================================
def check_osf(row) -> RuleCheck:
    """
    Ground truth: tool wear [min] x torque [Nm] exceeds a limit that depends on
    the product variant:
        L -> 11,000 minNm    M -> 12,000 minNm    H -> 13,000 minNm
    98 rows.

    This is the mode the demo script escalates on, so its strings matter most.
    Aim for a `measured_value` that reads like the handoff's example evidence:
        "tool wear x torque = 11,840 minNm, limit 11,000 for L variant"

    Helpers: _wear_torque(row). Constant: OSF_LIMITS_MIN_NM[row['type']].
    """
    # TODO(Dev 1): implement. Delete the raise when done.
    raise NotImplementedError("check_osf - see docstring above")


# ==========================================================================
# TWF - Tool Wear Failure                                        [ TODO Dev 1 ]
# ==========================================================================
def check_twf(row) -> RuleCheck:
    """
    Ground truth: the tool is replaced or fails at a RANDOMLY selected point
    between 200 and 240 minutes of wear. 46 rows.

    HONESTY WARNING - read this before you tune anything.
    Because the failure point is drawn at random inside the window, no
    deterministic rule can reproduce TWF exactly. Many rows sit at 200-240 min
    of wear and do NOT fail. So this check is a RISK WINDOW, not a verdict:
    recall will be high (~0.93), precision will be poor (~0.05), and that is
    correct behaviour, not a bug.

    Implement it as "tool wear is inside the 200-240 min replacement window",
    and let the `rule_name` say "risk window" rather than "failure". Then say
    this out loud in the demo. A team that reports an honest 0.15 precision on
    an irreducibly random mode reads as more credible, not less.

    Constants: TWF_WEAR_MIN = 200, TWF_WEAR_MAX = 240.
    """
    # TODO(Dev 1): implement. Delete the raise when done.
    raise NotImplementedError("check_twf - see docstring above")


# ==========================================================================
# Orchestration
# ==========================================================================
# Priority when several rules fire at once. Ordered by how actionable and how
# severe the mode is: an overstrain event damages the workpiece and the spindle,
# a power excursion trips the drive, heat is a slower burn, tool wear is routine.
MODE_PRIORITY = ["OSF", "PWF", "HDF", "TWF"]

MODE_LABELS = {
    "TWF": "Tool Wear Failure",
    "HDF": "Heat Dissipation Failure",
    "PWF": "Power Failure",
    "OSF": "Overstrain Failure",
    "RNF": "Random Failure",
}


def check_all(row) -> list[RuleCheck]:
    """
    Run every rule against one reading. Returns all four checks, triggered or
    not - the UI shows the non-triggered ones too, greyed out, because "we
    checked and it was fine" is itself evidence.
    """
    return [check_twf(row), check_hdf(row), check_pwf(row), check_osf(row)]


def triggered_modes(checks: list[RuleCheck]) -> list[str]:
    return [c.mode for c in checks if c.triggered]


def verified_root_cause(checks: list[RuleCheck]) -> str | None:
    """
    The single mode the agent is allowed to name, or None.

    None is a legitimate, important answer: it means no physical rule was
    breached, so if the ML model is still screaming, the agent must report a
    CONFLICT rather than invent a cause.
    """
    fired = set(triggered_modes(checks))
    for mode in MODE_PRIORITY:
        if mode in fired:
            return mode
    return None


def assess(row, failure_probability: float, threshold: float = 0.5) -> dict:
    """
    Fuse the ML opinion with the physics opinion. This is what Dev 2's
    `check_physics_rules()` tool should call.

    Returns a dict with:
        rootCause    the verified mode, or None
        confidence   'HIGH' | 'MEDIUM' | 'CONFLICT'
        ruleChecks   list of dicts matching the frontend contract
        rationale    one sentence a human can read

    Confidence logic:
        ML positive AND a rule fired      -> HIGH      (two independent methods agree)
        ML negative AND no rule fired     -> HIGH      (agreement on 'nominal')
        ML positive, no rule fired        -> CONFLICT  (model sees something physics
                                                        does not - could be RNF, could
                                                        be a false alarm; escalate,
                                                        never auto-close)
        ML negative, a rule fired         -> CONFLICT  (a hard physical limit was
                                                        breached; the rule wins)
    """
    checks = check_all(row)
    cause = verified_root_cause(checks)
    ml_positive = failure_probability >= threshold

    if ml_positive and cause:
        confidence = "HIGH"
        rationale = (f"Model risk {failure_probability:.0%} and the "
                     f"{MODE_LABELS[cause]} physical limit was breached - independent agreement.")
    elif not ml_positive and not cause:
        confidence = "HIGH"
        rationale = (f"Model risk {failure_probability:.0%} and no physical limit "
                     "breached - independent agreement on nominal.")
    elif ml_positive and not cause:
        confidence = "CONFLICT"
        rationale = (f"Model risk {failure_probability:.0%} but no physical rule fired. "
                     "No verified root cause - may be an unmodelled pattern or a random "
                     "failure. Escalating for human review rather than guessing a cause.")
    else:
        confidence = "CONFLICT"
        rationale = (f"{MODE_LABELS[cause]} limit breached despite low model risk "
                     f"({failure_probability:.0%}). A hard physical limit overrides the "
                     "statistical model.")

    return {
        "rootCause": cause,
        "rootCauseLabel": MODE_LABELS[cause] if cause else "Unverified",
        "confidence": confidence,
        "ruleChecks": [c.to_dict() for c in checks],
        "rationale": rationale,
    }


if __name__ == "__main__":
    import json
    from features import build_feature_matrix

    _, _, df = build_feature_matrix()
    row = df[df["osf"] == 1].iloc[0]
    print(f"Row UDI {int(row['udi'])} - ground truth OSF=1\n")
    try:
        print(json.dumps(assess(row, failure_probability=0.91), indent=2))
    except NotImplementedError as exc:
        done = []
        for name, fn in [("TWF", check_twf), ("HDF", check_hdf),
                         ("PWF", check_pwf), ("OSF", check_osf)]:
            try:
                fn(row); done.append(name)
            except NotImplementedError:
                pass
        print(f"Implemented: {done or 'none'}")
        print(f"Still TODO : {[m for m in ['TWF','HDF','PWF','OSF'] if m not in done]}")
        print(f"\n-> {exc}")
        print("-> Fill these in, then run:  python src/validate_rules.py")
