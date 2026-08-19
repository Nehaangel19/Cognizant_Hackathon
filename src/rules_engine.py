"""
Deterministic physics rules for the AI4I 2020 failure modes.

OWNER: Dev 1  (Dominic)
STATUS: COMPLETE - all four modes implemented and validated against the
        ground-truth flag columns. Run `python src/validate_rules.py` to
        reproduce the scores.

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

VALIDATED PERFORMANCE (all 10,000 rows, see src/validate_rules.py)
------------------------------------------------------------------
    HDF   115/115 caught,  0 false positives   precision 1.000  recall 1.000
    PWF    95/95  caught,  0 false positives   precision 1.000  recall 1.000
    OSF    98/98  caught,  0 false positives   precision 1.000  recall 1.000
    TWF    43/46  caught, 747 false positives  precision 0.054  recall 0.935

Three of the four rules are exact. TWF is not, and cannot be - see its docstring.

CONVENTIONS
-----------
Every check returns a RuleCheck with a HUMAN-READABLE `condition` and
`measured_value`. Those two strings are rendered verbatim in the UI rule table
(Dev 3) and quoted in the agent's evidence array (Dev 2), so they are written
for a maintenance engineer, not for a developer.

The rules never read the mode flag columns. They work from sensor values alone,
exactly as they would on a live machine with no labels available.
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

    Validated: 95/95 caught, 0 false positives. The safe band in the data runs
    3,515 W to 8,998 W, so both thresholds sit in a clean gap - no boundary
    ambiguity.

    Of the 95 real power failures, 31 are under-power and 64 are over-power.
    The evidence string names which side was breached, because a stall and an
    overdrive get completely different maintenance actions.
    """
    power = _power_w(row)
    under = power < PWF_POWER_MIN_W
    over = power > PWF_POWER_MAX_W
    triggered = under or over

    if under:
        detail = (f"power = {power:,.0f} W, {PWF_POWER_MIN_W - power:,.0f} W below the "
                  f"{PWF_POWER_MIN_W:,} W floor (tool stalling in the cut)")
    elif over:
        detail = (f"power = {power:,.0f} W, {power - PWF_POWER_MAX_W:,.0f} W above the "
                  f"{PWF_POWER_MAX_W:,} W ceiling (drive being overdriven)")
    else:
        detail = f"power = {power:,.0f} W, inside the safe band"

    return RuleCheck(
        mode="PWF",
        rule_name="Power envelope",
        condition=f"power outside {PWF_POWER_MIN_W:,}-{PWF_POWER_MAX_W:,} W "
                  f"(torque x angular velocity)",
        measured_value=detail,
        triggered=triggered,
    )


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

    Validated: 98/98 caught, 0 false positives. The boundary is unambiguous -
    the highest non-failing row reaches 10,994 minNm and the lowest failing row
    is 11,003 minNm, so `>` and `>=` give identical results.

    This is the mode the demo escalates on, so the strings here get read out
    loud. `measured_value` reports the overshoot in both absolute minNm and as a
    percentage of the limit, because "9% over" lands faster than a raw number.
    """
    wear_torque = _wear_torque(row)
    product_type = str(row["type"])
    limit = OSF_LIMITS_MIN_NM[product_type]
    triggered = wear_torque > limit

    margin = wear_torque / limit
    if triggered:
        detail = (f"tool wear x torque = {wear_torque:,.0f} minNm, "
                  f"{wear_torque - limit:,.0f} minNm over the {limit:,} limit "
                  f"({margin:.0%} of limit)")
    else:
        detail = (f"tool wear x torque = {wear_torque:,.0f} minNm, "
                  f"{margin:.0%} of the {limit:,} limit")

    return RuleCheck(
        mode="OSF",
        rule_name="Overstrain",
        condition=f"tool wear x torque > {limit:,} minNm ({product_type}-variant limit)",
        measured_value=detail,
        triggered=triggered,
    )


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

    Validated: 43/46 caught, 747 false positives. Precision 0.054, recall 0.935.
    Those numbers are correct behaviour, not a bug - 790 rows sit inside the
    wear window and only 43 of them actually fail.

    A note on the three misses. Three real tool-wear failures occur at 198, 246
    and 253 minutes, outside the documented 200-240 window. Widening the window
    to 198-253 would lift recall to 0.978, and we deliberately do NOT do that:
    it would be fitting the rule to the label column, which is the exact
    practice this engine exists to avoid. We keep the documented physics and
    report the three misses.

    Because this check cannot verify a cause on its own, it is ranked last in
    MODE_PRIORITY - any other rule that fires outranks it.
    """
    wear = _f(row, "tool_wear_min")
    triggered = TWF_WEAR_MIN <= wear <= TWF_WEAR_MAX

    if triggered:
        detail = (f"tool wear = {wear:.0f} min, inside the {TWF_WEAR_MIN}-{TWF_WEAR_MAX} min "
                  f"replacement window ({TWF_WEAR_MAX - wear:.0f} min to window end)")
    elif wear > TWF_WEAR_MAX:
        detail = f"tool wear = {wear:.0f} min, past the {TWF_WEAR_MAX} min window"
    else:
        detail = (f"tool wear = {wear:.0f} min, {TWF_WEAR_MIN - wear:.0f} min "
                  f"below the {TWF_WEAR_MIN} min window")

    return RuleCheck(
        mode="TWF",
        rule_name="Tool wear risk window",
        condition=f"tool wear within {TWF_WEAR_MIN}-{TWF_WEAR_MAX} min "
                  f"(replacement point is random in this range)",
        measured_value=detail,
        triggered=triggered,
    )


# ==========================================================================
# Orchestration
# ==========================================================================
# Priority when several rules fire at once. Ordered by how actionable and how
# severe the mode is: an overstrain event damages the workpiece and the spindle,
# a power excursion trips the drive, heat is a slower burn, tool wear is routine.
MODE_PRIORITY = ["OSF", "PWF", "HDF", "TWF"]

# Only these three rules are exact re-derivations of the dataset's documented
# physics: each recovers 100% of its true cases with zero false positives, so a
# trigger is a verified limit breach.
#
# TWF is deliberately excluded. Its "rule" is a risk window, not a limit - the
# replacement point is drawn at random inside 200-240 min, so 790 rows trigger
# it and only 43 actually fail. Treating it as a hard breach turned 665 of
# 10,000 healthy readings into CONFLICT verdicts and made the agent cry wolf.
# A TWF trigger on its own now yields MEDIUM confidence: worth scheduling a tool
# change, not worth stopping the line.
EXACT_MODES = ["OSF", "PWF", "HDF"]
ADVISORY_MODES = ["TWF"]

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


def verified_root_cause(checks: list[RuleCheck], exact_only: bool = False) -> str | None:
    """
    The single mode the agent is allowed to name, or None.

    None is a legitimate, important answer: it means no physical rule was
    breached, so if the ML model is still screaming, the agent must report a
    CONFLICT rather than invent a cause.

    exact_only=True restricts the answer to the three modes whose rules are
    exact (OSF/PWF/HDF), ignoring the tool-wear risk window. Used by `assess`
    to decide whether a genuine limit was breached.
    """
    fired = set(triggered_modes(checks))
    pool = [m for m in MODE_PRIORITY if not exact_only or m in EXACT_MODES]
    for mode in pool:
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

    Confidence logic. "Limit breached" means one of the three EXACT rules fired;
    the tool-wear risk window is advisory and never on its own creates a conflict.

        ML positive AND exact limit breached  -> HIGH      two independent methods agree
        ML negative AND nothing breached      -> HIGH      agreement on nominal
        ML positive, no exact limit breached  -> CONFLICT  model sees something physics
                                                           does not: unmodelled pattern,
                                                           random failure, or false alarm.
                                                           Escalate, never auto-close.
        ML negative, exact limit breached     -> CONFLICT  a hard physical limit was
                                                           breached; the rule wins
        ML negative, only tool-wear window    -> MEDIUM    routine: schedule a tool change,
                                                           do not stop the line
    """
    checks = check_all(row)
    breach = verified_root_cause(checks, exact_only=True)   # OSF / PWF / HDF only
    advisory = verified_root_cause(checks)                  # may return TWF
    ml_positive = failure_probability >= threshold

    if ml_positive and breach:
        cause = breach
        confidence = "HIGH"
        rationale = (f"Model risk {failure_probability:.0%} and the "
                     f"{MODE_LABELS[cause]} physical limit was breached - independent agreement.")
    elif ml_positive and not breach:
        cause = None
        confidence = "CONFLICT"
        rationale = (f"Model risk {failure_probability:.0%} but no physical limit was "
                     "breached. No verified root cause - may be an unmodelled pattern or a "
                     "random failure. Escalating for human review rather than guessing a cause.")
    elif not ml_positive and breach:
        cause = breach
        confidence = "CONFLICT"
        rationale = (f"{MODE_LABELS[cause]} limit breached despite low model risk "
                     f"({failure_probability:.0%}). A hard physical limit overrides the "
                     "statistical model.")
    elif not ml_positive and advisory == "TWF":
        cause = "TWF"
        confidence = "MEDIUM"
        rationale = (f"Model risk {failure_probability:.0%} and no limit breached, but the "
                     "tool is inside its 200-240 min replacement window. Routine advisory - "
                     "schedule a tool change at the next planned stop.")
    else:
        cause = None
        confidence = "HIGH"
        rationale = (f"Model risk {failure_probability:.0%} and no physical limit "
                     "breached - independent agreement on nominal.")

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
