"""
Pydantic models mirroring the TypeScript data contract in handoff section 7.

OWNER: Dev 2
These are the single source of truth for the JSON shape that crosses every
boundary: rules engine -> agent -> frontend. The field names are camelCase on
purpose, matching the TypeScript interfaces exactly, so the frontend consumes an
agent work order with zero transformation.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

FailureMode = Literal["TWF", "HDF", "PWF", "OSF", "RNF"]
Severity = Literal["NOMINAL", "ADVISORY", "WARNING", "CRITICAL"]
Confidence = Literal["HIGH", "MEDIUM", "CONFLICT"]


class RuleCheck(BaseModel):
    mode: FailureMode
    ruleName: str
    condition: str
    measuredValue: str
    triggered: bool


class ShapContribution(BaseModel):
    feature: str
    value: float
    contribution: float


class WorkOrder(BaseModel):
    """Mirrors the `WorkOrder` TypeScript interface exactly."""
    id: str
    cycleId: int
    machineId: str
    severity: Severity
    confidence: Confidence
    failureProbability: float
    rootCause: Optional[FailureMode] = None
    rootCauseLabel: str
    evidence: list[str] = Field(default_factory=list)
    ruleChecks: list[RuleCheck] = Field(default_factory=list)
    shapValues: list[ShapContribution] = Field(default_factory=list)
    recommendedActions: list[str] = Field(default_factory=list)
    partsRequired: list[str] = Field(default_factory=list)
    estimatedRepairHours: float = 0.0
    downtimeAvoidedHours: float = 0.0
    costAvoidedUsd: float = 0.0
    decision: str = ""          # escalate-now | schedule | monitor
    createdAt: str = ""         # stamped by the caller; scripts here cannot call now()
