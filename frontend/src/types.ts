/* Mirrors the Pydantic models in src/agent/schemas.py, which in turn mirror the
   TypeScript data contract in the project handoff. Field names are camelCase on
   both sides so no transformation happens at the boundary. */

export type FailureMode = 'TWF' | 'HDF' | 'PWF' | 'OSF' | 'RNF';
export type Severity = 'NOMINAL' | 'ADVISORY' | 'WARNING' | 'CRITICAL';
export type Confidence = 'HIGH' | 'MEDIUM' | 'CONFLICT';

export interface SensorReading {
  cycleId: number;
  machineId: string;
  productType: 'L' | 'M' | 'H';
  airTemperatureK: number;
  processTemperatureK: number;
  rotationalSpeedRpm: number;
  torqueNm: number;
  toolWearMin: number;
  tempDiffK: number;
  powerW: number;
  wearTorqueMinNm: number;
  anomalyScore: number;
  failureProbability: number;
}

export interface RuleCheck {
  mode: FailureMode;
  ruleName: string;
  condition: string;
  measuredValue: string;
  triggered: boolean;
}

export interface ShapContribution {
  feature: string;
  value: number;
  contribution: number;
}

export interface Signal {
  reading: SensorReading;
  severity: Severity;
  confidence: Confidence;
  rootCause: FailureMode | null;
  rootCauseLabel: string;
  rationale: string;
  ruleChecks: RuleCheck[];
  shapValues: ShapContribution[];
  evidence: string[];
  modelThreshold: number;
  recommendedActions: string[];
  partsRequired: string[];
}

export interface WorkOrder {
  id: string;
  cycleId: number;
  machineId: string;
  severity: Severity;
  confidence: Confidence;
  failureProbability: number;
  rootCause: FailureMode | null;
  rootCauseLabel: string;
  evidence: string[];
  ruleChecks: RuleCheck[];
  shapValues: ShapContribution[];
  recommendedActions: string[];
  partsRequired: string[];
  estimatedRepairHours: number;
  downtimeAvoidedHours: number;
  costAvoidedUsd: number;
  decision: string;
  createdAt: string;
}

export interface AnalyzeResponse {
  workOrder: WorkOrder;
  narration: string;
  decisionReason: string;
  toolTrace: string[];
  mode: string;
  fallbackReason: string | null;
  reading: SensorReading;
}

export interface Health {
  status: string;
  modelsLoaded: boolean;
  error: string | null;
  threshold: number | null;
  totalCycles: number;
  cachedCycles: number;
}

export interface RulePerf {
  mode: string; label: string; truePositives: number; actual: number;
  falsePositives: number; precision: number; recall: number; exact: boolean;
}
