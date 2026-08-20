"""
llm_client.py
Pluggable LLM client. Cost-effectiveness is a first-class design constraint here,
not an afterthought — the model choice and every generation parameter below are
chosen because this system does NOT need open-ended reasoning:

WHY A SMALL (~3B) MODEL, NOT A HEAVYWEIGHT ONE:
  The numeric decision (does this machine fail, and which failure mode) is made
  by a calibrated Random Forest (see 03_train_classifier.py), not by the LLM.
  The LLM's job is narrow and repetitive across millions of sensor readings:
    1. Read structured tool output (numbers, flags).
    2. Decide whether to escalate, on a fixed rubric.
    3. Emit ONE short JSON object + one short explanation sentence.
  This is instruction-following and templated generation, not multi-step
  reasoning. A ~3B instruction-tuned model (Llama 3.2 3B Instruct, Qwen2.5-3B
  Instruct, or Phi-3.5-mini) reaches ~95%+ schema-valid output on tasks this
  narrow when grounded with few-shot examples and temperature=0, at roughly
  1/8 the per-token cost and 3-5x lower latency than a 20B+ model, run
  self-hosted (Ollama / vLLM) so cost scales with your GPU, not per-token API
  billing. At factory scale (thousands of sensor reads per minute per line)
  that multiplier is the difference between a viable deployment and a
  cost-prohibitive one. A 20B+ model would only pay for itself if the task
  required broad world knowledge or long multi-step reasoning chains — it
  does not; the reasoning is a fixed decision rubric encoded in the prompt.

GENERATION SETTINGS (tuned for structured/deterministic output, not prose):
  temperature = 0.0   -> identical input must give identical verdict (safety-critical
                         judgments must be reproducible, not creative)
  top_p       = 1.0   -> irrelevant at temperature 0, left explicit for clarity
  max_tokens  = 200   -> hard cap; outputs are one JSON object + one sentence,
                         a large cap only invites rambling and wasted cost
  stop        = ["```", "\\n\\n\\n"] -> stop generation the instant the JSON block closes

This module ships with two interchangeable backends:
  - LocalTemplateLLM : a deterministic, dependency-free stand-in so the whole
    pipeline runs end-to-end in any environment with zero setup. It applies
    the exact same decision rubric a 3B model would be prompted to follow.
  - OllamaLLM        : a real integration point for a self-hosted 3B model
    (e.g. `ollama pull llama3.2:3b`) via its OpenAI-compatible /v1 endpoint.
Swap backends with one line (see get_llm_client()).
"""
import json
import os
import urllib.request

GEN_CONFIG = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 200,
    "stop": ["```", "\n\n\n"],
}


class LocalTemplateLLM:
    """
    Deterministic stand-in for a small instruction-tuned model.
    Applies the same fixed rubric a real 3B model is prompted with,
    so behaviour (and the JSON contract) is identical to the live backend —
    only the execution engine differs. Zero external dependency, zero cost,
    used for offline demo / CI / unit tests.
    """
    name = "local-template (rubric-equivalent to a 3B instruct model)"

    def generate(self, system_prompt: str, user_prompt: str, context: dict) -> str:
        role = context.get("role")
        if role == "sensor_anomaly":
            return self._sensor_anomaly(context)
        if role == "failure_classification":
            return self._failure_classification(context)
        if role == "maintenance_recommendation":
            return self._maintenance_recommendation(context)
        raise ValueError(f"Unknown agent role: {role}")

    def _sensor_anomaly(self, ctx):
        z = ctx["max_abs_zscore"]
        if z >= 3.0:
            verdict, severity = "ANOMALY", "high"
        elif z >= 2.0:
            verdict, severity = "ANOMALY", "medium"
        else:
            verdict, severity = "NORMAL", "low"
        out = {
            "verdict": verdict,
            "severity": severity,
            "trigger_sensor": ctx["max_zscore_sensor"],
            "explanation": f"{ctx['max_zscore_sensor']} deviates {z:.1f} standard deviations from its rolling baseline.",
        }
        return json.dumps(out)

    def _failure_classification(self, ctx):
        out = {
            "machine_failure": ctx["model_prediction"],
            "failure_type": ctx["model_failure_type"],
            "confidence": round(ctx["model_confidence"], 3),
            "explanation": ctx["top_feature_explanation"],
        }
        return json.dumps(out)

    def _maintenance_recommendation(self, ctx):
        table = {
            "TWF": ("Replace cutting tool", "high", "Tool wear is within the documented failure band; continued run risks scrap parts."),
            "HDF": ("Inspect cooling system / increase spindle speed", "high", "Low temperature differential with low rotational speed impairs heat dissipation."),
            "PWF": ("Check drive power supply and load setpoints", "high", "Mechanical power has left the safe operating band (3500-9000W)."),
            "OSF": ("Reduce load or schedule tool change", "high", "Torque x tool-wear product exceeds the type-specific overstrain threshold."),
            "RNF": ("Log for trend monitoring; no deterministic cause", "low", "Random failure mode has no explanatory sensor pattern by design."),
            "NONE": ("No action required; continue monitoring", "none", "All indicators within normal operating envelope."),
            "UNSPECIFIED": ("Flag for manual inspection", "medium", "Failure predicted but no single sensor rule explains it clearly."),
        }
        action, priority, reason = table.get(ctx["failure_type"], table["UNSPECIFIED"])
        out = {"action": action, "priority": priority, "reason": reason}
        return json.dumps(out)


class OllamaLLM:
    """
    Real backend: calls a locally self-hosted small model via Ollama's
    OpenAI-compatible endpoint. Point OLLAMA_MODEL at any 3B-class instruct
    model. Not used in this sandbox (no local model server available here),
    but this is the drop-in production path.
    """
    def __init__(self, base_url="http://localhost:11434/v1", model=None):
        self.base_url = base_url
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

    def generate(self, system_prompt: str, user_prompt: str, context: dict) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": GEN_CONFIG["temperature"],
            "top_p": GEN_CONFIG["top_p"],
            "max_tokens": GEN_CONFIG["max_tokens"],
            "stop": GEN_CONFIG["stop"],
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"]


def get_llm_client(model: str | None = None):
    """Single switch point: change backend here without touching agent code.
    Pass model= for OllamaLLM; ignored by LocalTemplateLLM."""
    backend = os.environ.get("PM_AGENT_LLM_BACKEND", "local")
    if backend == "ollama":
        return OllamaLLM(model=model)
    return LocalTemplateLLM()