# Predictive Maintenance Agent for Manufacturing

An explainable predictive-maintenance agent for CNC machining lines, built on the
**AI4I 2020** dataset. It does four things the brief asks for — detect anomalies,
predict failure probability, identify root causes, and recommend maintenance
actions — and it does the fourth honestly: it names a root cause **only when a
deterministic physics rule confirms what the ML model suspects.** When the model
and the physics disagree, it flags the conflict for a human instead of guessing.

That is the whole idea. A black box that says *"87% failure risk"* loses to an
agent that says *"overstrain: tool wear × torque = 11,840 minNm, over the 11,000
limit for L-variant parts — stop the line."*

## What makes it an agent, not a dashboard

For each machine cycle it reaches **one decision** — *escalate now*, *schedule at
the next stop*, or *keep monitoring* — and emits a structured work order that
justifies the call with evidence, a recommended action, the parts needed, and the
dollar downtime avoided.

Across all 10,000 readings, **when the agent names a root cause at high confidence
it is correct 100% of the time (287/287)** — because it is structurally forbidden
from naming any cause a physics rule did not verify.

## Architecture (5 layers)

```
LAYER 1  DATA        CSV replay simulator (~2 rows/sec); UDI is the replay index
LAYER 2  FEATURES    temp_diff, power, wear_torque, osf_margin, type_encoded
LAYER 3  INTELLIGENCE  IsolationForest (anomaly) + XGBoost (risk) + OvR root cause
                       + SHAP  +  deterministic physics rules engine
LAYER 4  AGENT       7 tools, confidence gate (ML vs physics), decision policy
LAYER 5  INTERFACE   dashboard + structured JSON work order
```

## Results (reproducible — run `python src/train.py`)

| Metric | Value | Note |
|---|---|---|
| **Failure-risk PR-AUC** | **0.870** | vs 0.034 baseline — a 26× lift |
| Recall @ tuned threshold | 0.868 | caught 59 of 68 test failures |
| Precision @ that threshold | 0.747 | 20 false alarms across 2,000 cycles |
| Physics rules HDF / PWF / OSF | **1.000 / 1.000** | precision and recall, zero false positives |
| Root-cause accuracy at HIGH confidence | **287/287 (100%)** | agent never names an unverified cause |

We lead with **PR-AUC**, not accuracy: at a 3.39% failure rate a model that
predicts "never fails" scores 96.6% accuracy and catches nothing.

## Setup

Windows users: use **Git Bash**, not Command Prompt.

```bash
# 1. Clone
git clone https://github.com/Nehaangel19/Cognizant_Hackathon.git
cd Cognizant_Hackathon

# 2. Virtual environment
python -m venv venv
source venv/Scripts/activate        # Windows (Git Bash)
# source venv/bin/activate          # macOS / Linux

# 3. Dependencies
pip install -r requirements.txt

# 4. (Agent LLM mode only) copy the env template and add your key
cp .env.example .env                # then edit .env, add ANTHROPIC_API_KEY
```

The dataset (`data/ai4i2020.csv`) is committed, so there is nothing to download.

## How to run

Run these from the repo root, in order. Steps 1–4 need no API key.

```bash
# 1. Sanity-check the dataset and see the engineered features
python src/features.py

# 2. Score the physics rules against ground truth (HDF/PWF/OSF = 1.000/1.000)
python src/validate_rules.py

# 3. Train all models + SHAP (~30s). Writes models/ and docs/model_metrics.json
python src/train.py

# 4. End-to-end signal for the pre-seeded demo cases
python src/predict.py

# 5. Run the agent on one cycle (offline, deterministic — no API key)
python src/agent/agent.py 70            # OSF escalation — the money shot
python src/agent/agent.py 78            # CONFLICT — agent refuses to bluff
python src/agent/agent.py --demo        # all pre-seeded demo cases

# 6. Agent with a real Anthropic tool-calling loop (needs ANTHROPIC_API_KEY)
python src/agent/agent.py 70 --llm

# 7. Exploratory data analysis
jupyter lab notebooks/01_eda.ipynb
```

> **Run order matters:** `src/predict.py` and the agent need the trained
> artefacts, so run `src/train.py` (step 3) first. Models are gitignored and
> regenerate in ~30 seconds.

## Repository layout

```
data/               ai4i2020.csv (committed) + integrity README
notebooks/          01_eda.ipynb              — EDA, executed with outputs
src/
  features.py       engineered features + leakage guard        [Dev 1]
  rules_engine.py   deterministic physics rules, all 4 modes    [Dev 1]
  validate_rules.py scores rules vs ground truth                [Dev 1]
  train.py          IsolationForest + XGBoost + OvR + SHAP       [Dev 1]
  predict.py        single inference entry point for the agent   [Dev 1]
  agent/
    schemas.py      Pydantic models mirroring the data contract  [Dev 2]
    tools.py        the 7 agent tools + Anthropic schemas        [Dev 2]
    agent.py        orchestrator: decides + emits work order     [Dev 2]
  app/              dashboard / replay UI                        [Dev 3]
docs/
  maintenance_playbook.md / playbook.json   actions per mode     [Non-tech A]
  cost_model.md / cost_params.json          downtime economics   [Non-tech A]
  demo_script.md, architecture.png, qa_log.md                    [Non-tech B]
models/             gitignored — regenerate with train.py
PROGRESS.md         rolling work log — read this for current status
```

## Named limitations (we state these openly in the pitch)

- **Synthetic data.** Physically grounded, but not a real production line, so the
  cost figures are an illustrative model, not measured plant data.
- **No timestamps, no machine IDs.** Rows are independent snapshots, not a time
  series; `UDI` is only a replay index.
- **RNF is irreducible.** 0.1% random by construction, no sensor signature,
  excluded from the root-cause model by design.
- **9 of 339 failures carry no cause label**, so deterministic rules can explain
  at most 330 of 339.
- **Tool-wear (TWF) is a risk window, not a verdict** — 0.05 precision is the
  mathematical maximum, so it is treated as advisory and never escalates alone.

## Dataset

AI4I 2020 Predictive Maintenance Dataset (S. Matzka, *Explainable AI for
Predictive Maintenance Applications*). 10,000 rows, 339 failures. UCI id 601;
Kaggle `stephanmatzka/predictive-maintenance-dataset-ai4i-2020`.
