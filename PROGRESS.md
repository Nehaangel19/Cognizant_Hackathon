# Progress Log — Predictive Maintenance Agent

Rolling work log. Append a dated entry per session; do not rewrite history.
Metrics quoted here are reproduced by `python src/train.py` and
`python src/validate_rules.py` — if you change a model, update this file in the
same commit.

---

## Status board

| Area | File | Owner | State |
|---|---|---|---|
| Shared plumbing | `.gitignore`, `requirements.txt`, `.env.example` | Dev 1 | **Done** |
| Dataset | `data/ai4i2020.csv` + `data/README.md` | Dev 1 | **Done** — verified |
| Features | `src/features.py` | Dev 1 | **Done** |
| Rules engine | `src/rules_engine.py` | Dev 1 | **Partial** — HDF done, 3 modes TODO |
| Rules validation | `src/validate_rules.py` | Dev 1 | **Done** |
| Models + SHAP | `src/train.py` | Dev 1 | **Done** — trained, artefacts saved |
| EDA | `notebooks/01_eda.ipynb` | Dev 1 | **Done** — executed with outputs |
| Playbook | `docs/maintenance_playbook.md` | Non-tech A | **Empty — blocks Dev 2 at H+10** |
| Cost model | `docs/cost_model.md` | Non-tech A | **Empty — due H+14** |
| Agent tools | `src/agent/tools.py` | Dev 2 | **Empty — H+8** |
| Agent loop | `src/agent/agent.py` | Dev 2 | **Empty — H+13** |
| Frontend | `src/app/` | Dev 3 | **Empty — H+2, and blocked on a decision** |
| Demo script | `docs/demo_script.md` | Non-tech B | **Empty — due H+24** |
| Architecture diagram | `docs/architecture.png` | Non-tech B | **Empty (0 bytes)** |
| QA log | `docs/qa_log.md` | Non-tech B | **Empty** |
| README | `README.md` | Non-tech B | **Empty** |

---

## Session 1 — Dev 1 (Dominic)

### Shared plumbing

Filled first, deliberately, because these are the files all five of us touch and
they are the classic hour-30 merge conflict.

- `.gitignore` — ignores `models/`, `.env`, `venv/`, notebook checkpoints, and
  `data/*` **except** `data/ai4i2020.csv` and `data/README.md`.
- `requirements.txt` — minimum-version pins (`>=`) rather than exact, so Windows
  and Linux both resolve. FastAPI and ChromaDB are commented out pending the two
  open decisions below.
- `.env.example` — `ANTHROPIC_API_KEY` and paths. Real `.env` is gitignored.

### Dataset — committed on purpose

`data/ai4i2020.csv`, 522 KB, UTF-8 with the BOM stripped. Committed so all five
of us train on a byte-identical file and nobody loses twenty minutes to a Kaggle
login at hour 30.

Verified against the published dataset:

```
rows 10,000 x 14 columns
Machine failure 339  (3.39% positive rate)
TWF 46   HDF 115   PWF 95   OSF 98   RNF 19
```

`features.py::verify_dataset()` re-checks these on every load and raises if they
differ, so no metric in this repo can be produced from the wrong file.

### `src/features.py`

Clean column names, five engineered features, and a leakage guard.

| Feature | Formula | Which rule it serves |
|---|---|---|
| `temp_diff_k` | process temp − air temp | HDF |
| `power_w` | torque × rpm × 2π/60 | PWF |
| `wear_torque_min_nm` | tool wear × torque | OSF |
| `osf_margin` | wear×torque ÷ type limit | OSF, normalised across L/M/H |
| `type_encoded` | L=0, M=1, H=2 | product variant |

`osf_margin` is the most useful column in the repo: dividing by the type-specific
limit collapses the three thresholds (11k/12k/13k) onto a single line at 1.0, so
one number is comparable across every product variant. It is what the agent
quotes in its evidence string.

`assert_no_leakage()` raises if any of `TWF, HDF, PWF, OSF, RNF, machine_failure,
udi, product_id` reaches the feature matrix. **Do not work around it.**

`row_to_reading()` emits the camelCase `SensorReading` shape from the frontend
data contract — Dev 3 can consume this directly.

### `src/train.py` — trained, artefacts saved

80/20 stratified split, `random_state=42`.

**Failure risk (XGBoost, `scale_pos_weight=28.5`)**

| Metric | Value |
|---|---|
| **PR-AUC** | **0.870** |
| Baseline (DummyClassifier) PR-AUC | 0.034 — a **26×** lift |
| ROC-AUC | 0.982 (reported for completeness; flattering at 3.4% positives) |
| Tuned threshold | 0.249, chosen for recall ≥ 0.85 |
| Recall | 0.868 — caught **59 of 68** real failures |
| Precision | 0.747 — 20 false alarms across 79 alerts |
| Accuracy | 0.986 — quoted **only** to dismiss it |

Confusion matrix on 2,000 held-out cycles: TN 1912, FP 20, FN 9, TP 59.

The headline sentence for the deck:
> PR-AUC 0.87 against a 0.03 baseline. At the tuned operating point we catch 87%
> of failures with 75% precision — 59 of 68 real failures, at the cost of 20
> unnecessary inspections across 2,000 cycles.

**Anomaly detection (IsolationForest)** — fit on the 7,729 *normal* training rows
only, because a detector trained on data containing failures learns to treat
failures as normal. PR-AUC 0.355 with no labels used. It is a corroborating
signal in the UI, not the primary detector.

**Root cause (one-vs-rest)**

| Mode | PR-AUC | Test positives |
|---|---|---|
| OSF | 0.988 | 16 |
| HDF | 0.987 | 29 |
| PWF | 0.928 | 13 |
| TWF | 0.070 | 10 |

TWF is near-random on purpose — see the limitation below. RNF is excluded from
the model entirely.

**SHAP** — `TreeExplainer` cached to `models/shap_explainer.joblib` (it is slow
to rebuild). Global importance, mean |SHAP| over 500 test rows:

```
tool_wear_min 1.80 | power_w 1.63 | rotational_speed_rpm 1.59 | torque_nm 0.99
temp_diff_k 0.85 | osf_margin 0.79 | process_temp_k 0.61 | air_temp_k 0.46
wear_torque_min_nm 0.34 | type_encoded 0.10
```

The top features are exactly the engineered physics quantities, which is the
result we want: the model and the rules engine are looking at the same things.

Artefacts: `models/*.joblib` (gitignored — run `python src/train.py`, ~30s).
Committed outputs: `docs/model_metrics.json`, `docs/shap_summary.png`.

### `src/rules_engine.py` — the differentiator, partially built

Deterministic re-derivation of the dataset's documented physics, giving the agent
a second opinion independent of the ML model:

| ML says | Rules say | Confidence | Agent behaviour |
|---|---|---|---|
| failure | a rule fired | HIGH | name the verified cause |
| nominal | nothing fired | HIGH | nominal |
| failure | nothing fired | **CONFLICT** | escalate, **never invent a cause** |
| nominal | a rule fired | **CONFLICT** | hard physical limit overrides the model |

That last column is the answer to "how do you stop the LLM hallucinating a
diagnosis": it is structurally incapable of naming a cause the rules did not
verify.

**HDF is implemented and validated: 115/115 true cases recovered, zero false
positives.** `assess()` and the confidence gate are complete and ready for Dev 2.

### `src/validate_rules.py`

Scores each rule against the ground-truth flag columns across all 10,000 rows.
Current output:

```
MODE    TRUE  FLAG   TP    FP   FN    PREC     REC      F1
HDF      115   115  115     0    0   1.000   1.000   1.000
TWF/PWF/OSF                          NOT IMPLEMENTED
```

This table is likely the strongest evidence slide in the deck — "our rules
recover 115 of 115 heat-dissipation failures with zero false positives" is not
an opinion.

### `notebooks/01_eda.ipynb`

Executed with outputs committed, so it renders on GitHub without anyone re-running
it. Establishes six facts, two of which are new and matter:

- **9 rows are labelled `Machine failure = 1` with no mode flag set at all.**
  They are unexplained by construction. This puts a hard ceiling on the rules
  engine: no deterministic system can explain more than 330 of 339 failures.
  Say this before a judge finds it.
- **A deterministic tool-wear rule cannot exceed ~0.05 precision.** 790 rows sit
  in the 200–240 min wear window; only 43 of them actually fail, because the
  failure point is drawn at random inside the window. Reporting that honestly
  reads as more credible than tuning it away.
- 18 further rows carry an RNF flag that did not escalate to a failure.

---

## Open items — owner and deadline

### Blocking, for Dev 1 (me), next

| Item | Detail |
|---|---|
| `check_pwf` | Power outside [3500, 9000] W. Note it is an **OR** (two-sided band), unlike HDF's AND. The evidence string should say which side was breached — a stall and an overdrive get different maintenance actions. |
| `check_osf` | wear×torque above 11,000 / 12,000 / 13,000 minNm for L / M / H. This is the mode the demo escalates on, so its strings matter most. |
| `check_twf` | Wear inside the 200–240 min window. Label it a **risk window**, not a verdict. Expect ~0.93 recall and ~0.05 precision, and report that. |

Each has a full spec in its docstring in `src/rules_engine.py`. Run
`python src/validate_rules.py` after each one — HDF, PWF and OSF must all score
1.000/1.000 or the threshold is wrong.

### Blocking other people

| Item | Owner | Due | Impact if late |
|---|---|---|---|
| `docs/maintenance_playbook.md` | Non-tech A | **H+10** | Dev 2 cannot build `lookup_playbook()`. Fallback: hardcode a Python dict, drop RAG. Currently 0 bytes. |
| `docs/cost_model.md` | Non-tech A | **H+14** | `estimate_cost()` has no defensible numbers, and "cost avoided" is one of the five things that wins this. |
| `src/agent/tools.py` | Dev 2 | **H+8** | `rules_engine.assess()` is ready to be wrapped now — it needs no playbook. Start there. |
| `src/app/` | Dev 3 | **H+2** | Blocked on the frontend decision below. |
| `docs/architecture.png` | Non-tech B | early | 0 bytes. Needed for the deck. |
| `docs/demo_script.md` | Non-tech B | **H+24** | Then three rehearsals. |
| Backup demo video | Non-tech B | **H+30** | Non-negotiable per the plan. |

### Open decisions — nobody has called these yet

1. **Frontend: Path A (React/Lovable + FastAPI) or Path B (Streamlit)?**
   Path A looks visibly better but adds a FastAPI layer that is **not assigned to
   anyone**. Dev 3 was due to start at H+2 and cannot start without this.
   Decide it now, not at H+6.
2. **Playbook RAG or a plain dict?** If the playbook lands late, ChromaDB and
   sentence-transformers get cut. A dict lookup over four failure modes is
   honestly hard to beat here anyway.

Both are commented out in `requirements.txt` until they are resolved.

---

## Named limitations — put these on a slide, do not hide them

1. **Synthetic data.** Physically grounded, but not a real production line.
2. **No timestamps, no machine IDs.** Rows are independent snapshots, not a time
   series. `UDI` is used purely as a replay index to make the demo look live.
3. **RNF is irreducible.** 0.1% random by construction, no sensor signature,
   excluded from the root-cause model by design.
4. **9 failures have no cause label.** Ceiling of 330/339 on rule-based
   explanation.
5. **Tool-wear rule is a risk window, not a verdict.** ~0.05 precision, and that
   is the mathematical maximum, not a tuning failure.
6. **Probabilities are not calibrated.** `scale_pos_weight` distorts them, so the
   "87%" in a work order is a ranking score, not a literal frequency. Isotonic
   calibration is a 10-minute fix if a judge presses on it.

---

## How to run

```bash
python -m venv venv
source venv/Scripts/activate      # Git Bash on Windows; use venv/bin/activate on Linux/macOS
pip install -r requirements.txt

python src/features.py            # sanity check: dataset + feature summary
python src/validate_rules.py      # score physics rules vs ground truth
python src/train.py               # train everything, ~30s, writes models/ and docs/
jupyter lab notebooks/01_eda.ipynb
```
