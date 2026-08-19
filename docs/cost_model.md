# Cost Model — Downtime Avoided per Failure Caught

> **Owner:** Non-tech A deliverable — drafted by Dev 1 to unblock `estimate_cost()`.
> Expand the prose, keep the numbers in `docs/cost_params.json` (that file is what
> the agent actually reads, so the demo and this document can never disagree).

## Why this exists

The brief says machine downtime "impacts production, quality and cost." A model
that only prints a probability answers two of those three. Translating a caught
failure into **money saved** is one of the five things that wins this (handoff
section 10). Judges remember one dollar figure said out loud; they forget an
F1 score.

## The headline number we defend

**A stopped production CNC/milling cell costs ~$2,600 per hour of unplanned
downtime.** Every avoided failure of average length (≈2.5 hrs to recover) is
therefore worth roughly **$6,500** in avoided downtime, before counting scrapped
work-in-progress or a damaged spindle.

We say "~$2,600/hr for a mid-market cell," not "$22,000/min," on purpose. The
famous $22k/minute figure is a full automotive final-assembly line where one
stoppage idles hundreds of downstream stations ([Advanced Technology Services /
Nielsen survey, 2006, widely cited](https://www.manufacturing.net/home/article/13055083/the-22000perminute-manufacturing-problem)).
Quoting it for a single CNC cell would be indefensible the moment a judge does the
arithmetic. We pick a figure we can stand behind.

## How the per-hour figure is built

Layered, so any judge can follow it and swap their own numbers in:

| Component | Rate | Basis |
|---|---|---|
| Direct machine burden | $120 / hr | Published shop rate for a 5-axis VMC ([Toolhive, 2025](https://www.toolhivesolutions.com/blog/cnc-machine-downtime-costs)) |
| Machines idled per stoppage in the cell | × 4 | A small cell of four machines sharing the affected process |
| Operator + maintenance labour | + $180 / hr | 3 people at a loaded ~$60/hr while the line is down |
| Lost contribution margin on unbuilt parts | + $1,200 / hr | Throughput × unit margin for a mid-volume cell |
| **Raw idle cost** | **$1,860 / hr** | sum of the above |
| True-downtime multiplier | × 1.4 | Setup friction, thermal ramp-up, re-qualification of first parts after restart ([Toolhive uses ×1.8 for a single machine](https://www.toolhivesolutions.com/blog/cnc-machine-downtime-costs); we use a conservative ×1.4) |
| **Loaded downtime cost** | **≈ $2,600 / hr** | the figure `estimate_cost()` uses |

Every input is in `docs/cost_params.json`. Change one number there and the whole
model — and the live demo — updates.

## Recovery time is not free either

Independent of the per-hour rate, restarts have been getting *slower*: average
time to resume production has climbed from 49 minutes to 81 minutes over five
years ([Siemens, via Toolhive](https://www.toolhivesolutions.com/blog/cnc-machine-downtime-costs)).
That is why "predict it before it stops the line" beats "fix it fast."

## Per-mode downtime avoided

Each failure mode has a different recovery profile. `estimate_cost()` multiplies
the per-hour figure by the mode's expected downtime, and adds the mode-specific
collateral (a scrapped workpiece, a damaged tool or spindle) from the playbook.

| Mode | Est. downtime if it runs to failure | Downtime cost @ $2,600/hr | Typical collateral |
|---|---|---|---|
| TWF — tool wear | 1.0 hr | $2,600 | Scrapped part in progress |
| HDF — heat dissipation | 2.5 hr | $6,500 | Coolant system service |
| PWF — power failure | 4.0 hr | $10,400 | Drive / electrical inspection |
| OSF — overstrain | 6.0 hr | $15,600 | Broken tool + possible spindle damage |

These downtime hours live in `docs/cost_params.json` alongside the rate, and the
repair-time and parts detail comes from `docs/maintenance_playbook.md`.

## The arithmetic for the pitch

> Downtime avoided this shift = (failures caught early) × (avg downtime hrs) × ($/hr)

Worked example from our own eval, using the tuned model that catches 59 of 68
test failures:

> 59 failures caught × 2.5 hr avg × $2,600/hr ≈ **$383,000** in avoided downtime
> across 2,000 production cycles — against 20 false alarms, each costing one
> ~15-minute inspection (≈$650 total). Net avoided cost dwarfs the false-alarm
> bill by ~590×.

That ratio — money saved vs. money wasted on false alarms — is the honest way to
justify tuning for recall over precision.

## Limitations we state openly

- The dataset is synthetic, so these dollar figures are an illustrative model
  layered on top of it, not measured plant data.
- We use a single blended $/hr. A real deployment would set it per line.
- Collateral costs (scrap, spindle) are order-of-magnitude, from the playbook,
  not from invoices.
