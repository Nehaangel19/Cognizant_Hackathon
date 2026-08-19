# Maintenance Playbook

> **Owner:** Non-tech A deliverable — drafted by Dev 1 to unblock Dev 2's
> `lookup_playbook()`. Expand the plain-English prose; keep the structured fields
> in `docs/playbook.json`, which is what the agent actually retrieves. If the RAG
> path (ChromaDB) is cut at the H+16 gate, the agent loads the JSON as a dict and
> nothing else changes.

This is the reference the agent consults after the rules engine has **verified** a
root cause. The agent never invents an action — it looks it up here, keyed by the
mode the physics rules confirmed. Each entry answers: what is happening, what to
do, what parts, how long, how urgent, and what it costs to ignore.

---

## TWF — Tool Wear Failure

**What is happening.** The cutting tool has reached the end of its usable life.
Edges are worn, cutting forces rise, surface finish degrades and the tool is about
to break in the cut. In the data, the failure point is drawn at random between
200 and 240 minutes of accumulated wear, so this is a *risk window*, not a
certainty — the agent treats a lone tool-wear signal as an advisory, not a
line-stop.

**Recommended actions.** Schedule a tool change at the next planned stop.
Inspect the cutting edge and the finish on the last few parts. Reset the tool-wear
counter after replacement.

**Parts.** Replacement cutting tool / insert set.

**Repair time.** ~0.5 hr (a routine tool change).

**Urgency.** LOW — schedule, do not stop the line. Escalates if it coincides with
rising torque (see OSF).

**Consequence of ignoring.** The tool breaks mid-cut, scrapping the workpiece and
potentially marking the spindle — which turns a 30-minute planned swap into an
unplanned OSF-class event.

---

## HDF — Heat Dissipation Failure

**What is happening.** The process is not shedding heat fast enough. The
temperature difference between process and air has fallen below 8.6 K while the
spindle turns below 1380 rpm, so friction heat accumulates instead of dissipating.
Left alone, thermal expansion pushes parts out of tolerance and bearings degrade.

**Recommended actions.** Check coolant flow and concentration. Clear any blocked
coolant lines or clogged filters. Verify the chiller / heat exchanger is running.
Reduce spindle load or raise speed to restore airflow if coolant is nominal.

**Parts.** Coolant, coolant filter; heat exchanger service kit if the chiller is
implicated.

**Repair time.** ~2.5 hr (coolant service and thermal re-stabilisation).

**Urgency.** MEDIUM — address within the shift. Thermal drift silently ruins
dimensional accuracy before it ever trips a hard fault.

**Consequence of ignoring.** Out-of-tolerance parts pass inspection and reach the
customer; sustained overheating shortens spindle-bearing life.

---

## PWF — Power Failure

**What is happening.** Mechanical power (torque × angular velocity) has left its
safe band of 3,500–9,000 W. **Under-power** (< 3,500 W) means the tool is stalling
in the cut — often a feed/speed mismatch or a binding tool. **Over-power**
(> 9,000 W) means the drive is being overdriven, stressing the spindle motor and
electronics. The two get opposite fixes, so the agent reports which side breached.

**Recommended actions.**
- *Under-power / stall:* raise spindle speed or reduce feed to unload the tool;
  check for a binding or dull tool.
- *Over-power / overdrive:* reduce feed rate; inspect the spindle drive, motor
  current and belt tension; check for a seized axis.

**Parts.** Usually none for a parameter fix; spindle drive / motor inspection kit
if over-power persists.

**Repair time.** ~4.0 hr (electrical inspection and re-qualification).

**Urgency.** HIGH — investigate promptly. A drive pushed past its envelope can
fail suddenly and take the spindle with it.

**Consequence of ignoring.** Spindle-motor or drive burnout — a multi-hour,
multi-thousand-dollar replacement instead of a parameter change.

---

## OSF — Overstrain Failure

**What is happening.** The product of tool wear and torque has exceeded the
overstrain limit for the product variant — 11,000 minNm for L, 12,000 for M,
13,000 for H. A worn tool is being pushed at high torque; the combined mechanical
strain is enough to snap the tool and damage the workpiece or spindle. **This is
the most severe mode and the one the demo escalates on.**

**Recommended actions.** Stop the operation before the next cut. Replace the
cutting tool. Reduce feed rate by ~15% to bring wear×torque back under the limit.
Inspect the workpiece and spindle for damage before resuming.

**Parts.** Replacement cutting tool; workpiece if already marked; spindle
inspection.

**Repair time.** ~6.0 hr (tool change, spindle inspection, first-part
re-qualification).

**Urgency.** CRITICAL — stop now. This is the failure most likely to cause
collateral damage beyond the tool itself.

**Consequence of ignoring.** Catastrophic tool fracture: scrapped part, possible
spindle damage, and the longest unplanned downtime of any mode.

---

## RNF — Random Failure (for completeness)

**What is happening.** A random failure with no sensor signature — 0.1% chance
regardless of parameters, irreducible by construction. The agent cannot and does
not attempt to diagnose a cause; it flags for human inspection.

**Recommended actions.** Manual inspection. There is nothing to predict.

**Urgency.** N/A — cannot be predicted. Belongs on the limitations slide, not in
the model.
