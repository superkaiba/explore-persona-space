---
name: Agent-spec gates vs modal launch lane
description: Doc-gate interventions (experimenter.md checklist items) never fire on the GCP/SLURM startup-script lanes - check WHO loads the doc on the DEFAULT lane before crediting prevention
type: feedback
---

When an infra plan claims class-recurrence prevention via an agent-spec
checklist item (e.g. a new `experimenter.md` "Before Running" gate), verify the
agent that loads that file actually RUNS on the project's MODAL launch path.
Under the GCP-FIRST auto default, GCP/SLURM workloads launch via the router's
rendered startup script (`--workload-cmd`, `GcpBackend.launch`) — `epm:run-launched`
is posted by the experimenter **on the RunPod path only** (SKILL.md Step 6b/6d,
~lines 2414, 2535-2540). So an experimenter-side gate covers only the opt-in
RunPod lane; the write-side rule in `experiment-implementer.md` is the ONLY
prose control that travels with the dispatcher across all lanes.

**Why:** Task #578 plan v1 framed the experimenter item as "the mandated
deliverable" and the implementer write-side bullet as trimmable ("Item 3a alone
satisfies the task body"), while assumption 11 claimed "experimenter is the sole
launch role" (Medium, unverified). On the default lane that left zero new
control if trimmed — the fatal recurrence path is a NEW dispatcher routed to
GCP ft-7b (4x A100) with no agent gate and no family smoke.

**How to apply:** For any plan whose protection is a checklist edit, ask
(1) which lanes spawn the agent that loads the edited file; (2) whether the
lane-independent control (write-side rule, lint check, regression test) is
bound as mandatory; (3) whether a mechanical control (workflow_lint rule on
`scripts/*dispatch*.sh` backgrounded launch lines, or a Step-6b pre-dispatch
grep) is named as follow-up. Grep-based verification commands also false-pass
on comment-only mentions — prefer a shape-anchored grep on the launch lines.
