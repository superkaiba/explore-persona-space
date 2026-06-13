---
name: Agent-spec gates vs modal launch lane
description: experimenter.md checklist gates never fire on GCP/SLURM startup-script launches; the implementer write-side rule is the only all-lane prose control — never trimmable (#578)
type: feedback
---

When an infra plan claims class-recurrence prevention via an agent-spec checklist item (a new `experimenter.md` "Before Running" gate), verify the agent that loads that file actually RUNS on the project's MODAL launch path. Under the GCP-FIRST auto default, GCP/SLURM workloads launch via the router's rendered startup script (`--workload-cmd`, `GcpBackend.launch`); the experimenter posts `epm:run-launched` on the RunPod path ONLY. An experimenter-side gate covers only the opt-in RunPod lane; the write-side rule in `experiment-implementer.md` is the only prose control travelling with the dispatcher across all lanes.

**Why (#578 v1):** the plan framed the experimenter item as "the mandated deliverable" and the implementer write-side bullet as trimmable, while assuming "experimenter is the sole launch role" (unverified) — on the default lane that left ZERO new control if trimmed; the fatal recurrence path is a NEW dispatcher routed to GCP ft-7b with no agent gate and no family smoke.

**How to apply:** for any plan whose protection is a checklist edit, ask (1) which lanes spawn the agent that loads the edited file; (2) whether the lane-independent control (write-side rule, lint check, regression test) is bound as mandatory; (3) whether a mechanical control (workflow_lint rule on backgrounded launch lines, pre-dispatch grep) is named as follow-up. Grep-based verification false-passes on comment-only mentions — prefer a shape-anchored grep on the launch lines.
