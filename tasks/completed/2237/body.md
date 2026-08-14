---
title: No gate checks that a plan's backend lane can mint N distinct pod names for
  an N-way fan-out (RunPod lane is per-issue; --lane-suffix does not cover it)
kind: infra
tags:
- wf-fix
created_at: '2026-08-11T21:04:17Z'
has_clean_result: false
origin_prompt: 'Surfaced during #2054 reduced-basis-refit-rungs789 dispatch: plan
  v16 section 9 specified 10 parallel RunPod shards via dispatch_issue.py launch,
  but backends/runpod.py:264 hardcodes pod-<issue> with no suffix and --lane-suffix
  explicitly excludes RunPod pod names. The plan passed verify_plan twice (0/0), both
  critic lenses APPROVE, and two code-review rounds; caught only at dispatch.'
workflow: v1
---
## Goal

No gate asks whether a plan's chosen backend lane can actually mint **N distinct pod/instance names** for an N-way concurrent fan-out. A plan can therefore specify an N-shard parallel fleet, pass every gate cleanly, and be undispatchable as written. Add a check (and/or the reviewer-lens clause) that catches it at plan time.

## The gap (verified in code, not inferred)

Two facts, both read directly:

1. `src/explore_persona_space/backends/runpod.py:264-271` — `_runpod_pod_name(issue)` returns a hardcoded `f"pod-{issue}"`. It accepts no suffix parameter. The router threads a name suffix into **SLURM only** (`backends/slurm.py` `job_name_suffix`, ~:258/:328/:497/:779); there is no RunPod equivalent anywhere in `backends/`.
2. `scripts/dispatch_issue.py launch --lane-suffix` does **not** close it, and says so in its own help text verbatim: *"GCP instance names AND SLURM job names + scratch dirs carry the suffix (#2055); **RunPod pod names remain per-issue (two lanes failing over to RunPod still contend on `pod-<N>`)**."* `--lane-suffix` distinguishes the handle sidecar (`issue-<N>-<suffix>-handle.json`) and GCP/SLURM names — not RunPod pod names.

Consequence: **N concurrent `dispatch_issue.py launch --backend runpod` commands for one issue all resolve to the same pod name.** `pod.py provision` refuses a duplicate pod for a task absent `resume`, an approved terminate, or `--name-suffix`, so the realized outcomes are collision, error, or — the dangerous one — silent co-location of all N shards on ONE pod, which invalidates every per-shard wall/RSS projection derived from a per-pod basis.

The correct mechanism for an N-way RunPod fan-out is `pod.py provision --name-suffix <slug>` per shard (which yields `pod-<N>-<slug>`, the documented second-pod convention and what per-pod `epm:run-launched` shields (#1961) and surgical `pod.py terminate --name-suffix` both require).

## How it surfaced (#2054 round `reduced-basis-refit-rungs789`)

Plan v16 §9 specified **10 parallel `cpu-bigmem` shards**, with launch commands of the form `dispatch_issue.py launch --issue 2054 --intent cpu-bigmem --backend runpod ... --workload-cmd "..."` repeated per shard — and, in its **own** pod-safety paragraph, the naming convention `pod-2054-rb789-<shard>` plus a per-pod `epm:run-launched` shield. Those two clauses are mutually unsatisfiable: the router path cannot produce the names the pod-safety clause requires.

The plan cleared, in order: plan approval; `verify_plan.py` **PASS 0 FAIL / 0 WARN of 57 checks** (twice — v14 and v16); a critic round 1 (REVISE on 3 unrelated blockers) and round 2 (**APPROVE on both the Methodology and Statistics lenses**); and two code-review rounds. It was caught only at dispatch, by the experimenter refusing a fresh-provision chain for an unrelated reason (`fresh-provision-in-subagent`), which forced the orchestrator to read the launch path closely. Full record: #2054 `epm:progress` **v225**.

Why each gate missed it:
- `verify_plan.py` **c46** dry-parses plan-embedded `dispatch_issue.py` commands against the live `build_argparser()`. The commands **parse fine** — every flag is valid. The defect is semantic (N concurrent same-issue launches contend on one pod name), which c46 by construction cannot see.
- **c52** (fan-out RAM/GPU-mem floor) reasons about per-leg RSS/VRAM, not name minting.
- The **Methodology** lens reviewed compute sizing, per-family pilots, shard width, staging shape, fence adequacy, reuse fitness — the lens rubric has no clause about whether the lane can name the fan-out.
- The **Statistics** lens is out of scope for dispatch mechanics by design.

## Deliverable

1. **A `verify_plan.py` check (WARN-first).** Fire when a plan declares an N-way concurrent fan-out (N > 1 same-issue launches / a shard registry / an explicit "N parallel pods" claim) whose resolved lane is RunPod — an explicit `--backend runpod` pin, or `auto` (which leads with runpod under #2054's runpod-first order) — and the plan names no per-pod naming mechanism (`pod.py provision --name-suffix`, or an equivalent). `--lane-suffix` present must NOT satisfy it: that is the specific false-comfort this check exists to remove. Follow the c-check calibration contract — dry-run against the persisted-plan corpus and report true/false-positive counts before shipping; ship WARN-only first.
2. **A Methodology-lens clause** (`.claude/rules/critic-lens-reference.md`, alongside the item-16 extension family): for any N-way fan-out, the plan must name a mechanism that mints N distinct pod/instance names on the resolved lane, and the plan's own naming convention must be producible by its own launch commands. The #2054 shape — launch commands and pod-safety clause specifying incompatible naming — is the worked example.
3. **Consider a runtime guard.** `dispatch_issue.py launch` could refuse (typed) when a RunPod-resolved launch would target an existing managed `pod-<N>` without an explicit reuse/suffix intent, rather than colliding or silently reusing. Decide and record; if declined, say why in `--lane-suffix`'s help so the next reader does not re-open it.
4. **Consider threading a name suffix into the router's RunPod lane** so `--lane-suffix` means what a reader naturally assumes. This is the deepest fix and the largest blast radius (pod naming feeds `_MANAGED_PREFIXES`, the watcher's pod-safety pass, `pod.py terminate --issue`, and the stale-pod audit) — scope it deliberately, and do NOT bundle it with (1)+(2) if it risks the lifecycle surfaces. Adjudicate and record either way.

## Acceptance

- The new check fires on #2054 plan v16 §9 (10 RunPod shards, no per-pod naming mechanism, `--lane-suffix` absent) and does NOT fire on a genuine single-launch RunPod plan or on a suffixed `pod.py provision` fan-out.
- Corpus calibration counts reported before the check ships non-WARN.
- `uv run python scripts/workflow_lint.py` passes, including `--check-lens-coverage` if a lens item is added.
- `--lane-suffix`'s help text and any new check agree about what it does and does not cover.

## Provenance

Surfaced 2026-08-11 during #2054 round `reduced-basis-refit-rungs789` dispatch, after the plan had passed `verify_plan` twice, both critic lenses, and two code-review rounds. Filed per `.claude/rules/workflow-fix-on-bug.md`. Both underlying facts were verified by direct code read (`backends/runpod.py:264-271`; `dispatch_issue.py launch --help`) before filing, not inferred from the failure. Sibling filed the same day: #2236 (a rule delegating enforcement to a critic-lens item that does not exist) — same family, different mechanism: #2236 is a dangling pointer, this is a missing check.
