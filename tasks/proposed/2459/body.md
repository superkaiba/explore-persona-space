---
title: 'workflow-fix: session-scoped pgrep misses the gate''s start_new_session pytest
  worker — healthy Step 9c/10d gates read as wedged'
kind: infra
tags:
- wf-fix
created_at: '2026-08-21T22:49:44Z'
has_clean_result: false
origin_prompt: 'surfaced during /issue 2260 Step 10d lint-gate health read: pgrep
  -s <gate pid> reported 0 pytest procs on a healthy gate whose worker was in its
  own session; py-spy prevented a kill-before-relaunch'
workflow: v1
---
kind: infra

## Goal

Document — and correct the prescribed probe form for — the fact that a SESSION-SCOPED liveness probe (`pgrep -s <gate pid>`) on a Step 9c / Step 10d gate structurally cannot see the gate's actual pytest worker, so an orchestrator health read can conclude WEDGE on a perfectly healthy gate and license killing multi-hour work under `crash-fix-rounds.md § Kill-before-relaunch`.

## The mechanism (measured, not inferred)

`step9c_baseline.py`'s `run_pytest` and `_mapped_baseline_pytest` spawn their pytest child with `start_new_session=True` (the docstring of `_killpg_bounded` states this explicitly: "the `start_new_session=True` pytest children ... a bare kill of the direct child orphans its own pytest workers, so the group is signalled"). A session leader is by definition NOT in its parent's session, so `pgrep -s <gate-script-pid>` — the form the Step 9c 1b and Step 10d launcher recipes already use for the choom sweep — returns the wrapper chain and NOTHING ELSE.

Measured live during the #2260 Step 10d pre-push lint gate (2026-08-21, gate pid 92763):

```
pid=92763   sid=92763   adj=-600  bash      (gate script)
pid=399804  sid=92763   adj=-600  bash
pid=399805  sid=92763   adj=-600  timeout
pid=399806  sid=92763   adj=-600  uv
pid=399809  sid=92763   adj=-600  python3   (step9c_baseline.py mapped-baseline)
pid=402817  sid=402817  adj=-600  python    (THE ACTUAL pytest worker)
```

`pgrep -s 92763` → `92763 399804 399805 399806 399809`. Membership test for 402817: **0**. Meanwhile 402817 held 10:42 of CPU in state `Rsl` and the leg's output file was growing.

## Why it reads as a wedge (the compounding second signal)

The parent (399809) sits in `subprocess.wait(timeout=...)`, and CPython's timeout-bearing wait is a **sleep-poll loop**, so `/proc/<pid>/wchan` reads `hrtimer_nanosleep` and the parent's own CPU time and IO counters stay STATIC indefinitely. An orchestrator running the prescribed output-growth health read therefore sees, simultaneously:

- zero pytest processes under the gate session,
- a parent sleeping rather than running,
- static `read_bytes`/`write_bytes`/`wchar` on that parent,
- a gate log that has not grown in ~20 min (the leg logs only at leg boundaries).

Every one of those is the documented signature of a wedge, and all four are FALSE here. The only instrument that disambiguates is a Python-level stack (`sudo -n env "PATH=$PATH" py-spy dump --pid <parent>`), which shows `_wait -> wait -> _mapped_baseline_pytest` and settles it in one call.

## Scope limit — this is NOT an earlyoom-protection bug

Worth stating so the fix does not overreach: `oom_score_adj` INHERITS across fork and `start_new_session` does not reset it, so the out-of-session pytest child measured `adj=-600` anyway (inherited from the workload's `sudo -n choom -n -600 -p $$` self-choom, which runs before any fork). Protection is intact. The `pgrep -s` sweep's residual value is only the #1315 case (a child forked before the parent's adjustment landed). **Do not "fix" the choom sweep on the strength of this task** — the defect is confined to LIVENESS/HEALTH READS.

## Fix direction (implementer to confirm the right shape)

1. Add a gotchas.md entry in the EXISTING doctrine family — line 95 already teaches "reap by RELATIONSHIP ... never by NVML-pid identity" for the container-pid-namespace twin; this is the session-namespace sibling. Prescribe the correct probe forms for a gate health read:
   - relationship-based: `ps --ppid <parent-pid> -o pid=,sid=,time=,stat=,args=`
   - cmdline-anchored: `pgrep -af 'pytes[t]'` (bracketed per the existing self-match rule)
   - and, for a parent that looks asleep, `py-spy dump --pid <parent>` as the disambiguating call, noting it needs `sudo -n env "PATH=$PATH" py-spy` (a bare `sudo py-spy` fails `command not found` — py-spy lives in `~/.local/bin`, not on the sudo PATH).
2. Add one sentence to the Step 9c 1b and Step 10d gate health-read guidance: a gate's pytest worker runs in its OWN session; `pgrep -s` is for the choom sweep ONLY and is never a liveness verdict. State explicitly that a parent in `hrtimer_nanosleep` with static IO is the EXPECTED shape of a healthy `subprocess.wait(timeout=...)`.
3. Keep the existing `pgrep -s` choom-sweep call sites byte-unchanged (see the scope limit above).

## Acceptance criteria

1. A gotchas.md entry (or Step 9c/10d guidance) names the `start_new_session` session-scoping blind spot and the three correct probe forms.
2. The guidance states that `hrtimer_nanosleep` + static parent IO is expected for a healthy timeout-bearing `subprocess.wait`, not evidence of a wedge.
3. No change to any `pgrep -s ... | xargs choom` call site.
4. Pinned by a test so the guidance cannot silently rot (the repo's prose-pin convention).

## Provenance

Surfaced during `/issue 2260` Step 10d (2026-08-21) while health-reading the pre-push lint gate at ~46 min elapsed. The orchestrator's own session-scoped probe reported "0 pytest procs" on a healthy gate and the run was one step from a kill-before-relaunch of a 46-minute-old healthy gate; `py-spy` is what prevented it. Fingerprint: (session-scoped-pgrep-misses-start-new-session-child, gate-liveness-health-read).
