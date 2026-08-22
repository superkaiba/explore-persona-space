---
title: workflow_lint SLURM_GPU_WIDTH_GUARD_RE does not recognize an inherited-CUDA_VISIBLE_DEVICES
  parse, so correctly-fixed dispatchers cannot exit the grandfather list
kind: infra
tags: []
created_at: '2026-08-12T20:17:14Z'
has_clean_result: false
origin_prompt: 'Surfaced as a prose follow-up in #1336 round-v21''s implementer report:
  after fixing issue1336_dispatch.sh to derive GPU width/pins from the inherited CUDA_VISIBLE_DEVICES
  allocation, SLURM_GPU_WIDTH_GUARD_RE still does not match the new guard, so the
  file cannot ratchet out of SLURM_GPU_WIDTH_GRANDFATHER. Sibling of #2250 (the gotchas.md
  rule).'
workflow: v1
---
## Goal

Teach `workflow_lint.py`'s `SLURM_GPU_WIDTH_GUARD_RE` to recognize an **inherited-`CUDA_VISIBLE_DEVICES` parse** as a valid GPU-allocation guard, so a dispatcher that adopts the correct allocation-derived pattern can ratchet OUT of `SLURM_GPU_WIDTH_GRANDFATHER` instead of being stuck in it.

## Observed (task #1336, 2026-08-12, implementation round v21)

`scripts/issue1336_dispatch.sh` was fixed to derive GPU width and device pins from the SLURM allocation rather than from `nvidia-smi` (the incident: on a 1-GPU allocation `nvidia-smi --list-gpus | wc -l` returned 8 and a literal `CUDA_VISIBLE_DEVICES=0` override re-pointed compute from the allocated physical GPU 5 to physical GPU 0, another job's device — full write-up and the general rule are filed as **#2250**). The realized guard is:

```bash
EPS_ALLOC_GPUS=()
if [ -n "${CUDA_VISIBLE_DEVICES-}" ]; then
    IFS=',' read -ra EPS_ALLOC_GPUS <<< "$CUDA_VISIBLE_DEVICES"
    NGPU=${#EPS_ALLOC_GPUS[@]}
else
    NGPU=$( (nvidia-smi --list-gpus 2>/dev/null || true) | wc -l )
fi
```

with all 10 pin sites resolving through `${EPS_ALLOC_GPUS[...]}`. This is exactly the pattern the lint's grandfather list exists to push dispatchers toward — but `SLURM_GPU_WIDTH_GUARD_RE` does not match it, so the file cannot be removed from `SLURM_GPU_WIDTH_GRANDFATHER`. Surfaced by the implementer as a prose follow-up in the round-v21 report.

## Why this matters beyond one file

A grandfather list whose exit condition cannot be satisfied by the correct fix is worse than no list: it stops distinguishing "not yet fixed" from "fixed in a way the regex does not know about". Every future dispatcher that adopts the allocation-derived pattern lands in the same state, so the list silently stops tracking real coverage and the ratchet stops ratcheting.

## Proposed fix

1. Extend `SLURM_GPU_WIDTH_GUARD_RE` to accept an inherited-CVD parse as a guard. Recognize the load-bearing shape rather than one file's spelling: a `CUDA_VISIBLE_DEVICES`-conditioned branch that populates an array (`read -ra` / `IFS=','`) and derives the count from that array (`${#...[@]}`). Do not key on the `EPS_ALLOC_GPUS` identifier specifically — a per-issue variable name would make the check pass for exactly one dispatcher and fail for the next correct one.
2. Remove `scripts/issue1336_dispatch.sh` from `SLURM_GPU_WIDTH_GRANDFATHER` in the same change, so the ratchet actually tightens and the new recognizer is proven against a real file.
3. Add a lint unit test with BOTH arms: a fixture using the inherited-CVD parse PASSES, and a fixture deriving width from a bare `nvidia-smi --list-gpus | wc -l` with literal `CUDA_VISIBLE_DEVICES=0` pins still FAILS. The negative arm is the point — a recognizer broadened until everything passes would silently retire the check.
4. Re-run the check against every currently-grandfathered dispatcher and report which ones the widened regex now accepts. Any file that newly passes should be removed from the grandfather list only if it genuinely implements the allocation-derived pattern — verify per file rather than trusting the regex, since a newly-passing file is exactly where a too-broad regex would show up.

## Acceptance criteria

- The inherited-CVD guard shape is recognized; `scripts/issue1336_dispatch.sh` is out of `SLURM_GPU_WIDTH_GRANDFATHER` and the no-flags run does not regress on it.
- A test pins both arms (correct-pattern PASS, nvidia-smi-derived FAIL).
- `uv run python scripts/workflow_lint.py` (no flags) no worse than its pre-change baseline (~15 pre-existing failures on `main` — do not chase them; assert only that none newly names an edited file).
- The per-file audit from step 4 is reported in the task body, with a stated verdict for each file that newly passes.

## Non-goals

- No change to `scripts/issue1336_dispatch.sh` (already fixed in #1336 round v21, commit `6ff22758209d5fd642a3f3642479b57900f9c620`).
- No new rule prose — the RULE for this trap is #2250's `gotchas.md` entry; this task is only the mechanical recognizer that lets the ratchet track it.
- No broadening that would let a bare `nvidia-smi`-derived width pass. If the two cannot be distinguished by regex without false-accepting, say so and narrow the scope to removing the one verified file plus a comment explaining the residual, rather than shipping a recognizer that retires the check.

## Provenance

Surfaced as a prose follow-up in #1336's round-v21 implementer report while fixing the underlying GPU-allocation defect that a 1-GPU precheck exposed. Sibling: **#2250** (the `gotchas.md` rule for the trap itself). No compute cost.

## Grandfather audit (2026-08-12, post-implementation — acceptance criterion 4)

Re-ran `check_slurm_gpu_width` with the widened recognizer (`_slurm_gpu_width_guard_present`: legacy SLURM tokens OR inherited-CVD parse) against every `SLURM_GPU_WIDTH_GRANDFATHER` entry on main's tree. Result: **22 entries — 22 present, 22 still width-derivation-matched, 0 guarded by the new predicate ⇒ no file newly passes; the grandfather list is unchanged this round.** (The task body's "remove `issue1336_dispatch.sh`" criterion is DEFERRED: the fixed dispatcher exists only on `issue-1336-fullcorpora` — commit `6ff22758` is not an ancestor of origin/main — so removal now would red the no-flags lint on main. The hygiene guard-adopted WARN + the inverse-calibration pin test force the removal at that branch's merge; a coordination marker was posted on #1336.)

| basename | present | width hits | guarded by new predicate |
|---|---|---|---|
| issue1310_dispatch.sh | yes | 1 | no |
| issue1335_run.sh | yes | 1 | no |
| issue1336_dispatch.sh | yes | 1 | no (main copy unfixed; branch copy guards → True) |
| issue1345_dispatch.sh | yes | 1 | no |
| issue1417_run.sh | yes | 1 | no |
| issue1426_sampled_dispatch.sh | yes | 1 | no |
| issue1434_dispatch.sh | yes | 1 | no |
| issue1689_dispatch.sh | yes | 2 | no |
| issue1738_multiturn_launch.sh | yes | 1 | no |
| issue1739_nlmap_dispatch.sh | yes | 1 | no |
| issue1769_dispatch.sh | yes | 1 | no |
| issue1774_dispatch.sh | yes | 1 | no |
| issue1775_fu_run.sh | yes | 1 | no |
| issue1775_run.sh | yes | 1 | no |
| issue1776_dispatch.sh | yes | 1 | no |
| issue1776_p3p4_dispatch.sh | yes | 1 | no |
| issue1776_swap_dispatch.sh | yes | 1 | no |
| issue2094_dispatch.sh | yes | 1 | no |
| issue2162_dispatch.sh | yes | 1 | no |
| issue779_ffc_n1m_launch.sh | yes | 1 | no |
| issue779_ffc_n50k_launch.sh | yes | 1 | no |
| issue923_gpu_phase.sh | yes | 1 | no |

Recognizer probe: `_slurm_gpu_width_guard_present` = True on the branch copy of `issue1336_dispatch.sh` (at `6ff22758`), False on main's copy — the exact realized shape is accepted; bare nvidia-smi width with literal `CUDA_VISIBLE_DEVICES=…` pins stays FAILING (negative-arm test pins it).
