---
title: 'Router: custom workload-command support for the GCP lane (unblock credit spend
  for dispatch-script experiments)'
kind: infra
tags: []
created_at: '2026-06-11T05:43:25Z'
has_clean_result: false
---
# Router: custom workload-command support for the GCP lane (unblock credit spend for dispatch-script experiments)

## Problem

The compute router's GCP lane (and the SLURM lanes) can only execute the standard Hydra entrypoint: the GCP startup script hardcodes `uv run python scripts/train.py <hydra args>` and `RunSpec` (`src/explore_persona_space/backends/base.py`) carries `hydra_args` only — there is no field for a custom workload command. Most real experiments are custom multi-phase dispatch scripts (`scripts/issue<N>_dispatch.sh`), so they cannot run on the GCP lane at all.

Incident #571 (2026-06-11, ~04:09Z): auto routing (GCP-first default, `92cebe966`) sent `scripts/issue571_dispatch.sh` to GCP; the startup script ran bare `scripts/train.py`, crashed at startup, the EXIT trap powered the VM off, and the run was re-dispatched to RunPod. The stopgap workflow fix (`a573e1a6c`, /issue SKILL.md Step 6b "Lane capability check") now forces every custom-dispatch workload to `--backend runpod` — i.e. real money — while ~$100k of GCP credits in `eps-persona-gpu-jun2026` expire 2026-08-02. This field is the single change that moves the bulk of experiment spend onto the credits.

## Deliverables

1. **`RunSpec` gains a custom workload-command field** (`src/explore_persona_space/backends/base.py`), e.g. `workload_cmd: str = ""` — a repo-relative shell command (typically `bash scripts/issue<N>_dispatch.sh ...`). Mutually exclusive with `hydra_args`; validate fail-loud if both or neither are set.
2. **GCP lane executes it** (`src/explore_persona_space/backends/gcp.py`): when `workload_cmd` is set, the rendered startup script runs it (from the repo checkout, same env bootstrap/.env push/HF cache redirect as the standard path) instead of the hardcoded `scripts/train.py` line. Preserve all existing lifecycle machinery unchanged: `eps/phase` guest-attribute updates, the workload-done success signal, log paths under the workload dir, the fail-fast EXIT trap, and the no-secrets discipline (see #569 for the metadata→Secret Manager migration; do not regress whatever is current).
3. **SLURM lanes** (`src/explore_persona_space/backends/slurm.py`): same support where the stage structure allows (sbatch script wraps `workload_cmd`); if a lane genuinely cannot support it, it must fail loud with a typed capability error the router's fallback logic catches (the pattern `backends/gcp.py` already documents), never crash mid-provision.
4. **Dispatch threading** (`scripts/dispatch_issue.py`, `src/explore_persona_space/backends/issue_dispatch.py`, `router.py` as needed): `dispatch_issue.py launch` accepts the custom command and the auto chain routes custom workloads through GCP-first like any other run.
5. **Update the /issue Step 6b lane capability check** (`.claude/skills/issue/SKILL.md`, the block added by `a573e1a6c`): narrow it to whatever residual gaps remain after this lands (or remove it if none), so auto routing is valid for dispatch-script workloads again.
6. **Tests**: unit tests for the spec validation + per-lane rendering of `workload_cmd`; the existing router suite stays green, including `test_no_auto_runpod_path_under_any_failure` (this task must NOT touch the real-money invariant — RunPod stays override-only).
7. **Live acceptance smoke** (mirror #535's pattern): dispatch a trivial custom script (echo + `nvidia-smi` + the success signal) through the GCP lane on the smallest mapped machine type and verify the poller observes phases and completion, then teardown. Keep it tiny — this draws credits, not card money, but still verify the draw lands on the credit billing account.

## Acceptance criteria

- A custom dispatch-script workload launched via `dispatch_issue.py launch` with empty `backend:` frontmatter provisions on the GCP lane and runs the script end-to-end (proven by the live smoke).
- `hydra_args`-only dispatch behavior is byte-for-byte unchanged on every lane.
- Full test suite green; lint clean.
- /issue Step 6b capability-check text matches the new reality.

## Out of scope

- Adding RunPod to the auto chain (separate decision with its own user authority).
- 70B intents on GCP (`inf-70b`/`ft-70b` machine-type mapping stays fail-loud).
- The #569 secrets migration (composes, but lands separately).
