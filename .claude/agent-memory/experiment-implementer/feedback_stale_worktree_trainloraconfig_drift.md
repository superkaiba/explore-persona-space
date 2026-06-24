---
name: Stale worktree → TrainLoraConfig API drift crashes at first train cell
description: A long-lived issue worktree behind origin/main has an older src/.../train/sft.py; a recipe passing newer kwargs (max_steps, lr_scheduler_type) via **extra crashes TrainLoraConfig.__init__; run the per-kwarg signature smoke before posting.
type: feedback
---

When an experiment recipe threads training kwargs into `train_lora` via a
`**extra` dict (e.g. `extra["max_steps"]=200`, `extra["lr_scheduler_type"]=
"linear"` for a #519-style EM recipe), those kwarg names MUST exist as
`TrainLoraConfig` dataclass fields — otherwise `TrainLoraConfig.__init__()`
raises `unexpected keyword argument` at the FIRST training cell on the pod, after
the GPU is already spent.

**Why:** a long-lived `issue-<N>` worktree can be hundreds-to-1000+ commits
behind `origin/main`, and `src/explore_persona_space/train/sft.py` drifts. `main`
may ALREADY have the field you need (e.g. as of 2026-06 `main`'s `TrainLoraConfig`
has `max_steps: int|None=None`, `lr_scheduler_type: str|None=None`, `optim`,
`warmup_steps`, `kl_aux_*`), while the worktree's stale copy does not and the
SFTConfig assembly hardcodes `lr_scheduler_type="cosine"`. The crash is invisible
to CPU smokes that stub `train_lora`. (Incident #653 round 2, 2026-06-24: the EM
recipe would have crashed every EM cell; v11's code-review even cited override
line numbers that don't exist in the stale worktree.)

**How to apply:**
- Before posting the implementation marker, run the per-kwarg signature smoke the
  agent spec mandates: assert `{f.name for f in dataclasses.fields(TrainLoraConfig)}`
  is a superset of every kwarg the dispatcher's call site + its `**extra` dict
  pass. This is cheap and catches the drift pre-launch.
- When a field IS missing, reconcile against `origin/main` — port the field block
  + its SFTConfig wiring (`if cfg.X is not None: sft_kwargs["X"]=cfg.X`) + any
  associated hook (e.g. `_maybe_attach_kl_aux`) VERBATIM from `main`, so the
  worktree's dataclass matches `main` (no future merge conflict, and the
  `*_invariant` tests that pin the full field set — e.g.
  `test_issue545_train_components.py::test_registry_overrides_match_train_lora_config`
  — pass again). Do NOT invent your own default (`main`'s `lr_scheduler_type`
  default is `None`, not `"cosine"`; the base SFTConfig dict keeps `"cosine"` and
  overrides only when set).
- Do NOT merge all of `origin/main` into a 1000-commit-behind worktree to fix one
  field — that's a conflict-prone infra operation; scope the port to the drifted
  fields. (A separate `type:infra` worktree-refresh task / a pre-launch
  `origin/main` sync owns the full merge.)
- Running the project's `*_invariant` train-component tests
  (`uv run pytest tests/test_issue545_train_components.py`) surfaces this drift:
  if they PASS on repo-root `main` but FAIL in the worktree, the worktree's
  `sft.py` is behind — a reliable drift detector.
