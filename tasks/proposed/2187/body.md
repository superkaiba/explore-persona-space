---
title: 'Make pre-teardown verification sweep the out-root TOP LEVEL: three artifact
  losses on one task, each caught only by a manual name-set diff'
kind: infra
tags:
- outroot-glob-coverage
created_at: '2026-08-08T00:38:52Z'
has_clean_result: false
origin_prompt: 'Filed by the #2162 orchestrator after the third out-root top-level
  artifact in one run turned out to be uploaded by no glob: pilot_gate_report.json
  at P5, stage2_results.json at P8, and upload_done.json on the margin leg (found
  only via a recursive pod-vs-HF name diff, 236 vs 235 — a count-only check would
  have looked clean).'
workflow: v1
---
# Per-issue upload phases repeatedly miss out-root TOP-LEVEL artifacts: three losses on one task, each caught only by a manual pre-teardown sweep

## Goal

Close a demonstrated, repeating durability gap: per-issue `phase_upload`
implementations glob their SUBDIRECTORIES and therefore silently omit files
written at the out-root TOP LEVEL. Each such file is absent from HF, absent
from git, and not a declared discard — so it dies at pod teardown, and the
only thing standing between it and permanent loss is an agent remembering to
run a manual sweep.

Task #2162 hit this **three separate times in one run**. That is a class, not
three coincidences.

## The three incidents (all #2162, all confirmed)

1. **P5 — `pilot_gate_report.json` (740 B).** Sat at `/workspace/issue2162_out/`
   top level; `phase_upload` globs only the six subdirs. Caught by the
   `upload-verifier`, which FAILed the run on exactly this. It recorded the
   MEASURED per-rollout generation cost (1.2019 s/rollout at gen_batch=16 over
   8 workers) that the poll fence and ETA were derived from — a successor would
   have had to re-run the pilot to recover it.
2. **P8 — `stage2_results.json` (754 B).** Same shape: out-root top level,
   while `phase_upload` globs only `shard_*.jsonl` inside `stage2/`. Carried the
   stage-2 reproducibility block plus two scope notes, one of which states that
   the plan's `phase_outputs.P8` V_a-shard entry is DELIBERATELY not produced —
   without it, a later upload-verify reconciling P8 against the plan would FAIL
   hunting an artifact that was never meant to exist.
3. **Margin leg — `upload_done.json` (2,580 B).** The manifests glob is
   `['*.json', 'blocks/*.done.json', 'margin_blocks/*.done.json']` and
   `upload_done.json` is written AFTER that upload completes, so the
   upload-completion marker structurally cannot be inside its own upload
   (236 files on the pod vs 235 uploaded). It carried the full
   `reproducibility_card` (seed_base, bank_seed, max_new_tokens, temperatures,
   draw counts, gen_batch, library versions) AND **three plan deviations
   recorded nowhere else**, all of which the report Methodology needs.

Incident 3 is the sharpest: the loss was found only because a recursive
pod-vs-HF name diff was run by hand before teardown. A count-only check would
have read 235 uploaded and looked fine.

## Why the existing safety nets did not cover it

- The `upload-verifier` DID catch incident 1 — so the workflow surface works
  when it runs. But it runs at Step 8 / P5-style checkpoints, not before every
  suffixed-pod or deferred-leg teardown, which is where incidents 2 and 3 lived.
- The completion-side teardown contract says "verify THIS round's artifacts,
  then terminate", but "verify" is not currently specified to include an
  out-root top-level sweep, so a conscientious agent can verify the declared
  prefixes 100% and still lose a file.
- Producer self-reports are useless here by construction: each upload logged
  "Bulk upload verified: N files" and each N was CORRECT for what it globbed.
  The omission is in the glob, not the upload.

## Candidate work (the implementing session decides the shape)

- **Primary — make the sweep a specified part of pre-teardown verification.**
  Amend the completion-side teardown recipe (`.claude/rules/pods.md` and the
  CLAUDE.md "Completion-side teardown" clause) so verify-then-terminate
  explicitly includes: enumerate every file the run produced under the out-root
  (recursively), diff NAME SETS against the union of uploaded prefixes, and
  either upload or git-commit any residue before terminating. Name the
  count-only trap: a matching count is not a matching set.
- **Consider a mechanical check** in `upload-verifier` (and/or a helper the
  per-issue upload code can call) that fails loud when any regular file under
  the out-root matches NO upload glob and is not in `discarded_artifacts`. This
  is the version that actually prevents a fourth occurrence, because it does not
  depend on agent diligence.
- **Consider the chicken-and-egg case explicitly.** An upload-completion marker
  written after its own upload can never be inside it. Either write such markers
  BEFORE the final upload, or give them a dedicated second tiny upload, or
  route them to git. Whichever is chosen, state it so per-issue authors stop
  reinventing it.
- **Consider `.claude/rules/upload-policy.md` guidance** for per-issue
  `phase_upload` authors: glob the out-root top level, not only subdirs.

## Scope notes

- The offending globs live in per-issue experiment code
  (`scripts/issue2162_run.py`, `scripts/issue2162_stage2.py`), which is NOT
  workflow surface and should not be retro-edited for a completed run. The
  durable fix belongs in the RULES and the shared verifier, so the next
  experiment's author inherits it.
- Do NOT weaken the existing fail-loud posture: a genuine discard still
  requires the declared `discarded_artifacts` entry with a regen recipe, and
  text/JSON is never discardable.
- All three #2162 losses were recovered before teardown, so nothing is
  currently missing — this task is about preventing the fourth, not repairing
  the first three. The recovered files are committed at `9fddd9e1a6`,
  `6eec51c77e`, and `92f25415ee` respectively if the implementer wants to see
  the payload shapes.
