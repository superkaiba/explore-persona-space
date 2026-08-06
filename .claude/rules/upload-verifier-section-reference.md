---
paths:
  - ".claude/rules/upload-verifier-section-reference.md"
description: >
  Extended verification recipes for upload-verifier.md's Steps 2, 2.5, 2.6,
  2.9, 3, 4 and 6 — permanent-storage reconciliation, the phantom-URL HEAD
  gate, per-cell WandB coverage, git-destination reconciliation, the N/A
  justification rubric, the verdict decision table, and the on-FAIL
  procedure. Loaded ONLY via the explicit pointers in upload-verifier.md;
  the self-matching `paths:` glob keeps it out of every other agent context.
  The step headings, operative triggers and the hard PASS/FAIL gate contract
  stay in the spec.
---

# Upload-verifier section reference (Step 2/2.5/2.6/2.9/3/4/6 recipes)

Relocated verbatim from `.claude/agents/upload-verifier.md` (the spec is a
per-spawn system-prompt cost). Read ONLY the section the step under review
needs.

## Step 2 — Verify against permanent storage

For each candidate that should be uploaded, confirm it's actually
reachable:

```bash
# HF Hub model repo — pin the revision the body / sentinel cites; "main"
# only proves the LATEST snapshot has the path, not the SHA you cited.
uv run python -c "from huggingface_hub import HfApi; HfApi().list_repo_files('superkaiba1/explore-persona-space', revision='<sha>')" \
  | grep <expected-path>

# HF Hub data repo
# bare list_repo_files on the ~1M-file data repo times out (>90 s, #833) —
# scope the listing to the expected prefix (gotchas.md):
uv run python -c "from huggingface_hub import HfApi; print('\n'.join(e.path for e in HfApi().list_repo_tree('superkaiba1/explore-persona-space-data', path_in_repo='<expected-prefix>', repo_type='dataset', recursive=True)))" \
  | grep <expected-path>

# WandB
uv run python -c "import wandb; wandb.Api().run('<run-path>')"

# Git on the issue branch (named files only — Step 2.9 reconciles whole
# git-destination directories per-file)
git ls-tree -r <issue-branch> -- <path>
```

`scripts/verify_uploads.py` is one tool that does some of this
automatically, but it's opt-in on `--hf-dataset-path` and doesn't
auto-discover. **You must auto-discover.** The script is a helper for the
checks it already covers (model, WandB, git); for anything new the script
doesn't know about, use the HF / git / WandB commands above directly.
On a training task with no single `--hf-model` / `--wandb-run` to pass
(the multi-cell sweep case), the script's training rows self-resolve
from the task's `epm:results` reproducibility card (`reproducibility_card`
or its `reproducibility` alias), MERGED across all `epm:results` markers
newest-wins per declared field — an empty resume-pass re-post
(`adapter_paths: {}`, #601) does not shadow the first marker's full
declaration. Per-cell `adapter_paths` verified under `hf_model_repo` via
`list_repo_files`, `wandb_run_names` + `wandb_project` resolved by
display name (#608) — so do NOT pre-emptively supersede those rows by
hand.

## Step 2.5 — Phantom-URL gate

**Hard gate. New as of #456.** The `epm:results` marker AND the body's
`## Reproducibility` section name HF/WandB URLs the downstream consumer
(analyzer, follow-up experiment, mentor reader) will dereference. A URL
in a sentinel is a STRING — it is NOT evidence the underlying files
exist. **A claimed-but-absent URL is the phantom-checkpoint condition;
your verdict MUST be FAIL.**

Build a single text blob containing the `epm:results` marker body + the
clean-result body's Reproducibility section, then HEAD-check every
HF/WandB URL it contains at its CITED REVISION (not at `main`):

```bash
# 1. Concatenate the claimed-URL surfaces into one file. ALL epm:results
#    notes, not just the newest — multi-launch runs post several markers
#    and a resume re-post claims fewer URLs than the first (#601).
RESULTS_NOTES=$(uv run python scripts/task.py view <N> --json \
  | jq -r '.events[] | select(.kind=="epm:results") | .note')
BODY_PATH=$(uv run python scripts/task.py find <N>)/body.md
{ echo "$RESULTS_NOTES"; echo; sed -n '/^## Reproducibility/,$p' "$BODY_PATH"; } \
  > /tmp/issue-<N>-claimed-urls.txt

# 2. HEAD-verify every URL in the blob via verify_uploads.py.
#    Reuses orchestrate.hub.verify_artifacts_exist — the same helper
#    /issue Step 6a.5 runs PRE-LAUNCH to block on phantom carry-overs.
uv run python scripts/verify_uploads.py --issue <N> \
  --type <training|eval-only|generation|analysis> \
  --claimed-urls-file /tmp/issue-<N>-claimed-urls.txt \
  --json
```

**Always pass `--type` from the experiment type you received as an
input.** When omitted, the script infers it from the task's frontmatter
`kind` — which exempts `analysis/infra/batch/survey` tasks from the
training-only rows but conservatively assumes `training` for
`kind: experiment` (frontmatter cannot tell a training run from an
eval-only one). On an eval-only experiment that default demands
WandB-run + HF-model rows that cannot exist and produces a false
overall FAIL you then have to supersede row by row (incident #563,
2026-06-10). The script also scans the `issue-<N>` branch refs for
eval JSONs + figures, since those land on the issue branch before the
Step 9b auto-merge.

A multi-cell SWEEP training task likewise has no single `--hf-model` /
`--wandb-run` to pass — but do NOT hand-supersede the resulting MISSING
training rows: re-run `verify_uploads.py` and expect them to resolve
from the task's `epm:results` reproducibility card (`reproducibility_card`
or `reproducibility`), merged across ALL `epm:results` markers
newest-wins per declared field (per-cell `adapter_paths` under
`hf_model_repo`; `wandb_run_names` + `wandb_project` [+ optional
`wandb_entity`] resolved by display name — #608; a resume-pass marker
with an empty card never shadows an earlier full one — #601). Manual row
supersession remains legitimate ONLY when NO marker's card declares the
fields (`adapter_paths` / wandb fields absent across the whole history) —
then verify the per-cell artifacts yourself with the Step 2 commands
and record the superseding evidence in the verdict row.

The `claimed_urls` row in the JSON report is FAIL whenever any cited
URL did not resolve. Common phantom patterns to watch for:

- A `{phase}_step_checkpoints/checkpoint-<N>` subfolder cited in the
  sentinel, but the training code only uploaded the FINAL adapter and
  never uploaded the per-step trajectory dir. (Incident #456: the
  sentinel + body cited `i432_..._marker_implant_step_checkpoints/checkpoint-1600`
  at a specific commit; no code path uploaded that subfolder, the WandB
  run had zero logged artifacts, and a downstream experiment had to
  re-train the checkpoint two months later.)
- A merged-checkpoint URL that soft-failed an HF push (quota / 5xx) but
  the launcher swallowed the error and the local `rm` ran anyway.
- A `revision` field in the sentinel that points at a SHA where the
  cited subfolder was renamed / moved later.

If `claimed_urls` is FAIL, escalate to FAIL overall (Step 4) regardless
of which other rows passed. List every unresolved URL in the
`Auto-discovered files NOT covered by standard rows` table with
`Status: FAIL` and `Action: re-upload to <claimed URL> OR amend the
sentinel + body to cite the URL that actually has the files`.

## Step 2.6 — Per-cell WandB coverage

**New as of #527.** A sweep dispatcher that trains N cells in one
process can silently log every cell into a single WandB run — the
per-cell `wandb.init` effectively fires once and subsequent Trainer
runs write into / over the same run. Every other row still PASSes (the
eval JSONs landed, the adapters uploaded), but the per-cell loss /
log-prob trajectories for N−1 cells were never captured, and training
telemetry is UNRECOVERABLE after the fact — it can only be salvaged
while the pod is alive. (Incident #527: an 18-cell sweep produced
per-cell WandB telemetry for exactly 1 cell; the gap passed
upload-verification silently and 17 cells' trajectories were lost at
pod termination.)

If the task trained more than one cell — detectable from the plan's
cell count, per-cell `run_result.json` files, per-cell adapter
subfolders, or the per-cell eval-JSON enumeration from Step 1 —
reconcile WandB run coverage against the trained-cell list. Pull the
entity/project from the plan, the training config, or the
`epm:results` marker:

```bash
uv run python -c "
import wandb
for r in wandb.Api().runs('<entity>/<project>'):
    print(r.name, r.state, r.created_at)"
```

Apply this verdict rule:

- **One run per trained cell** (run names reconcile against the cell
  list), OR an explicit plan-recorded accounting that covers every cell
  (e.g. a deliberate grouped-logging design) → PASS. Record run count
  vs trained-cell count in the verdict table.
- **Fewer runs than cells, with no recorded accounting** → coverage
  gap. Before grading it, check the pod for salvageable telemetry:
  local offline run dirs under `wandb/` (recoverable via
  `wandb sync <dir>`) and `checkpoint-*/trainer_state.json` (its
  `log_history` carries the per-step loss trajectory).
  - **Salvageable telemetry exists on the pod** → **FAIL**, with the
    exact salvage commands (`wandb sync <dir>`; upload the
    `trainer_state.json` files to the HF data repo). This is precisely
    the class that must be caught while the pod is still alive.
  - **Nothing salvageable** → **WARN**, never silent: name every
    uncovered cell in the verdict table, state that its training
    telemetry is permanently unrecoverable, and instruct the analyzer
    to carry the gap into the clean-result's `## Reproducibility` as a
    caveat.

## Step 2.9 — Git-destination reconciliation

**New as of #537.** A directory-level `git add` silently drops
gitignore-excluded files while the commit still "succeeds" — so grading
a git-destination row off its NAMED / expected files alone passes round
1 and the gap surfaces a round late, or never. (Incident #537:
`.gitignore`'s `*.npz` excluded
`eval_results/issue_537/G_tensor/G_tensor.npz`, a plan-primary
deliverable, from a directory-level add; the git row PASSed round 1 on
the named eval JSONs and the drop was caught only by the round-2
Step 2.7 glob re-check.)

For EACH git-destination directory the run produced
(`eval_results/issue_<N>/`, `figures/issue_<N>/`, ...), reconcile the
source enumeration against the committed git tree — per FILE, not per
named artifact. Reuse the pod-side `find` listing from Step 1 (or a
local working-tree `find` if the artifacts were produced locally). On a
GCP `eps-issue-*` instance the source `find` needs `sudo` per Step 1's
note (a root-owned tree returns an empty source list, which would falsely
read as "everything committed"):

```bash
ssh_execute epm-issue-<N> 'cd /workspace/explore-persona-space && \
  find <dir> -type f 2>/dev/null | sort' > /tmp/issue-<N>-src-<slug>.txt
git ls-tree -r --name-only origin/issue-<N> -- <dir> | sort \
  > /tmp/issue-<N>-git-<slug>.txt
comm -23 /tmp/issue-<N>-src-<slug>.txt /tmp/issue-<N>-git-<slug>.txt
# any output = source files NOT in the committed tree
```

For each hit, run `git check-ignore -v <file>` to identify the likely
gitignore rule, then apply this verdict rule:

- **The file verifiably resolves at another permanent home** (e.g. an
  `.npz` / binary tensor on the HF data repo per the Upload Policy) →
  PASS for that file; record the verified URL in the same verdict row.
- **Otherwise** → **FAIL**, naming the file AND the matching gitignore
  rule (the `git check-ignore -v` output) in the verdict body, with the
  exact remediation (uploader runs `git add -f` with a one-line
  rationale, or uploads it to its correct destination).

A directory that is WHOLLY uncommitted under an existing deferred
grading (figures the analyzer commits at Step 9) follows the existing
figures DEFERRED rule — this check targets the silent PARTIAL drop,
where a commit landed but excluded files.

## Step 3 — Justify every N/A

If a standard row is reported N/A, you must say *why* — concretely, and
in a way that can be audited.

- ❌ Wrong: `| Raw completions | N/A | metrics-only eval pipeline |`
- ✅ Right (a stage that genuinely produced NO generations): `| Raw
  completions | N/A | Pod filesystem has no raw_completions.json anywhere
  under eval_results/issue_<N>/. Eval code at src/.../eval_panel.py:285
  computes an ARC-C logprob score with no model generation (no sampling
  call), so there are no completions to persist. |`

If your "N/A" is "the experimenter didn't generate this kind of
artifact", **you must have looked at the pod's filesystem to confirm
the absence.** "Probably not generated" is not a valid N/A.

**A stage that DISCARDS model generations is NEVER an acceptable N/A.**
Persist-by-default (CLAUDE.md § Upload Policy) makes rollout text the
load-bearing minimum: text/JSON rides the non-LFS path and uploads
unconditionally, so there is no size excuse. A generation-and-reduce
stage (persona-vector extraction, an online-scored eval that streams
completions into a running statistic) that dropped its generations is a
FAIL — whether the drop is UNDECLARED (blocker tag
`generation-discarded-undeclared`) or "DECLARED" via a
`discarded_artifacts:` entry naming generations / rollout text (an INVALID
declaration, blocker tag `generation-discard-declared-invalid`). The
`discarded_artifacts:` slot (planner §10, `{name, reason, regen_recipe}`)
licenses ONLY large intermediate TENSOR discards (activation stores,
per-context `v(x)`) — and only when the regenerating rollout TEXT is
persisted under `raw_completions/<stage>/`. Examples:

- ✅ Right (declared TENSOR discard, text persisted): `| Per-context v(x)
  activations | N/A (declared) | Extraction stream-reduces activations
  into r_B; plan §10 discarded_artifacts declares {name: extraction
  per-context v(x), reason: full-corpus activation grid exceeds HF/LFS
  headroom (#541), regen_recipe: one teacher-forced forward pass over the
  persisted raw_completions/extraction/ rollouts — no re-sampling}.
  Rollout TEXT persisted at raw_completions/extraction/. |`
- ❌ Wrong (undeclared generation discard): `| Raw completions | N/A | ...
  discards completions |` — no `discarded_artifacts:` entry ⇒ FAIL,
  blocker tag `generation-discarded-undeclared`.
- ❌ Wrong (INVALIDLY declared generation discard): `| Raw completions |
  N/A (declared) | discarded_artifacts declares {name: eval_completions,
  ...} |` — text is NEVER a valid discard entry ⇒ FAIL, blocker tag
  `generation-discard-declared-invalid`.

**Forward-only legacy guard.** The generation-discard FAIL fires only on a
run whose plan carries the `discarded_artifacts:` slot CAPABILITY — the
predicate is capability-presence, NOT a date: the plan contains a §10
output-artifact declaration block (the post-#817 planner template
section), regardless of when it was written. A run whose plan PREDATES the
slot (no such §10 block) AND has a generation-discard emits a single WARN
row `generation-discard-spec-absent` (mirroring Step 2.7's
`primary-deliverable-spec-absent` legacy WARN) — never a hard FAIL — and
that WARN fires ONLY when the legacy plan ALSO has a generation-discard,
never on every legacy body. New plans (which all carry the slot) get the
FAIL; the ~30 in-flight legacy plans ship.

## Step 4 — Decide verdict

**FAIL** if any of:
- A locally-generated file passes the "needed permanent URL" test in
  Step 1 but isn't reachable in Step 2.
- A training experiment has no model on HF Hub model repo.
- A training experiment has no live WandB run.
- Eval JSONs aren't committed to git on the issue branch.
- Pod was terminated despite filed follow-ups.
- Any check raises an unexplained error.
- Eval JSONs / figures that the body's `## Reproducibility` section
  CLAIMS are committed MUST be verified present
  (`git cat-file -e <sha>:<path>`) OR present on the HF data repo.
  On-pod-only ("local") is a FAIL, not a PASS — the pod is ephemeral.
  Cross-check every checkable Reproducibility claim (the named files
  exist at the named SHA; the pod-terminated marker matches the live
  RunPod API) before emitting PASS. Incident #397: a Step-8 PASS
  accepted on-pod-only 72-cell JSONs that were then deleted in
  disk-full cleanup, publishing an irreproducible clean-result whose
  body falsely claimed the files were committed.
- **Any HF / WandB URL claimed in the `epm:results` marker OR the
  body's `## Reproducibility` section does NOT actually resolve at
  the cited revision (Step 2.5 phantom-URL gate, `claimed_urls` row
  in the JSON report).** A URL string in a sentinel is not evidence —
  the files must list under that path at that revision via
  `huggingface_hub.list_repo_files` / `wandb.Api().run(...)`. Incident
  #456: a training run reached `awaiting_promotion` with `has_clean_result=true`
  whose body cited a per-step checkpoint subfolder at a pinned revision
  that did not exist anywhere on HF; no code path had ever uploaded
  the per-step trajectory dir, the WandB run had zero artifacts, and
  upload-verification PASSed because it trusted the sentinel's string
  without HEAD-checking it. A downstream experiment had to re-train
  the checkpoint two months later.
- **A multi-cell / sweep task has fewer WandB runs than trained cells
  AND salvageable telemetry still exists on the pod (Step 2.6 per-cell
  coverage check — local `wandb/` offline dirs or
  `checkpoint-*/trainer_state.json`).** Terminating the pod here
  destroys the only copy of the missing cells' training trajectories;
  the remediation is cheap while the pod is alive (`wandb sync` /
  upload the trainer states). Incident #527: the per-cell `wandb.init`
  fired for 1 of 18 cells, the verifier passed silently, and 17 cells'
  loss / log-prob trajectories were permanently lost at termination.
- **Any row in the plan's `primary_deliverable:` block enumerates zero
  files on the pod (Step 2.7 primary-deliverable gate, blocker tag
  `primary-deliverable-missing`).** The headline phase that produces
  the Goal's primary dependent variable silently did not run —
  terminating the pod here destroys the cheap-fix window. SKILL.md
  Step 8 reads this blocker tag and refuses to call `pod.py terminate`;
  the /issue skill then AUTO-RECOVERS by flipping status back to
  `running` and re-dispatching the experimenter to re-drive the missing
  phase on the still-alive pod (it does NOT park-and-wait — only the
  generic `workflow.yaml § pivot_criteria` cap-3 path routes to
  `status:blocked` for this failure class). Incident #519: an experiment
  shipped a clean-result even though the headline activation-shift /
  SVD / steering phases were silently skipped at launch (dispatcher's
  `if args.X and args.Y` guard fell through on missing input JSONs,
  manifest recorded `skipped_phases: []`), pod was terminated, per-step
  checkpoints lost.
- **An artifact the plan's analysis / negative-control sections name as a
  downstream input exists on the pod but has no permanent URL (Step 2.8,
  #521).** Terminating the pod makes the plan's remaining controls
  permanently unrunnable; the remediation is cheap while the pod is alive
  (the files are KB-MB — upload to the HF data repo
  `issueN_<slug>/analysis_tensors/`).
- **A file under a git-destination directory exists at the source but is
  absent from the committed git tree AND has no other verified permanent
  home (Step 2.9 git-destination reconciliation, #537).** A `.gitignore`
  rule silently drops files from a directory-level `git add` while the
  commit succeeds; grading the git row off named files only defers the
  catch to a later round — or past pod termination.
- **A stage that produced model generations and dropped them (Step 3
  generation-discard gate, #779).** Persist-by-default: rollout text is the
  regenerating minimum; a generation-discard is silent data loss, the exact
  #779 / #365 class — and it FAILs whether UNDECLARED (blocker tag
  `generation-discarded-undeclared`) or INVALIDLY DECLARED via a
  `discarded_artifacts:` entry naming generations / rollout text (blocker
  tag `generation-discard-declared-invalid`; the slot licenses only large
  intermediate-TENSOR discards). A large-TENSOR discard with a `{name,
  reason, regen_recipe}` entry AND its regenerating rollout text persisted
  under `raw_completions/<stage>/` PASSes. Forward-only legacy guard: a plan
  PREDATING the §10 `discarded_artifacts:` slot capability emits WARN
  `generation-discard-spec-absent` instead of FAIL, and only when it also
  has a generation-discard (Step 3).

**WARN** is acceptable for:
- Pod stopped (can't verify cleanup post-hoc — note this and move on).
- Figures not yet committed (analyzer will commit them in Step 9).
- Per-cell WandB coverage gap where nothing salvageable remains on the
  pod (Step 2.6) — report it loudly, never silently: name every
  uncovered cell, flag the telemetry loss as permanent, and instruct
  the analyzer to carry it into the clean-result's `## Reproducibility`.
- A generation-discard on a run whose plan PREDATES the §10
  `discarded_artifacts:` slot capability (Step 3
  `generation-discard-spec-absent`) — legacy plans ship, never a hard
  FAIL; report it loudly so the analyzer notes the loss.

**PASS** only when every discovered file is accounted for.

## Step 6 — On FAIL, do NOT advance

Stay at `status:verifying` (there is no `uploading` status — task.py rejects it). List the remediation commands. The next
caller (uploader agent or experimenter) fixes the gaps; you re-verify.
