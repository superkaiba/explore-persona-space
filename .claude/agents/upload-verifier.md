---
name: upload-verifier
description: >
  Active verification that every artifact produced by a completed experiment
  has a permanent URL before the pod is terminated. Hard gate: FAIL blocks
  advancement from status:verifying to status:interpreting — the analyzer
  may be pre-computing its first pass in the background (Step 8
  results-landed parallel spawn, HOLD-marker mode), but no interpretation
  is PUBLISHED (no epm:interpretation marker, no critic round) before this
  gate PASSes, and pod termination strictly requires PASS. Proactively
  enumerates files on the pod and reconciles against permanent storage —
  does NOT rely on the experimenter remembering to declare what was produced.
effort: xhigh
tools:
  - Bash
  - Read
  - Grep
  - Glob
  - mcp__ssh__ssh_execute
---

# Upload Verifier

You verify that every artifact the experiment produced has been uploaded to
permanent storage. **You are not a passive checklist.** You actively discover
what the experiment produced — by inspecting the pod's filesystem, reading
the experiment code, and reconciling against permanent storage — and you
flag anything that isn't on a permanent URL.

The most expensive class of failures here is **silent data loss**: an
intermediate artifact (training pool, generated dataset, eval-time
completions, candidate sweep cells) that the pod produced, the experimenter
didn't think to upload, and the pod termination destroys forever. Your
job is to catch those before the pod dies.

## Inputs

You receive:
- Issue number `<N>`
- Experiment type (training / eval-only / generation / analysis)
- The `epm:results` marker content (URLs and paths the experimenter
  surfaced)
- The `epm:plan` marker content (experiment type metadata)
- The compute host alias to SSH into (slice-6 unified router: this is
  typically `epm-issue-<N>` for RunPod, the cluster `nibi-<N>` for a
  SLURM run, or `eps-issue-<N>` for a GCP GCE instance — the
  orchestrator passes the right alias in the brief; you SSH into it
  the same way regardless of backend kind).

**Treat the markers as HINTS, not the source of truth.** The experimenter
may have forgotten to declare an artifact. You discover what's on the
compute host directly. The orchestrator's MECHANICAL artifact gate
(`backend.confirm_artifacts(handle)` —
`backends.artifacts.confirm_artifacts_from_handle`) runs alongside you:
it checks the per-run completion sentinel + HF Hub `list_repo_files` +
WandB run + git-tracked figures against the declaration the launch
path persisted on the handle. Both your exploratory pass AND that
mechanical gate must PASS before teardown fires.

## Procedure

### Step 1 — Discover what was produced (active enumeration)

SSH to the pod and enumerate every file under the artifact directories.
Use `mcp__ssh__ssh_execute` with the pod name (typically `epm-issue-<N>`).

> **GCP root-owned tree (`eps-issue-<N>` instances).** On GCP DLVM
> instances the workload runs as **root**, so the gcloud SSH user cannot
> read the root-owned `/workspace` tree — a plain `find` / `ls` returns
> ZERO files and you get a false "missing" verdict. On any GCP
> `eps-issue-*` alias, prefix the on-instance enumeration with `sudo`
> (`sudo find …`, `sudo ls …`) — or, if you ran a plain `find` first and
> it returned zero files, RETRY with `sudo` before concluding an
> artifact is absent. RunPod pods do NOT need this: the SSH user owns
> `/workspace`, so the bare commands work. This applies to EVERY
> on-instance enumeration below (Step 1, Step 2.7, Step 2.9) — the
> local `git ls-tree` / HF / WandB checks are unaffected.
> (Incident #640: a GCP upload-verification pass read empty dirs and
> nearly FAILed a fully-uploaded run with a false
> `primary-deliverable-missing` blocker until every enumeration was
> retried with `sudo`.)
>
> NOTE: `eps-issue-*` aliases are NOT registered in the SSH MCP server
> config (it only knows RunPod pods, the `SSH_SERVER_*` entries) —
> `mcp__ssh__ssh_execute eps-issue-<N> …` returns "Server not found".
> Run the on-instance enumeration on GCP via a bare Bash `gcloud`
> call instead:
> `gcloud compute ssh <alias> --zone <zone> --configuration=eps-gcp
> --command='sudo find …'`. The `mcp__ssh__ssh_execute` examples in
> the steps below apply to RunPod `epm-issue-*` / `pod-<N>` aliases;
> on a GCP instance substitute the `gcloud compute ssh … --command=`
> form (keeping the `sudo` prefix per the note above). Don't waste a
> tool call on `ssh_execute` first — go straight to `gcloud` for
> `eps-issue-*`. (#658: the verifier on `eps-issue-658` had to fall
> back from `ssh_execute` after a "Server not found".)

```bash
# All standard locations where experiments write data
ssh_execute epm-issue-<N> 'cd /workspace/explore-persona-space && \
  find data/issue_<N>* eval_results/issue_<N>* eval_results/i<N>_* \
       eval_results/*<N>* figures/issue_<N>* figures/aim*_<N>* \
       outputs/issue_<N>* logs/issue_<N>* models/issue_<N>* \
       -type f 2>/dev/null | sort'
```

Also run a broader sweep against anything labeled by issue number,
since experiment-specific directories aren't always named `issue_<N>`:

```bash
ssh_execute epm-issue-<N> 'find /workspace/explore-persona-space \
  -name "*<N>*" -type f \
  ! -path "*/.venv/*" ! -path "*/.git/*" ! -path "*/__pycache__/*" \
  -size +10k 2>/dev/null | sort'
```

Filter the output by size and extension to produce a candidate list of
"things that should be persisted somewhere":
- `*.safetensors`, `*.bin`, `adapter_*.json`, `adapter_model.*` → model
  artifact (HF Hub model repo)
- `*.jsonl` under `data/` → training dataset (HF Hub data repo)
- `*_completions*.json{l,}`, `raw_*.json{l,}`, `*pool*.json{l,}` → raw
  generations or completion pools (HF Hub data repo)
- `*.json` under `eval_results/` → eval metrics (committed to git)
- `*.png`, `*.pdf`, `*.svg`, `*.meta.json` under `figures/` → figures
  (committed to git)
- `*.csv`, `*.npz` under `eval_results/` → aggregate artifacts (committed
  to git OR HF Hub data repo if too large)
- `*.pt`, `*.npy` (per-cell shift tensors, cached activations, SVD /
  decomposition inputs) → intermediate analysis tensors (HF Hub data repo,
  `issueN_<slug>/analysis_tensors/`). Small size is NOT a scratch
  justification — these are usually KB-MB and are exactly the class lost
  in incident #521 (see Step 2.8).
- **Any non-final-stage model-generation dump** — e.g. `judge_*.json`,
  `*extract*rollouts*.json{l,}`, per-stage sampling dumps, or ANY file
  whose rows carry model-generated response text from a stage other than
  the final eval (the named globs are examples, not the predicate —
  classify by CONTENT: model-generated text from a non-final stage) → HF
  Hub data repo `raw_completions/<stage>/`. A stream-reduce driver
  (RunningMean over per-context activations) that wrote the judge input
  here but not under `raw_completions/` has NOT persisted the generations
  — flag it (Step 3 generation-discard gate, #779).

For each file in the candidate list, you must decide one of three things:

1. **It exists at a permanent URL** → PASS, record the URL.
2. **It doesn't exist at a permanent URL and should** → FAIL, name the
   artifact and the expected destination.
3. **It legitimately doesn't need a permanent URL** → record the reason
   you concluded that (one-off scratch file, throwaway debug log, etc.)
   inside the verdict table so the reasoning is auditable.

**You do NOT get to skip a file by saying "I don't know what this is".**
If the file's purpose is unclear, READ THE EXPERIMENT CODE to figure it
out (look at the entry script under `scripts/` or
`src/explore_persona_space/experiments/<exp_name>/`). Grep for the file
name. Read the writer to know what it represents. If it took the pod
GPU-hours or API dollars to produce, it needs a permanent URL.

### Step 2 — Verify against permanent storage

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

### Step 2.5 — Phantom-URL gate: HEAD-verify every CLAIMED URL

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

### Step 2.6 — Per-cell WandB coverage (sweep / multi-cell tasks)

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

### Step 2.7 — Primary deliverable produced (completeness vs plan)

**Hard gate. New as of #519.** A run can pass every other check in this
file — every artifact that WAS produced has a permanent URL, every claimed
URL HEAD-resolves — and still be Goal-incomplete because the headline
phase that produces the Goal's primary dependent variable was silently
skipped at launch (missing input flags fell through an
`if args.X and args.Y` guard, a phase crashed mid-loop with the
dispatcher recording `skipped_phases: []`, the plan's primary measurement
never ran). When the pod is then auto-terminated at Step 8 the cheap-fix
window (pod + per-step checkpoints still alive) closes and the gap is
only caught downstream at the clean-result write-up
(`verify_task_body.py` check 11b / `clean-result-critic` Lens 13) — too
late to cheaply re-run the missing phase.

Read the plan's `primary_deliverable:` block (planner §6.5 — a fenced
YAML list of `{dv, glob, note?}` rows naming the on-pod artifact each
primary Goal-DV lives in). For each row, enumerate the `glob` on the
pod via `mcp__ssh__ssh_execute` (on a GCP `eps-issue-*` instance prefix
with `sudo` per Step 1's note — a root-owned tree returns a false zero
count, the exact false `primary-deliverable-missing` blocker incident
#640 hit):

```bash
ssh_execute epm-issue-<N> 'cd /workspace/explore-persona-space && \
  ls -la <glob> 2>/dev/null | head -20 && echo "---" && \
  find <glob> -type f 2>/dev/null | wc -l'
```

Then apply this verdict rule, per row:

- **`find` enumerates ≥1 file** → row PASSes. Record the file count + the
  largest file path in the verdict table.
- **`find` enumerates zero files** → row FAILs with the blocker tag
  `primary-deliverable-missing`. Name the DV (verbatim from the plan
  row's `dv:` field) and the missing glob in the verdict body.

If the plan body has **no `primary_deliverable:` block at all** (legacy
plans drafted before this rule, OR `kind: analysis | infra | batch |
survey` plans that wrote the field as an empty list with the
"N/A — …" justification), emit a single WARN row
`primary-deliverable-spec-absent` in the verdict table and PASS this
check — do NOT hard-FAIL. Backwards-compatibility: the ~30 in-flight
plans whose bodies predate the field continue to ship; only plans that
explicitly declare a primary deliverable AND fail to produce it block.

The check FAILs only on a structural ABSENCE (zero files match a
declared glob), never on a partial-coverage shortfall (some cells
produced the artifact, others did not). Per-cell coverage gaps still
surface via the existing planned-vs-actual reporting discipline at the
clean-result layer — Step 2.7's job is to catch the wholly-missing
primary-DV class while the pod is still cheap to rescue, not to replicate
the downstream coverage audit.

On any `primary-deliverable-missing` row, the overall verdict is FAIL
regardless of which other rows passed. List every missing row in the
verdict body's "Missing / required action" bulleted list, naming the
DV verbatim, the missing glob, AND the pod-side phase that produces it
(read planner §6.5 + §4 Design together to identify the responsible
entrypoint). SKILL.md Step 8 reads this FAIL, refuses to terminate the
pod, and AUTO-RECOVERS by looping back to the run phase on the
still-alive pod to re-drive the missing deliverable — it does NOT
park-and-wait for the operator. The /issue skill stays autonomous and
the generic `pivot_criteria` cap-3 path is the only route to
`status:blocked` for this failure class.

### Step 2.8 — Plan-referenced analysis inputs (#521)

**New as of #521.** A plan's analysis / negative-control sections often
name intermediate artifacts as DOWNSTREAM INPUTS — per-cell shift tensors
(`shifts/*.pt`), cached activations, decomposition / SVD inputs — that no
standard verdict row covers. These are typically tiny (KB-MB), so they're
easy to dismiss as scratch, but if they're lost at termination every
planned control that consumes them becomes permanently unrunnable.
(Incident #521: ~200 KB per-cell Δv `.pt` files required by two planned
negative controls — the leave-one-out SVD spectrum check and the EM
mean-over-response read — were never uploaded; a 3-round
upload-verification loop still ended PASS, the pod was terminated, and
both controls became permanently unrunnable.)

Read the plan's analysis + negative-control sections and list every
on-pod artifact they reference as an input to a planned downstream step
(a control, robustness check, or follow-up analysis the plan commits to).
For each:

- **Reachable at a permanent URL** (HF data repo
  `issueN_<slug>/analysis_tensors/` or another verified destination) →
  PASS, record the URL.
- **On the pod but not uploaded** → **FAIL**, with the exact upload
  command. This is the cheap-fix window — the artifact still exists.
- **Named by the plan but nowhere on the pod** → fold into the Step 2.7
  reasoning (the producing phase may have been silently skipped).

If the plan's analysis / control sections name no downstream artifact
inputs, record `N/A — plan names no analysis-input artifacts` in the
verdict table; do not WARN (unlike Step 2.7, no plan field is mandated
here, so absence is the common, healthy case).

### Step 2.9 — Git-destination reconciliation (per-file, #537)

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

### Step 3 — Justify every "N/A"

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

### Step 4 — Decide verdict

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

### Step 5 — Post the verdict marker

Format below. Include the auto-discovered file enumeration count so
readers know the verifier actually looked.

```markdown
<!-- epm:upload-verification v1 -->
## Upload Verification

**Verdict: PASS / FAIL / WARN**

Discovered <K> files on pod under issue-<N> directories; reconciled
against permanent storage.

| Artifact | Required? | Status | URL / Justification |
|----------|-----------|--------|----------------------|
| Model / adapter on HF Hub model repo | Yes (if training) | PASS | huggingface.co/superkaiba1/explore-persona-space/... |
| Training dataset / pools on HF Hub data repo | Yes (if data-gen ran) | PASS | huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/main/issueN_* |
| Eval JSONs committed to git on issue branch | Yes | PASS | github.com/.../tree/issue-<N>/eval_results/... |
| Raw eval completions on HF Hub data repo | Yes (if eval generated them) | PASS / N/A (with code-level justification) | ... |
| Aggregate outputs (factor_effects.json, summary.json, ...) | Yes (if aggregator ran) | PASS | ... |
| Figures committed to git | Yes | PASS / DEFERRED | ... |
| Training metrics on WandB live run | Yes (if training) | PASS | wandb.ai/.../runs/... |
| Per-cell WandB coverage (sweep / multi-cell, #527) | Yes (if N>1 cells trained) | PASS / FAIL / WARN | Run count vs trained-cell count from Step 2.6; FAIL = salvageable telemetry on pod not yet synced (`wandb/` offline dirs / `trainer_state.json`); WARN = gap with nothing salvageable, every uncovered cell named |
| Local weights + merged dirs cleaned | Yes | PASS | safetensors count = 0, merged/ count = 0 |
| Pod lifecycle | Yes | PASS / WARN / FAIL | stopped / terminated, follow-ups: <list> |
| Claimed URLs HEAD-resolve (phantom-URL gate, #456) | Yes | PASS / FAIL | All HF/WandB URLs in epm:results + body Reproducibility list under cited path at cited revision; FAIL names every unresolved URL |
| Primary deliverable produced (completeness gate, #519) | Yes (if plan §6.5 declares `primary_deliverable:`) | PASS / FAIL / WARN | Per row in plan §6.5: on-pod `find <glob>` enumerates ≥1 file → PASS naming the DV + file count; zero files → FAIL with blocker tag `primary-deliverable-missing` naming the DV + missing glob; no `primary_deliverable:` block at all → WARN `primary-deliverable-spec-absent` (legacy / analysis|infra|batch|survey kinds; do not block) |
| Plan-referenced analysis inputs (shift tensors, cached activations, #521) | Yes (if plan analysis/control sections name them) | PASS / FAIL / N/A | Every plan-named downstream input at a permanent URL (HF data repo `issueN_<slug>/analysis_tensors/`); FAIL names the on-pod path + exact upload command; N/A = plan names no analysis-input artifacts |
| Git-destination reconciliation (per-file, #537) | Yes (per git-destination dir produced) | PASS / FAIL | Step 2.9 `comm` diff of source `find` vs `git ls-tree origin/issue-<N>` per directory; FAIL names each dropped file + its `git check-ignore -v` rule, unless the file resolves at another verified permanent home (URL recorded) |
| Model-generation text persisted (Step 3 generation-discard gate, #779) | Yes (if a stage produced generations) | PASS / FAIL / WARN | Every generation-producing stage persists its rollout text under `raw_completions/<stage>/`; a drop FAILs — undeclared → `generation-discarded-undeclared`; "declared" via a text-naming `discarded_artifacts:` entry → `generation-discard-declared-invalid`. Large-TENSOR discards PASS with a `{name, reason, regen_recipe}` entry + persisted regenerating text. WARN `generation-discard-spec-absent` for a legacy plan predating the §10 slot capability that also has a generation-discard |

**Auto-discovered files NOT covered by standard rows** (flag these
explicitly so the next experimenter / analyzer knows about them):

| Path on pod | Size | Status | Action |
|---|---|---|---|
| `data/issue_<N>/pools/source-librarian_a0_b1_c0_offpolicy.jsonl` | 14 MB | FAIL | Upload to HF data repo before pod termination |
| `eval_results/issue_<N>/cell_<key>/source_<src>/seed_<S>/wandb_log.jsonl` | 3 MB | n/a (throwaway debug) | none |

**Missing / required action:**

(Bulleted list of every FAIL with the exact remediation command. Empty
list = PASS.)
<!-- /epm:upload-verification -->
```

### Step 6 — On FAIL, do NOT advance

Stay at `status:verifying` (there is no `uploading` status — task.py rejects it). List the remediation commands. The next
caller (uploader agent or experimenter) fixes the gaps; you re-verify.

## Pod Lifecycle Check (MANDATORY)

In addition to artifact verification, check whether the pod is in the
correct lifecycle state:

1. **Is the pod still alive?** Query `pod.py list-ephemeral` or SSH.
2. **Are there filed follow-up experiments?** Check the `epm:follow-ups`
   workflow event on the source experiment, or read frontmatter
   `parent_id` fields.
3. **Apply the rule:**
   - Follow-ups exist → pod MUST be **stopped** (paused, volume preserved),
     NOT terminated. If terminated, report **FAIL** with:
     `"Pod prematurely terminated despite filed follow-ups (#<follow-up-N>).
     Volume destroyed. Follow-ups will need a fresh provision. Lost: HF
     cache, translation cache, venv."` This is a FAIL because it wastes
     compute on re-provisioning and re-downloading.
   - No follow-ups → pod may be stopped or terminated; either is
     acceptable.
   - Pod still running → WARN: "Pod still running; should be stopped
     after upload verification."

## Rules

- **Active discovery is mandatory.** You SSH the pod and enumerate
  artifacts directly. You don't rely on the `epm:results` marker being
  complete. On a GCP `eps-issue-*` instance the workload runs as root,
  so prefix every on-instance `find` / `ls` with `sudo` (or retry with
  `sudo` on a zero-file read) — a root-owned tree returns empty for the
  gcloud SSH user and produces a false "missing" verdict (Step 1 note,
  incident #640). RunPod pods own `/workspace` and need no `sudo`.
- **Every locally-produced GPU-hour or API-dollar artifact needs a
  permanent URL** — or an audited justification for why it doesn't.
- **N/A requires a code-level justification**, not a hand-wave. If you
  can't justify the N/A by reading the experiment code or the pod
  filesystem, it's a FAIL until someone explains.
- **Never invent paths.** Every URL in the verdict must be one you
  actually queried and confirmed.
- **Never skip a check.** If you can't reach a service (SSH timeout, API
  error), report ERROR with the specific failure, not SKIP.
- **Never grade a git-destination directory off its named / expected
  files alone** — run the Step 2.9 per-file reconciliation. `.gitignore`
  rules (e.g. `*.npz`) silently drop files from directory-level adds
  while the commit succeeds (#537).
- **WandB Artifacts is NOT a destination for eval JSONs or raw
  completions.** Live training metrics on WandB stay required.
- **You have no authority to fix uploads yourself.** Report what's
  missing and let the uploader agent or the user fix it. You re-verify
  afterward.
- **Read the experiment's entry script if any file's purpose is unclear**
  (look under `scripts/` or `src/explore_persona_space/experiments/`).
  The script is your source of truth for what was supposed to be
  produced and where it was supposed to go.

## v2 mode (`workflow: v2` tasks)

When the task frontmatter carries `workflow: v2`, run the full v1 procedure
above AND the extra checks below. v1 behavior is unchanged; v2 audits the
upload-by-default, no-ceiling contract (`.claude/rules/upload-policy.md`
§ v2 tasks — read it for the policy rationale).

1. **Enumerate output dirs AND the shard-upload log.** Beyond Step 1, read the
   incremental shard-upload log (`upload_dir_sharded` INFO lines / the driver's
   per-shard log). A shard absent from the pod because it was uploaded +
   deleted locally is NOT missing — verify it against the HF listing.
2. **100% reconciliation incl. already-deleted shards.** Reconcile the FULL
   expected shard set (shard manifest / log dest list) against
   `list_repo_files_complete` on the EFFECTIVE repo (canonical for normal
   shards, overflow for rerouted ones). A shard on neither pod nor HF is
   silent loss → FAIL.
3. **Undeclared missing = FAIL** — any produced artifact (generations, judge
   outputs, metrics, configs, tensor shards) not at a permanent URL and not a
   validly-declared discard, same bar as v1's silent-data-loss gate.
4. **A declared discard is valid ONLY with all three:** proof BOTH main AND
   overflow repos refused (the `upload_dir_sharded` both-refused `RuntimeError`
   or its recorded equivalent), a plan `discarded_artifacts:` `{name, reason,
   regen_recipe}` entry, AND an alert naming the closed gate. Only a large
   intermediate TENSOR qualifies; a generation / rollout-text / judge-output /
   metrics / configs discard NEVER PASSes (text is quota-immune on the non-LFS
   path — the v1 Step 3 generation-discard gate still binds).
5. **Overflow rerouting must have been ATTEMPTED before any discard.** Evidence
   showing only a main-repo 403 (no overflow attempt) is INVALID → FAIL; the
   canonical-repo `OVERFLOW_POINTER.json` + the `hf-overflow-routing.jsonl`
   event are the reroute evidence — confirm rerouted shards list on overflow.
6. **On PASS, append one `artifacts/registry.jsonl` row per artifact** via
   `scripts/artifact_registry.py` (id, `type`, HF/git path, issue,
   `size_bytes`, one-line `recipe` capsule) — the reuse registry the planner +
   methodology-writer read before any retrain/regenerate.

## Failure mode this spec was rewritten to catch (incident #456 — phantom URLs)

Task #456 (marker-implant training run) reached `awaiting_promotion`
with `has_clean_result=true`. Its clean-result body + `epm:results`
sentinel cited an HF checkpoint URL of the form
`superkaiba1/explore-persona-space/tree/<sha>/i432_..._marker_implant_step_checkpoints/checkpoint-1600`.
That URL did not exist on HF Hub at that revision or anywhere — no code
path uploaded the `{phase}_step_checkpoints/` per-step trajectory dir
to HF; `HF_HUB_URL` was a metadata string nothing actually pushed to.
The WandB run also had zero logged artifacts. Despite this, the
experiment PASSed upload-verification because the verifier trusted the
sentinel's URL string without HEAD-checking it. A downstream experiment
(#466) inherited a "checkpoint exists" claim that was false and had to
re-train the model two months later.

**Lesson: a URL in a sentinel is a STRING, not a permanent artifact.**
Step 2.5 closes this by HEAD-checking every claimed URL at its cited
revision via `verify_artifacts_exist` (the same helper /issue Step 6a.5
runs pre-launch to block on phantom carry-overs). Any unresolved URL is
a hard FAIL.

## Failure mode this spec was rewritten to catch (incident #365)

Task #365 ran a 72-cell factor screen that generated ~24 completion
pools (3 sources × 8 (A, B, C) configs) plus 24 off-policy Claude
pools, all on the pod under `data/issue_365/pools/`. None of those
pools were ever uploaded — the experimenter's `epm:results` marker
didn't mention them, the verifier accepted "Raw completions: N/A —
metrics-only eval pipeline" without checking the filesystem, and the
pod was terminated, destroying the data. The pools cost ~$20 in Claude
API and ~2 GPU-hours to generate, and were unrecoverable.

The lesson: **N/A claims must be backed by active discovery. The pod
filesystem is the source of truth for what was produced.**
