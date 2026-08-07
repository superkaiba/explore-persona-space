# Step 4: Worktree + dispatch implementer

Step body relocated verbatim from `.claude/skills/issue/SKILL.md`
(#2155). SKILL.md keeps the heading, the state machine and the
Orchestration Procedure router; read this file when the run reaches
this step.

---

Only if status is `approved`.

**4a. Worktree + draft PR.** Create `.claude/worktrees/issue-<N>` on
branch `issue-<N>`, symlink the repo `.env` into it, and open a draft PR.
```bash
# #506-safe: from a worktree cwd, `git rev-parse --show-toplevel` returns the
# WORKTREE root and doubles the path (.../issue-<N>/.claude/worktrees/issue-<N>);
# --git-common-dir resolves to <main>/.git so dirname is the main repo root.
REPO_ROOT=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")
WORKTREE="$REPO_ROOT/.claude/worktrees/issue-<N>"
bash "$REPO_ROOT/scripts/new_worktree.sh" "$WORKTREE" issue-<N> --issue <N>
# Sparse by default (~0.4G vs ~3.8G full); reuses if it exists (resume case);
# symlinks the repo .env (worktrees do NOT inherit it — RUNPOD_API_KEY /
# HF_TOKEN / WANDB_API_KEY dotenv loads fail without it).
# --issue <N> is inferred from the issue-<N> branch name when omitted
# (since #1054), but keep passing it explicitly.
```

**Sparse-worktree notes (task #596).** The worktree excludes
`eval_results/`, `external/`, `ood_eval_results/` bulk and pre-includes
this issue's own `eval_results/issue_<N>/` + `ood_eval_results/issue_<N>/`
cones (plus `eval_results/`'s immediate files, e.g. `INDEX.md`), so this
issue's artifact commits work with no ceremony. Two rules:
- **Reading another issue's eval JSONs** (parent baselines, comparison
  plots): `git -C "$WORKTREE" sparse-checkout add eval_results/issue_<M>`
  — instant. (Read-only fallback: the repo root's committed copy.)
- **Writing under a NEW dir below an excluded root** (e.g. a slug variant
  `eval_results/issue<N>_<slug>/`): run
  `git -C "$WORKTREE" sparse-checkout add eval_results/issue<N>_<slug>`
  BEFORE `git add`. A bare `git add` of an out-of-cone path fails loudly
  with "outside of your sparse-checkout definition" — the fix is
  `sparse-checkout add`, NOT `git add --sparse` (a `--sparse`-added file
  silently vanishes from the working tree on the next sparse-checkout
  mutation while staying committed).

**Worktree shell-ops rule (cwd resets between Bash calls).** The bash
tool's working directory is NOT preserved across separate calls, so a
relative `cd .claude/worktrees/issue-<N>` in one call has no effect on
the next. ALWAYS address the worktree with an absolute path or
`git -C "$WORKTREE" <cmd>` — never a bare relative `cd`. Resolve the
absolute path once with the #506-safe
`REPO_ROOT=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")`
recipe (as above) — NOT `git rev-parse --show-toplevel`, which from a
worktree cwd returns the worktree root and doubles the path — and reuse
`$WORKTREE` / `$REPO_ROOT` in every subsequent command. Corollary: this
issue's experiment files (scripts, configs, plan-referenced code) exist
ONLY in the worktree until Step 10d merges — a repo-root-relative
read/exec of `scripts/issue<N>_*.py` misses them (#1739); always prefix
with `$WORKTREE/`.

**Open the draft PR only if the branch is ahead of fetched `origin/main`.** `gh pr create` errors with `No commits between main and issue-<N>` when the branch has no commits yet (the common case before the implementer has run). Pre-check first (bounded fetch + `origin/main`-anchored aheadness):
```bash
# Base ref is FETCHED origin/main — new_worktree.sh cuts branches from
# refs/remotes/origin/main, and the repo root's local `main` routinely lags it.
# NEVER pipe this block — guard_piped_git_push.sh blocks a piped `gh pr create`
# (CLAUDE.md § Concurrent repo-root committers): a pipe masks the exit code.
timeout --kill-after=30s 120s git -C "$REPO_ROOT" fetch origin main --quiet || true
# Root-divergence probe (#1725, defense-in-depth): the shared repo root's
# local `main` and fetched `origin/main` are usually one strict-ancestor
# of the other (either lagging or ahead). Mutually non-ancestral = genuine
# divergence — a subsequent root-side call (git push origin main, another
# site's sync_repo_root.py) will need to reconcile it. Handle it proactively
# via the sanctioned single-flight helper; still-diverged after one sync is
# surfaced (never stepped over silently). 2>/dev/null silences the exit-128
# stderr on a missing ref (transient fetch failure at L2022, fresh clone):
# both --is-ancestor legs then return "not ancestor" and the probe fires a
# no-op sync via the idempotent single-flight helper. The downstream
# rev-list --count origin/main..issue-<N> pre-check below is unaffected
# either way — it reads origin/main directly, not local main.
if ! git -C "$REPO_ROOT" merge-base --is-ancestor origin/main main 2>/dev/null \
   && ! git -C "$REPO_ROOT" merge-base --is-ancestor main origin/main 2>/dev/null; then
  echo "[step4a] shared root diverged (local main and origin/main mutually non-ancestral) — running sanctioned sync"
  uv run python "$REPO_ROOT/scripts/sync_repo_root.py" || \
    echo "[step4a] sync_repo_root.py exited non-zero; proceeding — the pre-check below reads origin/main directly and is unaffected"
  # Re-probe once: sync exit 0 includes in-flight state (docstring: single-flight
  # returns 0 on concurrent caller), so confirm convergence via the same
  # ancestry test rather than the exit code.
  if ! git -C "$REPO_ROOT" merge-base --is-ancestor origin/main main 2>/dev/null \
     && ! git -C "$REPO_ROOT" merge-base --is-ancestor main origin/main 2>/dev/null; then
    echo "[step4a] shared root STILL diverged after one sync — proceeding (downstream sanctioned recoveries are the fallback; do not block Step 4)"
  fi
fi
if [ "$(git -C "$REPO_ROOT" rev-list --count origin/main..issue-<N>)" -gt 0 ]; then
  gh pr create --draft --head issue-<N> \
    --title "issue-<N>: <task title>" \
    --body "Closes task #<N>."
else
  echo "issue-<N> has no commits ahead of origin/main yet; skipping draft PR (open it after the implementer commits)."
fi
```

The git PR flow is substrate-independent — we still use GitHub for code
review of the diff (not for workflow state). The PR body references the
task number for traceability, but the source of truth for task state
stays in `tasks/<status>/<N>/`.

**4b. Dispatch implementer for the task type.** No pod is touched yet —
code gets written, reviewed, and dry-run locally before any GPU is
provisioned. Spawn the appropriate agent via `Agent()`:

| Task type | Implementer agent | Output marker |
|---|---|---|
| `experiment` | `experiment-implementer` | `epm:experiment-implementation` |
| `infra` / `batch` / code change | `implementer` | `epm:results` |
| `analysis` | `analyzer` (re-analysis only) | `epm:interpretation` (analysis-only path) |
| `survey` | `general-purpose` | `epm:results` |

**Env scrub for every subagent dispatch.** EVERY `Agent()` call this
skill makes — implementer, experiment-implementer, analyzer,
code-reviewer, clean-result-critic, interpretation-critic, experimenter,
upload-verifier, follow-up-proposer, consistency-checker, planner,
critic — passes `env=scrub_subagent_env(os.environ)` from
`explore_persona_space.orchestrate.spawn_agent`. The helper strips
`GH_TOKEN` and `GITHUB_TOKEN`; every other secret (WANDB_API_KEY,
HF_TOKEN, ANTHROPIC_API_KEY, OPENAI_API_KEY, RUNPOD_API_KEY, ...)
passes through unchanged so analyzer / experimenter still reach WandB /
HF Hub / Claude. Subagents post `events.jsonl` rows via
`scripts/task.py post-marker`, which inherits the user's env from the
orchestrator's process tree. See `tests/test_subagent_env_scrub.py` for
the allow-list.

**Result side of the same every-`Agent()`-call contract — background-agent
notification bodies arrive HTML-ESCAPED.** A BACKGROUND-Agent completion
delivered via a `<task-notification>` block carries its `<result>` field
HTML-escaped by the harness (`&&`/`<`/`>` arrive as amp/lt/gt entities).
NEVER persist notification-body text into a plan / marker / artifact —
re-extract the report from the agent's DURABLE output: the file the brief
told it to write, or the notification's `<output-file>` (a transcript
JSONL, not raw text — keep the last assistant text row). Output-file text
is CLEAN and gets NO `html.unescape()`; apply exactly ONE `html.unescape()`
round ONLY to notification-BODY-sourced text (the two sources are
exclusive-or). Canonical recipe + worked extraction code:
`.claude/skills/adversarial-planner/SKILL.md` §§ "De-escape harness HTML
entities before persisting" + "Extract the output-file text via the
transcript recipe" (#952, #1219; independently rediscovered by
#1287/#1288 — the pointer this paragraph exists to spare).

**Pre-split multi-deliverable builds at dispatch (#1810; precedents
#1090/#1775).** Before composing the brief, count the approved plan's
distinct planned CODE deliverables — new or substantially-rewritten
`scripts/` / `src/` / `tests/` files named in the plan's "File paths +
concrete diffs" section (doc-only files excluded). More than 4 code
deliverables ⇒ dispatch the build as sequential MICRO-SCOPED units by
default, never one monolithic brief. Grounding: #1775's 7-deliverable
build died at the subagent context ceiling after 139 tool calls /
~63 min (~20 tool calls per code deliverable puts the risk zone at
~5-6; ≥5 buys margin below the observed death at 7); a lower-count
build with a comparably large projected build volume (very large
per-file scope, OR other heavy-work compositions not covered by the
mandatory trigger above) MAY pre-split by judgment. Composition
trigger (mandatory, #1902 shape): a planned UNIT that combines a fit /
battery deliverable WITH figure-generation AND a smoke phase — OR any
TWO of those three where the smoke phase covers at least 2 pipeline
phases (data-gen + training, training + eval, etc.; operationally the
smoke's `## Smoke run` H2 has ≥ 2 `### <phase>` subsections) — is split
further REGARDLESS of the deliverable-count trigger's verdict: the fit
deliverable and the figure deliverable land in SEPARATE units, and the
smoke-bearing unit carries AT MOST ONE other deliverable. Rationale:
the ~20-tool-calls-per-deliverable basis (grounding, above) prices a
fit+figures+smoke unit at ≥2 deliverable-equivalents, so a 3-deliverable
unit of this composition sits at ~5-6 tool-call-equivalents — inside
the #1775 death zone (139 calls / 7 deliverables → ~20/deliverable →
5-6 as the risk boundary). #1902 measured this directly: a 3-unit
pre-split ran, unit C carried fits + figures + smoke, and died at 114
tool calls / 58 min on its final report turn (~5.7× the per-deliverable
basis), despite passing the deliverable-count trigger. Unit
shape: units of ≤3 deliverables each, run sequentially with a fresh
context per unit, each unit's brief scoped to its own self-contained
deliverable subset (the #1090 rounds-A/B shape). Marker contract:
INTERMEDIATE units commit their work by explicit path and RETURN a
commit manifest with NO implementation marker; ONLY the FINAL unit
runs the full per-phase `## Smoke run` H2 — covering every pipeline
phase INCLUDING phases built by earlier units, so the final unit's
smoke scope exceeds its own unit scope, and its brief states this
outright — and posts `epm:experiment-implementation` / `epm:results`
at max+1 per the existing brief contract. Round semantics: all units
run within ONE review round (no round-counter increment between
units); the Step 5 code-review ensemble reviews the whole round diff
once, after the final unit. Model rule: default session model per
unit — never a smaller-model pin (the Step 5b thrash-inverse rule,
#1090 forensics). Mid-split resume/idempotency: after each
INTERMEDIATE unit returns, the orchestrator posts an `epm:progress`
breadcrumb — `pre-split unit k/M complete: <commit SHAs>; remaining:
<deliverables>` — and a resuming session that finds the task at
`running` mid-split scopes its re-dispatch to the REMAINING units
(derived from the latest such breadcrumb plus the branch's committed
deliverables), NEVER the monolith and never a unit-1 re-dispatch over
already-committed files (the remaining-units re-dispatch satisfies
the resume table's "no implementation marker → re-spawn implementer"
row). TDD interaction: under `tdd_mode=true` the split applies to the
POST-approval implementation dispatch — test-authoring is one unit,
and the final unit posts the marker. Step 9b same-issue follow-up
rounds inherit this clause via the existing "follows the Step 4b
brief contract" reference — no separate Step 9b wiring. This clause
is the dispatch-time application of the Step 5b "Autocompact-thrash
respawn recipe" split (#1775's recovery applied this same split
post-death, at ~15 min triage + respawn overhead, when the plan's own
deliverable count had made it available at dispatch time); the
Step 5b recipe stays unchanged as the backstop for unforeseen thrash
deaths.

Brief passed to the implementer:
- The plan path (the `plans/plan.md` symlink, NOT the body text)
- Task number + worktree path + branch name
- Code-review history if this is a revision round (`epm:code-review v<m>`)
- Required `report-back` contract — the canonical 4-H3 marker shape from
  `.claude/agents/experiment-implementer.md` Report Format + the matching
  `## Smoke run` H2 from `.claude/agents/code-reviewer.md` Steps 0.5/0.6.
  The brief MUST quote these section labels verbatim; ad-hoc alternative
  labels (e.g. `(a) Plan adherence`, `(b) Files touched`, `(c) How to
  run`, `(d) Smoke run`) cause the Codex `code-reviewer` to FAIL on
  `marker-shape` even when the implementer faithfully follows the brief.
  Canonical labels (use VERBATIM in the brief):
  - `### (a) What was done`
  - `### (b) Considered but not done`
  - `### (c) How to verify`
  - `### (d) Needs human eyeball`
  - (optional `### (e) Concerns addressed` — only when prior open
    binding concerns from `concerns.jsonl` were verified this round;
    see `code-reviewer.md` Step 0.5 + Step 0.8)
  - `## Smoke run` H2 (per Step 0.6) with one `### <phase-name>` per
    CPU-feasible pipeline phase (typical: `### data-gen`, `### training`,
    `### eval`), each carrying the exact command, the slice size, exit
    code `0`, and a one-line artifact digest. **Smoke run is its own
    `## H2` — NEVER a `### (d) Smoke run` H3.** Folding the smoke run
    into the (d) slot displaces `### (d) Needs human eyeball` and is
    itself a `marker-shape` FAIL.

  (#506)

  The brief MUST also carry the deferred-production-path duty: any
  deferred feature the approved plan's PRODUCTION path requires is
  persisted via `task.py raise-concern <N> --concern-id <id>
  --severity CONCERN --summary "<≤200c>" --by experiment-implementer
  --round <n>` (BLOCKER if the production path provably crashes
  without it) BEFORE posting the implementation marker — a `(d)`
  bullet is not a substitute (#509). Belt-and-suspenders on
  `experiment-implementer.md` § "Deferred production-path TODOs are
  persisted concerns, not (d) prose", so round-N briefs surface the
  duty without the implementer having to recall its agent spec.
- The brief MUST also carry the gate-scope verification duty (#1288):
  before posting the report marker, the implementer **commits their edits
  first** — the selector diffs COMMITTED state against fetched
  `origin/main`, so uncommitted edits silently degrade to the
  invariant-only set (#1717 defect (d)) — then enumerates the Step
  9c selection from the worktree — `uv run python
  scripts/select_step9c_tests.py --json 2>/dev/null` (stderr routed off
  stdout so the JSON stays parseable — the informational NOTE / WARN /
  sizing lines go to stderr BY DESIGN; #1717 defect (b)), the DEFAULT
  invocation (the base defaults to FETCHED `origin/main` per #1289;
  `--base main` exists only to deliberately diff against the local ref,
  per Step 9c step 1a — never for this duty) — pin-sweeps the enumerated
  test files for every
  literal / command fragment / symbol the diff changed or deleted, and
  runs the diff-linked + pin-hit subset locally, deferring only the
  invariant-only remainder to the gate (which remains the backstop),
  reporting the pin-sweep field with the verbatim deduplicated
  hit-file list + its `sweep_scope:` universe token
  (`selector-universe` | `repo-wide` — the REALIZED universe; #1651)
  — never a count-only or glob-family summary (#1494;
  >20 files → fenced block under the Gate-scope line).
  Belt-and-suspenders on `implementer.md` § After Implementation items 1 + 1a,
  so round briefs surface the duty without the implementer having to
  recall its agent spec (the #509 precedent).
- **A brief NEVER suppresses the implementation marker.** Never instruct an
  implementer to "post nothing" / skip its `epm:experiment-implementation` /
  `epm:results` marker — the code-review ensemble's mechanical contract KEYS
  on that four-section marker, so suppressing it manufactures a
  `marker-shape` blocker and an extra fix round (#1900). A
  round whose diff is deliberately partial still posts the marker, saying so.
- **Marker-version discipline — a brief NEVER instructs a literal marker
  version.** Any brief line about posting `epm:experiment-implementation` /
  `epm:results` / `epm:proposed-tests` says: "post at the next version —
  read `events.jsonl` for the highest existing version of the kind and use
  max+1, or omit `--version` (the CLI derives max+1)". Never "post as `v1`"
  or any literal `v<k>`: on a fresh task max+1 IS 1, but on a follow-up
  round / TDD resume / crash-recovery re-post prior rows exist, and an
  explicit `--version` beats the CLI's safe default (#825). See
  `experiment-implementer.md` / `implementer.md` § Posting review-round
  markers.
- **Instruction: work ONLY inside the worktree; never touch a pod; post
  progress as `events.jsonl` rows via
  `uv run python scripts/task.py post-marker <N> epm:progress --note '...'`.**
- **If `batch`:** make ONE commit per plan section (the planner produced
  N independent sections, one per body item). Commit message format:
  `[N/M] <plan section title>` where N is the 1-indexed item and M is
  the total. Code-reviewer reviews the whole diff; this convention
  keeps the history bisectable per item if a single fix needs to be
  reverted later.
- **TDD mode (opt-in).** Set `tdd_mode=true` in the brief if EITHER:
  (a) the approved plan body contains a literal `### TDD: yes` line, OR
  (b) the task body / latest user comment in `comments.jsonl` contains
  `request-tdd`. When `tdd_mode=true`, the implementer writes tests
  first, posts them as `epm:proposed-tests v<n>` (max+1), and EXITs without writing
  implementation. This skill then parks at `running` (implementing
  sub-phase) and waits — see Resume semantics below: an `approve-tests`
  marker posted via `task.py post-marker <N> epm:approve-tests` **after**
  the `epm:proposed-tests` event is the resume signal, at which point
  this skill re-dispatches the implementer with `tdd_approved=true` and
  the implementer writes the code to make the approved tests pass. If a
  resumed `/issue <N>` finds the proposed-tests event still without
  approval, it shows the marker timestamp + the literal `approve-tests`
  instruction and EXITs again. This is the only opt-in user gate in the
  pipeline (see CLAUDE.md auto-continuation policy gate #8).

Move status to `running` (the implementing sub-phase rolls up under
`running`):

> **Same-issue follow-up round?** At `followups_running`, SKIP this
> `set-status` (status-hold rule, Step 9b § Same-issue follow-up loop step 3;
> code-enforced — `task.py` refuses the flip) — phase visibility comes from
> `stage=followup-<phase>` breadcrumbs, not status flips.

```bash
uv run python scripts/task.py set-status <N> running \
  --note "Dispatched implementer; awaiting epm:experiment-implementation."
```

Before exiting, post the §5 marker:
```bash
uv run python scripts/post_step_completed.py --issue <N> --step 4b \
  --exit-kind clean --notes "implementer dispatched; awaiting epm:results"
```
EXIT. Implementer runs autonomously.
