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
  # Title transport (#2241 r2): the title is resolved AS DATA — command
  # output is never shell-parsed — so a hostile title cannot inject.
  # Resolver fence (#2241 r3, concern title-resolution-failure-masking):
  # rc-gated + timeout-fenced as its own step — every task.py invocation
  # (reads included) pays the branch-guard resolution (#996 bounded rebase
  # wait, EPM_TASKPY_REBASE_WAIT_SECONDS default 120 s; RuntimeError on
  # detached HEAD / husk timeout); unfenced+unchecked, that failure was
  # masked by jq exiting 0 on empty input and the create shipped a
  # degraded "issue-<N>: " prefix-only title. On resolver failure, jq
  # failure, or an empty / whitespace-only title (#2241 r4, concern
  # whitespace-only-pr-title — set-title stores input unstripped, so a
  # bare -z passed "   "): log, SKIP the create, fall through.
  TITLE_RC=0
  TASK_JSON=$(timeout --kill-after=30s 150s uv run python "$REPO_ROOT"/scripts/task.py view <N> --json) || TITLE_RC=$?
  RAW_TITLE=""
  if [ "$TITLE_RC" -eq 0 ]; then
    RAW_TITLE=$(printf '%s' "$TASK_JSON" | jq -r '.frontmatter.title // empty') || TITLE_RC=$?
  fi
  if [ "$TITLE_RC" -ne 0 ] || [ -z "${RAW_TITLE//[[:space:]]/}" ]; then
    echo "[step4a] title resolution failed or empty (rc=$TITLE_RC) — skipping draft-PR create; the Step 5 draft-PR ensure (#2241) retries at the first review round; Step 10d's payload-aware arm (#2240) stays the merge-time backstop"
  else
    PR_TITLE="issue-<N>: $RAW_TITLE"
    # #2241 r3 (sanctioned opportunistic fence): the create is rc-gated +
    # timeout-fenced too, matching the Step 5 ensure idiom.
    if timeout --kill-after=30s 120s gh pr create --draft --head issue-<N> \
         --title "$PR_TITLE" \
         --body "Closes task #<N>."; then
      echo "[step4a] opened draft PR for issue-<N>"
    else
      echo "[step4a] gh pr create failed (rc!=0) — proceeding; the Step 5 draft-PR ensure (#2241) retries at the first review round"
    fi
  fi
else
  # This arm fires by construction on a fresh branch: Step 4a runs at the
  # approved->running transition, BEFORE the implementer's first commit.
  # The Step 5 draft-PR ensure (#2241) re-runs the create at the first
  # review round — the first point where commits exist — so this skip is
  # EXPECTED, not a defect. Step 10d's payload-aware no-usable-PR arm
  # remains the merge-time backstop, posting a [step10d-no-pr-anomaly]
  # epm:progress note when it fires.
  echo "issue-<N> has no commits ahead of origin/main yet; skipping draft PR — the Step 5 draft-PR ensure (#2241) opens it at the first review round; Step 10d's payload-aware arm (#2240) opens it at merge time if the branch carries novel payload."
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

**Fan-out completion contract in every work-producing brief (#2041).**
Sibling of the env-scrub contract above — EVERY brief this skill composes
whose subagent PRODUCES work products (implementer builds, fold/fan-out
analysis agents, scouts, the Step 10d residual-conflict dispatch) RESTATES
the CLAUDE.md § Teammate coordination (d)/(f)/(g) contract: (1)
deliverables land durably IN the producing turn — commit+push by explicit
path; a repo-root code payload carries the Step 9a-ter § Worker-brief
composition duty; (2) the report is the turn's FINAL action; (3) a
delegated gate-wait is waited out SYNCHRONOUSLY inside the turn (a bounded
`Monitor` until-loop — foreground `sleep` chains are hook-blocked — never
end the turn on a background call the subagent itself armed). At every
fan-out JOIN the orchestrator consolidates the returned reports into a
durable home (task `artifacts/`, repo doc) in the same turn —
offer-to-save is the banned shape. Durability pin:
`tests/test_teammate_coordination_pins.py::test_fanout_completion_contract_pinned`.

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
fit+figures+smoke unit at ≥2 deliverable-equivalents — inside the
#1775 death zone; #1902 measured this directly (its unit C carried
fits + figures + smoke and died at 114 tool calls despite passing the
deliverable-count trigger). Unit
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
post-death, when the plan's own deliverable count had made it
available at dispatch time); the
Step 5b recipe stays unchanged as the backstop for unforeseen thrash
deaths.

Shared-worktree note (#2158): a shared worktree is the EXPECTED shape for
a multi-unit split — the units build on one branch, and a branch checks
out in exactly one worktree — so an INDEPENDENT concurrent session may
legitimately be working in the same tree. Cross-session writer
arbitration (probe for live writers, durable file-set claim markers,
sequence-after-commit or split file sets) is therefore a NORMAL
pre-dispatch requirement here, not an edge case:
`.claude/rules/cross-session-writer-arbitration.md`. Emitter convention
(arm B of the pre-split review guard, #2158): each unit's `stage-dispatch`
note carries `unit=<k>` — the guard's second arm keys on that token, and
a pre-split round that omits it gets zero arm-B protection in exactly the
pre-breadcrumb window where the #1336 v132 incident lived.

Brief passed to the implementer:
- The plan path — the ABSOLUTE canonical main-checkout form
  `$(uv run python "$REPO_ROOT"/scripts/task.py find <N>)/plans/plan.md`,
  resolved at compose time with `plan_version=v<K>` (the extensionless
  readlink) stated in the brief (NOT the body text) — never a relative
  `tasks/...` path: the implementer's cwd is the worktree, whose `tasks/`
  tree is frozen at base (#2422). A manifest-consuming round names
  `$TASK_DIR/artifacts/planned_manifest.json` the same way.
- Task number + worktree path + branch name
- Code-review history if this is a revision round (`epm:code-review v<m>`)
- The § Fan-out completion contract restatement (#2041, above) with BOTH
  halves explicit in the brief text: the staged-but-uncommitted prohibition
  AND the wait mechanism — any delegated gate-wait is waited out
  SYNCHRONOUSLY via a bounded in-turn `Monitor` until-loop (foreground
  `sleep` chains are hook-blocked). A brief stating the prohibition without
  the mechanism invites an invented background-watcher shape that orphans
  the landing (#2422 folded sibling: ~62 min of work left uncommitted
  behind an unobserved lint gate).
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
  persisted concerns, not (d) prose".
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
  Belt-and-suspenders on `implementer.md` § After Implementation
  items 1 + 1a (the #509 precedent).
- **A brief NEVER suppresses the implementation marker.** Never instruct an
  implementer to "post nothing" / skip its `epm:experiment-implementation` /
  `epm:results` marker — the code-review ensemble's mechanical contract KEYS
  on that four-section marker, so suppressing it manufactures a
  `marker-shape` blocker and an extra fix round (#1900). A
  round whose diff is deliberately partial still posts the marker, saying so.
  DISPLACEMENT is the same defect (#2248, from #1336 round v20): a brief
  that specifies its own return contract ("Report back: the commit SHA,
  the diffstat, which tests ran, …") WITHOUT naming the marker duty crowds
  the marker out exactly as an explicit skip would — returned Agent text
  is NOT durable task state, so a return-only contract loses the record
  the mechanical contract keys on, and the Step 5c-bis strip cannot
  rescue it (the strip's precondition is a present + conforming marker;
  here the missing marker IS the blocker). Any brief that specifies a
  return format therefore names the marker duty alongside it — e.g.
  "post your report as the `epm:experiment-implementation` marker at the
  next version (omit `--version`; the CLI derives max+1) AND return a
  short summary as your final text".
- **Marker-version discipline — a brief NEVER instructs a literal marker
  version.** Any brief line about posting `epm:experiment-implementation` /
  `epm:results` / `epm:proposed-tests` says: "post at the next version —
  read `events.jsonl` for the highest existing version of the kind and use
  max+1, or omit `--version` (the CLI derives max+1)". Never "post as `v1`"
  or any literal `v<k>` — an explicit `--version` beats the CLI's safe
  default (#825). See
  `experiment-implementer.md` / `implementer.md` § Posting review-round
  markers.
- **Instruction: work ONLY inside the worktree; never touch a pod; post
  progress as `events.jsonl` rows via
  `uv run python scripts/task.py post-marker <N> epm:progress --note '...'`.**
- **If `batch`:** make ONE commit per plan section (the planner produced
  N independent sections, one per body item). Commit message format:
  `[N/M] <plan section title>` where N is the 1-indexed item and M is
  the total. Code-reviewer reviews the whole diff; the convention keeps
  history bisectable per item.
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
