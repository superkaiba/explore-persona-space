---
paths:
  - ".claude/rules/code-reviewer-section-reference.md"
description: >
  Full sub-check recipes, FAIL templates, waiver forms, and incident grounding
  for code-reviewer.md's Step 0.5-0.9 pre-diff gate stack, relocated from
  .claude/agents/code-reviewer.md (the agent spec is a per-spawn system-prompt
  cost; the #1090/#2054 autocompact deaths are fixed-overhead deaths). Loaded
  ONLY via the explicit § pointer lines in code-reviewer.md — the self-matching
  `paths:` glob keeps this file out of every other agent context.
---

# Code-reviewer section reference (Step 0.5-0.9 gate detail)

One H2 per relocated gate, detail relocated verbatim from
`.claude/agents/code-reviewer.md`. Read ONLY the section the gate under
review needs: Grep the heading, then a chunked `Read` of that span (per
code-reviewer.md § Context budget) — never the whole file. The OPERATIVE
trigger + blocker-tag + severity contract for every gate stays in
code-reviewer.md; this file carries the extended recipes, verbatim FAIL
templates, waiver forms, and incident grounding.

## Step 0.5 detail — implementation marker shape

Before reading the plan, verify the implementer's report marker conforms to the
required four-section shape. Fetch the highest-version `epm:experiment-implementation`
(for `type:experiment`) or `epm:results` (for `type:infra` / `type:survey`
code-change paths) on the experiment, and check that
the body contains all four H3 subsections in order:

- `### (a) What was done`
- `### (b) Considered but not done`
- `### (c) How to verify`
- `### (d) Needs human eyeball`

Plus, inside `(c)`, at least one copy-pasteable command (fenced code block) and
one observable success signal — no "see PR" / "tests pass" handwaves.

**FAIL only when contract evidence is genuinely ABSENT — not when it is
present but imperfectly formatted.** Distinguish two cases:

- **Genuine absence** (a required H3 section is missing, mislabeled, or
  empty, OR `(c)` contains no copy-pasteable command at all): return verdict
  FAIL with a single `Critical` issue tagged `marker-shape`, AND still read
  the diff and report any substantive findings in the same pass (do not
  short-circuit — see Step 0.7). Before claiming a section is *entirely
  absent*, confirm you are reading the highest-version marker in canonical
  task state (`uv run python scripts/task.py view <N>`), not a stale worktree
  copy — a stale read is the most common false "absence":

  > `epm:<kind> v<n>` does not conform to the four-section shape required by
  > `markers.md` and `agents/<name>.md` Report Format. Missing/empty
  > sections: [list]. Re-post `v<n+1>` with the required structure.

- **Present but imperfect** (all four sections exist with real content and
  `(c)` carries at least one command, but the ordering is slightly off, a
  section is terse, or you would have phrased the digest differently): this
  is at most a `CONCERNS` bullet under "Style / Consistency", NEVER a
  standalone FAIL. PROCEED to review the diff (Steps 1–7).

**Mandatory `(c)` fields — pin-invocation check (#1716).** Inside `(c)`, the
required-field roster includes the ruff-policy pin field (`implementer.md`
§ (c) — the field that quotes `uv run pytest
tests/test_ruff_policy.py::test_live_workflow_helpers_clean_under_full_ruleset
-x` AND its exit code). When the diff touches any path in
`tests/test_ruff_policy.py`'s `LIVE_WORKFLOW_HELPERS` roster and this field
is entirely ABSENT from `(c)`, return the marker-shape FAIL above with the
missing-field name in the [list]. When the field is PRESENT but its
formatting differs from the template (e.g. the command is spelled slightly
differently, the exit-code token uses `rc=0` vs `returncode=0`), this is
"Present but imperfect" and at most a `CONCERNS` bullet under "Style /
Consistency" — NEVER a standalone FAIL (the present-but-imperfect →
CONCERNS rule above governs). The `marker-shape` tag remains strippable by
/issue Step 5c-bis (`code-reviewer.md` L649-651) when the pin evidence IS
present but formatted differently — a stripped FAIL indicates the
substantive pin evidence was recovered elsewhere in the marker body.

This gate exists because the four-section shape is the user's primary
verification surface — a marker that omits `(c)` forces the user back into
the diff. But its job is to catch *absence*, not to police cosmetics: a
reviewer that FAILs round after round on the *presentation* of evidence that
is demonstrably present never reviews a line of code (the gate-hopping
failure mode). Catching genuine absence here is cheaper than catching it at
Step 10d merge; nitpicking present evidence is more expensive than letting
it through as a CONCERNS.

For `type:experiment` `epm:results` markers, check the existing `## Sample
outputs` requirement in `markers.md` instead — the four-section shape applies
to implementation reports, not experiment-run results which have their own
contract.

**Optional 5th section `### (e) Concerns addressed`.** When prior rounds
left open binding concerns in `concerns.jsonl` (see Step 0.8 below), the
implementer marker SHOULD include this OPTIONAL 5th H3 listing per-
concern_id what they did and the round at which `task.py address-concern`
was called. The four-section shape (a/b/c/d) remains the contract: when
no prior open concerns exist, the marker is fully PASS-able WITHOUT a
(e) section, and a missing (e) is NEVER a `marker-shape` FAIL. When
prior concerns DID exist and the implementer claims to have fixed them,
the absence of (e) becomes a CONCERNS bullet under "Style / Consistency"
(not a standalone FAIL — the reviewer still verifies via `task.py
list-concerns <N> --open-only --json`, which is the canonical signal).

## Step 0.55 detail — smoke-architecture marker presence and shape

For `type:experiment` tasks, verify a separate `epm:smoke-architecture-check`
events row EXISTS in canonical task state — `uv run python scripts/task.py
view <N> --json`, never a possibly-stale worktree `events.jsonl` (the same
false-absence caution as Step 0.5) — with a parseable `verdict:` line, one of
`PASS_UNIFIED` | `PASS_PARTIAL arms_stubbed=<list>` | `PASS_AUTHORIZED_STUB
arms_stubbed=<list>` | `PASS_CANARY canary_cell=<id>` | `FAIL_NO_CANARY`. The
implementer posts it ONCE at pre-flight (experiment-implementer.md "Before
writing code" item 5); fix rounds do NOT re-post, so the check is
presence-on-task (any version), NEVER presence-per-round — a fix-round review
with a round-1 marker PASSes this gate.

- **Genuine absence** (no such events row at all, OR the row carries no
  recognizable `verdict:` line): return verdict FAIL with a single `Critical`
  issue tagged `marker-shape` whose body NAMES `epm:smoke-architecture-check`
  (the orchestrator's Step 5c-bis strip is keyed PER BLOCKER on that name —
  a Step 0.55 blocker body names exactly ONE marker kind,
  `epm:smoke-architecture-check`, never a combined Step 0.5 + 0.55 blocker),
  AND still read the diff (Step 0.7):

  > No `epm:smoke-architecture-check` events row exists in canonical task
  > state. experiment-implementer.md "Before writing code" item 5 mandates it
  > before code-review-PASS, and /issue Step 6d.0 will refuse dispatch without
  > it — AFTER pod provisioning has already run. Post it as a separate events
  > row (`verdict: PASS_UNIFIED` | `PASS_PARTIAL arms_stubbed=<list>` |
  > `PASS_AUTHORIZED_STUB arms_stubbed=<list>` |
  > `PASS_CANARY canary_cell=<id>` |
  > `FAIL_NO_CANARY`); prose in a dispatcher header or an HTML comment inside
  > the `epm:experiment-implementation` note does NOT count (incident #811:
  > the claim lived in a dispatcher header across 5 rounds, both reviewers
  > PASSed, and the gap surfaced only at Step 6d.0 post-provision).

- **Present with `verdict: FAIL_NO_CANARY`**: NOT a reviewer FAIL — Step 6d.0
  (gates.inline id=10) owns FAIL_NO_CANARY adjudication (bounce to planning).
  Note it as a CONCERNS bullet so the orchestrator sees it early.
- **Present + parseable** (any PASS verdict): verify the marker's
  internal SHAPE before proceeding. Verify a line-anchored
  `arm-registry:` line is present in one of its two accepted forms
  (#2176) — structured: `arm-registry: source=<expr> file=<path>
  n=<int> members=<sorted-comma-list>`; or vacuous:
  `arm-registry: N/A — <reason>` when no registry exists — a missing
  or malformed line is a `marker-shape` blocker. Placement note: the
  `arm-registry:` line is a TOP-LEVEL key, a sibling of
  `per-arm-resolution:` — inside the per-arm span it TERMINATES the
  span rather than reading as a row (the #2176 one-token
  `_MARKER_TOP_KEY_RE` extension pins that it is never swallowed as a
  phantom arm). Grep the plan §4 Design for the
  arm/rung/condition names it declares (a `kind: experiment` plan
  typically enumerates these; a `kind: infra` plan often names none —
  the vacuous `per-arm-resolution: N/A — no registry or plan-named arms`
  line, paired with `arm-registry: N/A — no phase/arm registry`,
  satisfies the shape check by construction). For every registry or plan-named
  arm (the marker's `arm-registry:` members list ∪ the plan-named
  arms), confirm the marker's `notes: per-arm-resolution:` sub-block
  contains a row. Also verify the `notes: import-resolution: <cmd>`
  line matches one of the three shapes named in
  experiment-implementer.md Axis 1: (a) the dispatcher's
  `--import-check` mode, (b) a `from <mod> import (<names>)` form
  enumerating every deferred symbol, or (c) a bare `import <mod>`
  tagged `import-resolution-mode: top-level-only` — only when the
  entrypoint's changed lines contain no function-body imports (grep
  the entrypoint's diff for `def .*:` blocks containing `from` or
  `importlib`; a match returns a `marker-shape` blocker). Then bind
  per verdict:
  - `verdict: PASS_UNIFIED` — every per-arm row must read `REAL` or
    `N/A`. Any `FALLBACK` row is a `marker-shape` blocker (the
    verdict should have been `PASS_PARTIAL`). An `N/A` row whose text
    cites authorized-stub vocabulary (e.g. `N/A — authorized smoke
    stub`) is likewise a `marker-shape` blocker — authorized stubs
    take `PASS_AUTHORIZED_STUB`, never a re-labeled `N/A` (the
    #2163-v4 improvisation, retired by #2171).
  - `verdict: PASS_PARTIAL arms_stubbed=<list>` — the `<list>` must
    equal (as a set) the names of every `FALLBACK`-rowed arm.
  - `verdict: PASS_AUTHORIZED_STUB arms_stubbed=<list>` — same
    set-equality binding as `PASS_PARTIAL` (the `<list>` must equal
    the `FALLBACK`-rowed arms); the marker-vs-PLAN subset check is NOT
    the reviewer's — Step 6d.0's `task.py check-authorized-stub`
    checker owns it (#2171).
  - `verdict: PASS_CANARY canary_cell=<id>` — same REAL / N/A
    invariant as `PASS_UNIFIED` (a `FALLBACK` row here is a
    `marker-shape` blocker).
  - `verdict: FAIL_NO_CANARY` — no per-arm binding (Step 6d.0
    bounces regardless).

  You do NOT re-adjudicate whether a `REAL` row actually ran real
  code — that substance remains Step 6d.0's; the reviewer only
  checks the marker's internal shape (rows present, verdict
  consistent with rows, import-resolution shape matches one of the
  three, arm-registry line well-formed). A shape violation returns a
  single `Critical` blocker tagged `marker-shape` whose body NAMES
  `epm:smoke-architecture-check` (Step 5c-bis strip is keyed on that
  name; the blocker names exactly ONE marker kind — never combined).

  **Arm-registry substance split (#2176)** — stated ARM-EXPLICITLY.
  Substance here means: `members=` equals the driver registry's ACTUAL
  key set. Whenever the marker's `file=` resolves in the worktree, that
  set-equality is owned by the MECHANICAL arm — Step 6d.0's checker in
  driver-recompute mode (`task.py check-smoke-arch-registry <N>
  --repo-root <worktree>`). The REVIEWER owns it as the FALLBACK arm
  whenever it does not resolve (the checker's `OK` line then reads
  `marker-only`), and as defence-in-depth whenever the diff itself
  touches the named driver file. The reviewer duty is a COMMAND SHAPE,
  not a vibe: on a structured `arm-registry:` line whose `file=` is in
  the diff, ENUMERATE the named `source=` symbol's keys — open the file
  at the diff and read the dict-literal keys, or run
  `uv run python scripts/task.py check-smoke-arch-registry <N>
  --repo-root <worktree>` — and assert SET-EQUALITY with `members=`. A
  mismatch is a **substantive** blocker, NOT `marker-shape`. A
  symbol-presence grep is EXPLICITLY INSUFFICIENT — it proves the
  registry exists, not that `members` equals its key set. On the N/A
  form, verify the no-registry claim against the diff with the same
  concreteness (does a changed entrypoint define a `PHASES`-style
  dispatch table? — `grep -n '^PHASES' <changed entrypoints>`), keeping
  the two duties parallel so neither reads as optional.

## Step 0.8 detail — prior open binding concerns

Before reading the plan, fetch the canonical concerns ledger:

```
uv run python scripts/task.py list-concerns <N> --open-only --json
```

Inherit each open concern (severity=`BLOCKER` or `CONCERN`, latest event
`raised` or `verified-open`) as context for this round. Two consequences:

- Any open binding concern from a prior round MUST be addressed (the
  implementer claims fix → verify; not fixed → re-raise; addressed and
  no longer visible → call `task.py address-concern <N> --concern-id
  <id> --by code-reviewer --round <n>` to record verification).
- A new substantive concern this round that you want the orchestrator to
  bind MUST be persisted via `task.py raise-concern <N> --concern-id
  <kebab-id> --severity CONCERN|BLOCKER --summary <80c> --by
  code-reviewer --round <n>`. The `--summary` is capped at 200 chars —
  compose the one-liner within the cap and put detail in the evidence
  field / verdict body; an over-cap `--summary` via the CLI is
  auto-truncated at a word boundary with a loud warning (full text
  shifted into `--evidence` when evidence is empty; programmatic callers
  still get `ValueError: summary too long`). Verdict-body
  concern bullets that are NOT persisted remain opportunistic (the
  historical PASS+CONCERNS auto-advance contract applies).
- **A deferred feature the plan's PRODUCTION path requires is ALWAYS a
  persisted concern — never prose-only.** When the implementer's report
  (a `(d) Needs human eyeball` bullet, a TODO in the diff like
  `# Per-seed reconstruction goes here (TODO inflow)`) or your own
  reading of the code shows that a registered statistic, correction, or
  data input the approved plan requires on the production path is
  deferred — such that the production run would crash or silently
  degrade (e.g. a load-bearing adjustment quietly no-ops to its
  uncorrected value) without it — you MUST persist it via `task.py
  raise-concern` (severity CONCERN minimum; BLOCKER when the production
  path provably crashes without it), even when your verdict is PASS.
  "Surface as a follow-up before the production run" in report or
  verdict prose is NOT a substitute: the /issue Step 5c-ter dispatch
  gate reads `concerns.jsonl`, not prose, so an unpersisted deferral
  dispatches the pod and the crash lands at run time (incident #509: a
  known-at-review-time guaranteed production crash on the fact arm's
  missing per-seed-SE inflow lived only in verdict/report prose across
  rounds 2-3, review PASSed, the pod dispatched, production scoring
  crashed exactly as predicted, and the run descoped to `--smoke` —
  shipping un-attenuation-adjusted statistics).

See `workflow.yaml § concerns_protocol` for the full severity tier
mapping and reviewer round protocol.

## Step 0.6 detail — end-to-end smoke gate

For `type:experiment` tasks, a PASS is INVALID on a script that was only
`--help`'d, import-checked, or `--dry-run`. Before reviewing the diff,
confirm the implementer's `epm:experiment-implementation` report carries a
`## Smoke run` section showing EACH PHASE of the experiment pipeline was
run ONCE on a tiny real slice — not just training or data-gen. "Phase" =
any distinct entrypoint the pipeline executes end-to-end (typical
experiments: data-gen, training, eval; some add separate analysis / upload
steps). Eval rigs especially must be smoke-exercised end-to-end on a tiny
slice (1 seed, the minimum contexts / cells, the base model or a tiny
throwaway checkpoint); a never-before-run eval script that was only
import-checked or that piggy-backed on the training script's smoke is the
canonical missing-phase case — shallow latent bugs (corpus-size floors,
missing helpers, generator-reuse, sentinel filters, aggregation-tuple
unpacks) surface one-per-run at the real eval phase otherwise, each costing
a full pod cycle (incident: #408 burned six relaunches catching one bug
per cycle on a 203 KB eval rig that had never been run end-to-end).

For each phase the implementer should record a sub-section under
`## Smoke run` — recommended layout `### <phase-name>` (e.g.
`### data-gen`, `### training`, `### eval`). Each sub-section MUST show:

- the exact command that was run,
- the slice size (how it was kept tiny),
- the exit code (must be `0`),
- a one-line digest of the produced artifact (path + shape / row count) —
  proving a REAL output was written, not a stub.

**Many-call fit/battery phases: production-shape unit timing + full-scale
extrapolation (REQUIRED).** When the pipeline contains a phase that loops a
fit / dense factorization (svd/eigh/lstsq/GCV-ridge) / draw battery over
cells × folds × layers × arms × draws, that phase's `## Smoke run`
sub-section MUST additionally report: (a) the wall-time of ONE unit call at
PRODUCTION shape (full N/H — a tiny-slice per-call time does not scale;
#823's smoke hid a ~62× per-call error — 2 s asserted vs ~125 s measured —
exactly this way), and (b) the extrapolation
`smoke_per_call_wall × full_call_count / parallelism` compared against the
plan §9 row. A >2× gap with NO separate `epm:compute-deviation` row in
`events.jsonl` is a FAIL with a `substantive` blocker (NOT
`smoke-run-missing` — the smoke ran; the projection contract was skipped). A
phase genuinely unable to run one production-shape unit call locally uses the
GPU-bound carve-out's substitute-coverage form with a FLOP/kernel-derived
per-call estimate instead. N/A when no phase loops a
fit/factorization/battery.

**Harmful-content corpora digest note.** For phases over EM / refusal-bait /
harmful-advice / real-world-corpus (LMSYS/WildChat-class; #1073) corpora
the digest is path + row count + hash + field names
ONLY — the implementer spec forbids pasting row text
(experiment-implementer.md § Content hygiene). Never request raw-row or
sample-text evidence for such artifacts, and never `cat` them yourself when
verifying; a path + count + hash digest fully satisfies this gate for those
corpora (incident: task #537, 2026-06-10).

**FAIL only when there is no proof some phase ran on real data.** That means
the `## Smoke run` section is absent, OR any phase the pipeline actually
executes is missing a sub-section, OR any sub-section shows only
`--help` / `import` / `--dry-run` evidence, OR the exit code is non-zero,
OR there is no artifact digest at all (no proof a real output was written).
The most common case: training has a smoke sub-section, the eval rig does
not. In that case return verdict FAIL with a single `Critical` issue tagged
`smoke-run-missing` (naming the missing phase in the body), AND still read
the diff and report substantive findings in the same pass (do not
short-circuit — see Step 0.7):

> `epm:experiment-implementation v<n>` has no proof the <phase> script ran
> on a tiny real slice (`## Smoke run` missing the `### <phase>` sub-section,
> shows only --help/import, exits non-zero, or carries no artifact digest).
> An experiment script that has never produced a real artifact is not
> PASS-able — a `404` / shape bug / empty-dataset silent-fail would only
> surface after a pod is provisioned and GPU-minutes are burned. Re-post
> `v<n+1>` with a `### <phase>` smoke sub-section (command + slice size +
> exit code 0 + artifact digest).

**GPU-bound-phase carve-out (do NOT FAIL `smoke-run-missing`).** Do NOT
FAIL `smoke-run-missing` on a phase whose `## Smoke run` sub-section is
explicitly titled `### <phase-name> — Carve-out (GPU-bound)` AND lists
all three substitute coverage items (REAL CPU smoke of the CPU-runnable
portion + dispatcher dry-run + signature smoke per
`experiment-implementer.md` § GPU-bound-phase carve-out). Each
substitute item must carry its own command, exit code 0, and one-line
artifact digest; the sub-section must also name the GPU constraint in
one sentence (e.g. "4× H100 ZeRO-3 required; local VM has no
CUDA-capable GPU"). The carve-out exists because phases like
`accelerate launch` + ZeRO-3 full-FT, vLLM batched eval, or TP=8 ≥7B
inference cannot be smoke-run on the local VM in their production
shape — the three substitute items together exercise the same dispatcher
plumbing, env passthrough, sentinel + `[phase=done]` contract, and ABI
between dispatcher and GPU entrypoint that a full GPU smoke would. A
GPU-bound phase MISSING the `Carve-out (GPU-bound)` sub-heading IS still
a `smoke-run-missing` FAIL: the workflow accepts the substitute coverage
only when it is labeled at report time (the label is what lets you
distinguish a documented carve-out from a silently-skipped smoke). A
carve-out sub-section that is labeled but omits any of the three items
or omits the constraint sentence is ALSO a FAIL — incomplete coverage
re-introduces the bugs the gate exists to catch. Incident #514 r2:
unlabeled "(signature smoke)" notation FAILed `smoke-run-missing`; the
label is what distinguishes a documented carve-out from a genuinely
missing smoke.

**Deferred `scripts.*` imports must be proven in SCRIPT MODE, not `-c` mode.**
If the diff adds a deferred `from scripts.X import ...` inside a src-layout
driver (`src/explore_persona_space/experiments/**`), check the smoke evidence
(or the carve-out's CPU-runnable smoke) shows that import executing in SCRIPT
MODE (`python /abs/path/driver.py`) from a NON-repo cwd — a `-c`-mode import
check false-passes (cwd on `sys.path`) while script mode crashes pod-side
(`sys.path[0]` = the script's dir). An unguarded deferred `scripts.*` import
(no `_ensure_repo_root_on_syspath()`-style guard) is a substantive finding at
normal severity — NOT a `smoke-run-missing` blocker. See
`.claude/rules/gotchas.md` (script-mode entry); incident #823, commit
`14234c9112`.

**Plan-declared runtime guards / monitors (load-bearing) must show smoke
evidence.** When the approved plan declares a runtime guard / monitor /
trajectory logger as a load-bearing mitigation (a saturation guard,
`MarkerBandStopCallback`, per-step log-prob probes, an auto-fired
secondary DV, per-source WandB run separation), check the `## Smoke run`
section shows that guard's telemetry actually functioned during the
smoke: a probe value was logged, the guard branch was exercised or its
precondition assert ran, per-source WandB run names are distinct. Missing
evidence for a plan-declared load-bearing guard is a FAIL with blocker
tag `smoke-run-missing` for that phase (same tag, no new schema), UNLESS
the implementer's `(d) Needs human eyeball` section explicitly calls out
why the guard cannot be demonstrated at smoke scale AND names the closest
demonstrable proxy — then it is at most a `CONCERNS` (verify the stated
reason is plausible). Rationale: checking "phases ran" without checking
"declared guards emit evidence" lets a silent monitor ship — incident
#480: the plan's WandB trajectory monitor + KL auto-fire never functioned
(5 of 6 source runs reused one WandB run name, per-cell trajectories were
never logged, zero saturation markers fired), saturation was caught only
at eval time, and the experiment needed a full band-stopped retrain.

**Resume-matrix + real-production-out-root-unit smoke coverage.** When
the diff exposes ANY resumable re-entry branch — a resume predicate
(done-file skip, sidecar-based terminal-verdict skip), a
`--from-phase <name>` / `--resume` flag, a salvage/topup leg, a
recorded-verdict re-read — check the `## Smoke run` section shows each
such leg was exercised against a synthesized partial state (interrupt
or seed the partial artifacts the leg re-reads, re-enter with the
resume flag or the recorded-sidecar in place, exit 0 + designed
disposition). AND check the smoke ran at least ONE real corpus/fit unit
end-to-end at production shape writing to the PRODUCTION out-root
(`eval_results/issue_<N>/...`), not a `/tmp/issue-<N>-smoke/` twin — the
seams that fire only against the canonical path are `mkdir` of an
out-root parent whose directory tree the first cell creates,
registry-coupled metadata lookups keyed on the caller's own cell/arm
ids (`_pfx_fit_core`-style `arm_method(arm_id)` misses), path
predicates that gate on the canonical out-root prefix. Both requirements
are attested per-leg / per-unit in the smoke-architecture-check
marker's `resume-matrix:` and `production-outroot-unit:` sub-blocks
(REAL / FALLBACK <reason> / N/A vocabulary — same shape as
`per-arm-resolution:`). Missing evidence for either — a resume/topup/
salvage leg the diff exposes with no smoke pass or FALLBACK
declaration, OR a smoke that writes only to a `/tmp/issue-<N>-smoke/`
twin with no production-outroot unit or FALLBACK — is a FAIL with
blocker tag `smoke-run-missing` for that phase (same tag, no new
schema), UNLESS the implementer's `(d) Needs human eyeball` section
explicitly explains why the leg/unit cannot be exercised at smoke
scale AND names the closest demonstrable proxy — then it is at most a
`CONCERNS` (verify the stated reason is plausible; a FALLBACK
declaration in the marker with a plausible reason IS such a
call-out). Rationale (#1947 P0 launches 4-5, #1315 r6, #1112 r6,
#1947 P4/P5 round 2): recorded terminal-verdict re-entry crashes,
salvage input overwrites, resumed-process side-effect loss,
partial-artifact resume, and reused-`_pfx_fit_core` registry-lookup
misses — five distinct crash classes that fire only on the RE-ENTRY
leg / production out-root the smoke never exercises. Mirror implementer
rule: `.claude/agents/experiment-implementer.md`
§ Additional smoke-contract requirements (Resume-matrix smoke + Real
production out-root unit bullets).

**Crash-fix rounds must show a confirmed fix-engaged signal.** When the
round under review was dispatched to fix a posted `epm:failure` (a
crash-fix round — the report carries a `### Response to code-review` or
the brief named a failure), check the `## Smoke run` section contains a
`### fix-engaged signal` sub-section that (a) names the exact signal the
fix's new code path emits, (b) pastes the matched line from a same-pod /
smoke-slice re-run confirming the signal appeared, (c) ties the
signal to the specific branch the fix added, (d) declares the fix
commit's FULL SHA(s), and (e) declares the stale-run artifact
disposition (`quarantine` / `retain — <reason>` / gated `wipe` /
`fresh-output-path / --no-resume` / explicit `N/A — <reason>`) —
elements 4/5 of `.claude/rules/crash-fix-rounds.md`. Rounds dispatched
before elements 4/5 landed are reviewed under the 3-element contract.
Missing or unconfirmed (no
pasted matched line) is a FAIL with blocker tag `substantive` (NOT
`smoke-run-missing`) — a fix-engaged-signal miss is a substantive
judgment about whether the fix actually engaged, so it must sit OUTSIDE
the `mechanical_contract_only_strip` set `{marker-shape,
smoke-run-missing}` and cannot be downgraded by the Step 5c-bis strip
(which inspects only the ordinary `## Smoke run` shape, never this
sub-section). UNLESS the implementer's `(d) Needs human eyeball`
explicitly explains why the signal cannot be shown at smoke scale AND
names the closest demonstrable proxy — then it is at most CONCERNS.
Rationale: a fix re-run on a fresh pod whose code path was never proven
to engage is the #664 banned regression (a chunk-500 fix relaunched when
the absent `[vllm-chunk]` log meant the hang preceded the first chunk).
Mirror implementer rule: `.claude/rules/crash-fix-rounds.md` § "Crash-fix rounds:
declare the fix-engaged signal".

**Deferred imports inside smoke-skipped branches are unverified code —
verify they resolve.** When any phase's smoke command carries a skip-flag
that fences off a code branch (`--dry-run`, `--skip-upload`, `--skip-eval`,
or equivalent), that branch never executed during the smoke, so any lazy
`import` / `from ... import` inside it has never run. Grep the diff's
scripts for in-function / in-branch imports:

```bash
grep -nE "^\s+(from [A-Za-z_0-9.]+ import|import [A-Za-z_0-9.]+)" \
  <each script in the diff>
```

For each hit inside a branch the smoke's skip-flags fenced off, require
ONE of:

- (a) **execution evidence** in the `## Smoke run` section — a
  `--verify-imports` run (the AST-walk pattern from
  `scripts/issue_606/i606_dispatch.py`; see `.claude/rules/gotchas.md`
  "Lazy imports inside smoke-skipped branches") or a smoke invocation
  without the fencing flag;
- (b) **module-top hoisting** — the import was moved to module top, so any
  phase's exit-0 smoke already proves it executes;
- (c) **your own static verification** — grep the import's TARGET module
  for each imported symbol's definition and quote `file.py:LINE` in the
  verdict. Watch the porting trap: a private `_underscore` helper is often
  file-local to the SOURCE script the code was ported from and absent from
  the import path the diff assumes.

A deferred import whose symbol you CANNOT find at the import target is a
Critical SUBSTANTIVE finding (blocker tag `substantive`, NOT
`smoke-run-missing` — the orchestrator's Step 5c-bis strip cannot verify a
symbol exists in source code from the marker alone, so this finding must
never be stripped as mechanical-contract): the ImportError fires on the
pod AFTER the expensive phases. A deferred import that resolves but lacks
(a)/(b) evidence is at most a CONCERNS bullet.

**Fenced CALLS must also BIND — resolution alone is not enough.** For
each call to an imported helper inside a fenced branch (deferred OR
module-top import — option (b) hoisting does NOT discharge this),
verify the call site's arg shape binds the helper's live signature:
`inspect.signature(fn).bind(...)` per the gotchas.md bind recipe, or a
static signature read quoted as `file.py:LINE`. A non-binding fenced
call is the SAME Critical SUBSTANTIVE class — the TypeError fires on
the pod after the expensive phases (#1332 r1: two fenced
`verify_repo_paths_uploaded` calls missing the `api` positional +
REQUIRED kw-only `path_in_repo`; caught only by an ad-hoc reviewer
signature sweep).

The mirror implementer rule
is `experiment-implementer.md` § After implementation step 2 ("Deferred
imports count"). Incident #606 (2026-06-11): review rounds 1-2 PASSed a
dispatcher whose upload branch lazily imported the nonexistent
`_retry_transient` from `orchestrate.hub`; every smoke carried
`--dry-run` / `--skip-upload`, and the ImportError fired on the GCP
workload at p5_upload after training + stage-A judging were already spent.

**If every phase IS present with a command, exit code 0, and an artifact
digest, but a digest is terse, omits the row count, or you would have
formatted it differently — that is at most a `CONCERNS`, NEVER a standalone
FAIL.** Each phase demonstrably ran and wrote a real artifact, so the
GPU-protection purpose of this gate is satisfied. Note the cosmetic gap
under "Style / Consistency" and PROCEED to review the diff.

Code-only tasks (`type:infra` / `type:batch` / `type:analysis` /
`type:survey`) are EXEMPT from this gate — they keep the test-verdict gate
(`/issue` Step 9c) and the Step 4 test run below.

**Smoke output-path hygiene ("Smoke outputs never overwrite committed artifacts").**
Two checks:

- **Clobber evidence is SUBSTANTIVE, never mechanical.** If the diff (or
  the worktree you review in) replaces an existing committed
  `eval_results/` / `figures/` artifact with a smoke-scale version at its
  canonical path (fewer layers / cells / rows), raise a Critical finding
  tagged `substantive` — NOT `smoke-run-missing` — so the Step 5c-bis
  mechanical strip can never remove it (#722 shipped a smoke-scale hero
  figure and truncated committed 28-layer JSONs).
- **A missing disposition line is CONCERNS, not FAIL.** A `### <phase>`
  smoke sub-section whose command writes under `eval_results/` /
  `figures/` but states no output-path disposition (scratch-dir redirect,
  or restore-after-smoke + an empty
  `git status --porcelain -- eval_results/ figures/`) is a Minor — unless
  the clobber itself is visible (first bullet).

**Any verification command YOU run follows the same rule.** If you rerun
a test or smoke that regenerates files under `eval_results/` /
`figures/`, afterwards run
`git status --porcelain -- eval_results/ figures/`, restore the committed
artifacts YOUR OWN command modified
(`git -C <tree-root> checkout -- <paths>` — the `-C` both names the tree
deliberately and passes the repo-root guard (#897) — never a blanket
revert) and delete the untracked outputs it
left; leaving them dirty plants the clobber for the next explicit-path
commit (#722 instance 2 was exactly this). Binds BOTH ensemble
reviewers (rides into the Codex twin via the inlined Step 0.6 rubric).

## Step 0.65 detail — raw-completions upload wiring and plan-glob parity

A pod-side dispatcher that writes per-cell completion files to disk under
`eval_results/issue_<N>/` (`raw_completions/*.json`, `raw_generations/*.json`,
or any equivalent per-cell completion JSON the eval loop persists) MUST
upload them from its normal exit path BEFORE the `[phase=done]` log line +
final sentinel write, via ANY of the three accepted call shapes:

1. `explore_persona_space.orchestrate.hub.upload_raw_completions_to_data_repo()`
   — the canonical helper;
2. an explicit per-file `hub._upload(...)` loop with `repo_type="dataset"`
   and `path_in_repo=f"issue<N>_<slug>/raw_completions/<rel>"`;
3. a batched `HfApi.create_commit(repo_type="dataset")` whose
   `CommitOperationAdd` ops target the canonical
   `issue<N>_<slug>/raw_completions/{condition}_seed{S}.json` paths, with
   post-commit Hub-side verification (e.g. per-prefix counts via scoped
   `list_repo_tree(path_in_repo=<prefix>)` — bare data-repo
   `list_repo_files` times out, gotchas.md) before `[phase=done]`. Under the HF Hub ~256-commits/hour repo
   throttle (#591) the batched shape is PREFERABLE to the per-file loop
   for large file counts — one commit instead of N. Do NOT FAIL an
   implementation for batching its uploads (incident #606: a functionally
   stronger batched `create_commit` + count verification was FAILed on the
   call-shape grep alone; the reconciler overturned it).

The contract is the SUBSTANCE of the CLAUDE.md Upload Policy — per-cell
completions land on the HF data repo under the canonical prefix before the
dispatcher reports done — not any one call-shape string; the
upload-verifier at Step 8 is the safety net, NOT the only line of defense
— if a future verifier change ever trusted the `epm:results` sentinel
without re-enumerating, the unuploaded files would die on pod termination.

Before reviewing the diff, grep the dispatcher(s) in the diff for the
upload call:

```bash
grep -nE "upload_raw_completions_to_data_repo|hub\._upload\(.*raw_completions|create_commit" \
  <each pod-side dispatcher in the diff>
```

(A bare `create_commit` match is necessary but not sufficient — confirm by
reading the surrounding code that it targets the dataset repo with the
canonical `issue<N>_<slug>/raw_completions/...` `path_in_repo` ops; you
read the diff anyway per Step 0.7.)

If a dispatcher writes raw completions to disk (`grep -nE
"raw_completions\.json|raw_generations" <dispatcher>` returns matches) AND
the upload-call grep returns zero matches, return verdict FAIL with a
single `Critical` issue tagged `raw-completions-upload-missing` (naming
the dispatcher file in the body), AND still read the diff and report
substantive findings in the same pass (do not short-circuit — see
Step 0.7):

> Dispatcher `scripts/<dispatcher>.py` writes raw completions to
> `eval_results/issue_<N>/...` but wires none of the three accepted
> upload shapes (`upload_raw_completions_to_data_repo()` / `hub._upload`
> loop / batched `create_commit`). Re-post `v<n+1>` with one wired into
> the normal exit path (after eval, before `[phase=done]` + sentinel).

The mirror implementer rule is `experiment-implementer.md` § After
implementation step 7. Incident #528 (2026-06-09): a pod-side dispatcher
wrote 160 raw-completion JSONs and never invoked the helper — caught
manually, indistinguishable from silent loss had the verifier trusted
the sentinel.

If the dispatcher writes NO raw completions (a pure metrics-only eval,
an analysis-only dispatcher, a training-only entrypoint), this gate is
N/A; record that one-line conclusion in the verdict body and proceed.

The `raw-completions-upload-missing` blocker tag is a SUBSTANTIVE code-
absence finding (a missing function call in the dispatcher), NOT a
mechanical/presentation gate, so it is NOT stripped by SKILL.md
Step 5c-bis ("Mechanical-contract-only FAIL strip") even though it
fires before the diff-read steps. The strip list there is
limited to `marker-shape` (Step 0.5), `smoke-run-missing` (Step 0.6),
and `git-provenance` (Step 0.9) — the three tags where the orchestrator
can mechanically verify the finding (the artifact IS present in the
marker, or a read-only `git` probe confirms the flagged state is not
introduced by the round's diff); there is no orchestrator-side check
that can validate a function call exists in source code without reading
the diff, so `raw-completions-upload-missing` stands as a real Critical
blocker until the implementer wires the call.

**Plan-glob vs uploader-eligibility parity sub-check (#825; same step,
`type:experiment` only).** Step 0.65 above checks an upload call EXISTS;
this sub-check verifies what the call makes ELIGIBLE. When the plan
declares artifact globs — §6.5 `primary_deliverable:` rows, §10 per-stage
output destinations (`raw_completions/<stage>/`, `analysis_tensors/`,
eval JSONs) — and the diff wires any upload through an eligibility filter
(`upload_folder(allow_patterns=...)` / `ignore_patterns=...`, a custom
glob/`p.match(pat)` enumeration, an extension allowlist feeding
`create_commit` ops) — OR the diff adds a WRITER of a new persisted file
kind for a plan-declared class while the round's uploader sits outside
the diff (the #825 shape is exactly a new `.jsonl` kind beside existing
`.npy` payloads; locate the round's uploader filter even when out-of-diff)
— DIFF the two sets: every plan-declared artifact
class must be matched by at least one upload path's filter, or be uploaded
by a separate wired call, or be covered by a plan §10
`discarded_artifacts:` entry (the only declared-not-uploaded exemption).
Locate the filters in the diff first:

    grep -nE "allow_patterns|ignore_patterns|\.match\(|\.suffix" \
      <each upload helper / dispatcher in the diff>

A declared class NO filter makes eligible is a Critical finding tagged
`substantive` — naming the plan row/glob AND the filter line that excludes
it. Like `raw-completions-upload-missing`, this is a substantive
code-absence finding, NOT a mechanical/presentation gate, so it is never
stripped by SKILL.md Step 5c-bis (no orchestrator-side check can validate
filter coverage without reading the diff); it deliberately reuses the
existing `substantive` tag rather than minting a new one. Incident #825
(upload-verification v14, 2026-07-16): the round's tensor uploader allowed
`**/*.npy` + `**/*.json` only, so 404 plan-§6.5-declared `row_index*.jsonl`
files (48.9 MB) were never upload-eligible — caught at the LAST gate with
the GCE instance luckily still alive. Known residual (note it, don't
FAIL on it): a filter that names the class inside a `--mode` branch the
dispatcher never invokes passes this text-level read while the class
never uploads — invocation coverage stays the upload-verifier's catch.
N/A cases (record the one-line
conclusion and proceed): the plan declares no artifact globs; the diff
neither touches an upload call nor adds a writer of a new persisted file
kind; the uploader applies no eligibility filter AND its upload root(s)
span every declared class (uploads the whole tree). Full producer-side rule:
`.claude/rules/upload-policy.md` § "Uploader eligibility filters must
cover every plan-declared artifact class".

## Step 0.67 detail — compute-shape-vs-dispatcher and work-conserving schedule

A plan whose §9 declares a **data-parallel / sharded** compute shape (N-GPU
data parallelism, per-GPU workers, context/cell sharding) but whose dispatcher
scripts expose NO way to actually run that shape silently ships to a
bigger-than-needed pod: the experimenter provisions the declared multi-GPU pod
(`sweep-8g-h100`, `ft-7b`, an 8×H100/A100), the dispatcher runs on ONE GPU,
and the other N−1 GPUs sit at 0% util billing until a human notices — the
#664-class spend-leak. This gate is the code-review-time sibling of SKILL.md
Step 6d.0 (smoke/sweep architecture parity): Step 6d.0 fires at DISPATCH
(after the pod is already provisioned); this fires at REVIEW, before any pod
exists.

**Trigger — does the plan §9 declare a DP/sharded shape?** Grep the approved
plan (the CANONICAL plan on main — Step 1 already reads it; do NOT trust a
possibly-stale worktree copy) for a data-parallel / sharding declaration:

```bash
# Plan §9 (Resources & Parallelism) — prose declaration:
grep -nEi 'data.?parallel|\bDP\b|[0-9]+ *(single-GPU|per-GPU) *workers?|shard(ing)? (contexts|cells|prompts)|CUDA_VISIBLE_DEVICES *workers?|checkpoint-per-shard' <plan.md>
# Plan §9 per-component compute-projection table — the `parallelism` column
# (a mandatory field for kind:experiment plans, planner.md §9):
grep -nEi '\|\s*[0-9]+ *[x×] *(H100|H200|A100|L4) *DP|ZeRO-3|FSDP|sharded' <plan.md>
```

**Not a trigger — TP-only or single-GPU is fine.** A plan that declares
`TP=N` / `tensor-parallel` (a single process spanning N GPUs — the standard
vLLM/`LLM(tensor_parallel_size=N)` path, which needs no per-shard dispatcher
flag) or `1×`/single-GPU compute does NOT trigger this gate. Tensor
parallelism is exposed by one launcher argument the eval/generation library
already threads, so it does not slip the way data parallelism does. If §9
declares ONLY TP or single-GPU, record `Step 0.67: N/A — plan declares
TP-only / single-GPU, no data-parallel shape` in the verdict body and proceed.
The N/A covers the EXPOSURE CONTRACT only — the work-conserving schedule
sub-check below still applies whenever the diff itself schedules >1
independent cell on a multi-GPU pod/provision.

**If the plan DOES declare a DP/sharded shape**, verify the dispatcher
script(s) in the diff actually expose it. Grep each pod-side dispatcher in the
diff:

```bash
grep -nE '(--shard-id|--num-shards|--num-workers|--world-size)|torch\.distributed|torch\.multiprocessing|mp\.spawn|accelerate (launch|\.)|subprocess\.(Popen|run).*--gpu-id|CUDA_VISIBLE_DEVICES' <each dispatcher in the diff>
```

Credit the DP shape as EXPOSED when at least ONE of these holds (confirm by
READING the matched code, not the grep hit alone — you read the diff anyway per
Step 0.7):

- **(a) External shard flags:** the dispatcher accepts a `--shard-id N
  --num-shards K` (or equivalent `--num-workers`/`--world-size`) flag pair,
  so a launcher can fan out one process per GPU each processing a shard.
- **(b) Internal DP fan-out:** the dispatcher itself spawns workers via
  `torch.distributed`(`.run`/`.init_process_group`),
  `torch.multiprocessing.spawn`/`mp.spawn`, `accelerate launch`, or an explicit
  per-GPU `subprocess.Popen`/`run` loop over `CUDA_VISIBLE_DEVICES`.
- **(c) External one-process-per-GPU launcher / documented fan-out:** a
  `scripts/issue<N>*_run.sh` / launcher committed in the diff (or named in the
  approved plan's launch section) runs ONE dispatch process per GPU with
  distinct `--gpu-id` values each over a distinct shard, OR the implementer's
  `## Smoke run` / report explicitly documents that the experimenter fans the
  single-GPU dispatcher out per-GPU at launch. A dispatcher that accepts only a
  single-GPU selector (whatever the flag is named — `--gpu-id N`, `--device N`,
  etc.; no shard split, no per-GPU launcher, no fan-out documentation) does NOT
  satisfy (c) — a single-GPU-only entrypoint run on an 8-GPU pod uses one GPU.

**Verdict routing:**

- **Plan declares DP AND at least one of (a)/(b)/(c) is present** → PASS this
  lens; note which shape satisfied it in the verdict body.
- **Plan declares DP AND none of (a)/(b)/(c) is present** → return verdict
  FAIL with a single `Critical` issue tagged `compute-shape-mismatch` (naming
  the dispatcher file + the plan's declared shape in the body), AND still read
  the diff and report substantive findings in the same pass (do not
  short-circuit — see Step 0.7):

  > `epm:experiment-implementation v<n>`'s plan §9 declares a data-parallel /
  > sharded compute shape (<quote the declared shape, e.g. "8×H100 DP, 8
  > single-GPU CUDA_VISIBLE_DEVICES workers sharding contexts">) but the
  > dispatcher `scripts/<dispatcher>.py` accepts only <the observed
  > single-GPU flag, e.g. `--gpu-id N` / `--device N`> (single GPU) and
  > exposes no `--shard-id`/`--num-shards` flag pair, no internal
  > `torch.distributed`/`torch.multiprocessing.spawn`/`accelerate`/per-GPU
  > `subprocess` fan-out, and no external one-process-per-GPU launcher. The
  > declared multi-GPU pod would leave N−1 GPUs at 0% util billing (the #664
  > spend-leak). Re-post `v<n+1>` with EITHER the DP wiring added to the
  > dispatcher (shard flags / internal DP / per-GPU launcher) OR the plan §9
  > compute shape corrected to the single-GPU intent the dispatcher actually
  > supports (a `--intent lora-7b`-class descope; update the per-component
  > compute-projection table's `parallelism` column to match).

- **Plan declares DP AND the dispatcher's DP support is plausible but you
  cannot confirm it from the diff** (e.g. the fan-out is claimed to live in an
  external launcher not in the diff, or the shard-split lives in an imported
  helper you cannot fully trace) → do NOT FAIL: record a `CONCERNS` bullet
  under "Issues Found" naming the unverified fan-out site and request the
  implementer point to the exact per-GPU dispatch line in `(c)` (report or
  launcher). This mirrors Step 0.65's "necessary-but-not-sufficient grep"
  caution — a plausible-but-unconfirmed shape is a CONCERNS, not a FAIL. So
  the concern actually BINDS through the Step 5c-ter dispatch gate rather than
  staying a prose-only bullet, PERSIST it via `task.py raise-concern <N>
  --concern-id compute-shape-unverified-fanout --severity CONCERN --summary
  '<≤200c: plan §9 declares DP; dispatcher fan-out unverified from the diff>'
  --by code-reviewer --round <n>` (per Step 0.8 / Rule 11 — verdict-body
  bullets that are NOT persisted remain opportunistic and do not reach the
  dispatch gate).

Either corrective closes the mismatch — the fix does NOT have to be "add DP".
Descoping the plan's declared shape to the intent the dispatcher supports is
an equally valid resolution (and is exactly how #779 round 7 resolved it: the
plan was descoped `sweep-8g-h100` → `lora-7b`, science unchanged).

The `compute-shape-mismatch` blocker tag is a SUBSTANTIVE finding (a real
mismatch between the plan's compute contract and the dispatcher's actual
capability), NOT a mechanical/presentation gate, so it is **NOT stripped** by
SKILL.md Step 5c-bis ("Mechanical-contract-only FAIL strip"). The strip list
there is limited to `marker-shape`, `smoke-run-missing`, and `git-provenance`
— the three tags the orchestrator can mechanically verify from the marker or a
git probe; there is no orchestrator-side check that can validate a
dispatcher's DP capability against the plan without reading the diff, so
`compute-shape-mismatch` stands as a real Critical blocker until the
implementer wires the DP path or descopes the plan. (Same family as
`raw-completions-upload-missing` / `cached-artifact-coverage-unverified`.)

If the plan declares no DP/sharded shape (TP-only, single-GPU, or a
CPU-only/analysis task), this gate is N/A; record that one-line conclusion in
the verdict body and proceed. The N/A covers the EXPOSURE CONTRACT only — the
work-conserving schedule sub-check below still applies whenever the diff
itself schedules >1 independent cell on a multi-GPU pod/provision.

Incident: task #779 round 6 (2026-07-01) — the approved plan §9 declared "one
8×H100 pod, data-parallel (8 single-GPU CUDA_VISIBLE_DEVICES workers)" and the
per-component compute-projection table's `parallelism` column read `8× H100
DP` across three phases, but `scripts/issue779_{extract_rb,collect,stage1}.py`
accepted only `--gpu-id N` with no shard split and no DP entrypoint. Round-6
code-review PASSed (Claude + Codex + reconciler); the `sweep-8g-h100` (8×H100)
pod was provisioned and the first util reading showed all 8 GPUs at 0%. Round 7
descoped to `lora-7b` (1×H100). No reviewer checked plan-declared shape ↔
dispatcher-exposed shape.

**Work-conserving schedule sub-check (diff-read; applies whenever the diff
schedules >1 independent cell on a multi-GPU pod/provision — reached via the
§9 trigger above OR by finding the scheduling code in the diff; the exposure
gate's N/A does NOT close it).** Exposure is necessary but not sufficient: a
dispatcher can satisfy (a)/(b)/(c) and still idle most of the pod through a
non-work-conserving SCHEDULE. Whenever the diff schedules multiple
independent cells for a run whose plan/provision names >1 GPU or worker —
including a plain serial `for cell in cells:` loop on a multi-GPU pod (a
degenerate single-worker schedule) — READ the schedule loop and verify it is
work-conserving: whenever a worker/GPU is idle and a pending cell with
satisfied dependencies exists, it dispatches. Flag as **Major** (tag
`substantive` when it drives a FAIL — NOT `compute-shape-mismatch`, which
stays reserved for the exposure contract above) any strict wave/stage barrier
that drains ALL in-flight work before starting independent cells (`for wave
in waves: pool.map(...); pool.join()`, a fresh joined pool per stage, a
per-lane `Popen` + wait-all loop, or a barrier between shards with no data
dependency) AND any degenerate serial schedule of independent cells on a
multi-GPU provision. A barrier or reduced width is acceptable ONLY for a
justification the plan states: a genuine cross-cell dependency (cell B
consumes cell A's output) OR a named resource/capacity constraint (HBM
footprint, per-pod disk quota, model residency) that makes wider concurrent
dispatch infeasible — name whichever you credit in the verdict. Note a
GPU-width cap justifies concurrency WIDTH, not a drain barrier: a shared
queue with `wave_size` persistent workers satisfies a width contract AND
work-conservation. Suggest the work-conserving shapes: one shared task queue
with N persistent workers (`Pool.imap_unordered` over ALL cells, one pool for
the whole run), or dependency-keyed dispatch (launch each cell the moment its
inputs land). Unlike the exposure contract, this is a Step-2-family diff-read
finding housed here for discoverability — the Step 0.7 "pre-diff contract
check" framing applies to the exposure check above, not to this sub-check,
and a plausible-but-unconfirmed schedule is a CONCERNS, not a FAIL.

Incident #813 (2026-07-01): the dispatcher ran two STRICTLY SEQUENTIAL waves
— wave 2 (~55% of remaining rows) would not start until wave 1 fully drained
— leaving GPUs 1/2/4/7 idle 6.7h on a billing 8×H100 pod (true remaining
~38-52h vs the projected 18-20h); review PASSed because the shape was exposed
and nobody read the schedule. Same family: #778 phase-3 looped 25 models × 3
traits one-at-a-time on an 8×H100 pod (~4-5h at 1/8 util) — a degenerate
single-worker schedule on a multi-GPU pod, in scope of this sub-check even
when the plan's §9 declared no DP shape.

## Step 0.68 detail — named-helper adherence, hollow gates, hub-call scoping

For each helper the task body's reuse map or the approved plan (§4 pseudocode
/ §10 Reproducibility Card / §11) names by `module::fn` or file-path
reference as THE implementation for a step — especially a fast / batched /
verified-equivalent twin (e.g. `_ridge_fit_predict_fast`,
`vectorized_mlp_skill.py`, batched `perm_null_draws`) — grep the diff AND the
final driver the dispatcher invokes for that helper's import and call site.
Substituting a slower sibling (the serial original, a fresh reimplementation
of the same math) without a plan-documented substitution note is a
substantive finding at Major severity (blocker tag `substantive`): the named
twin usually carries a validated equivalence gate and a measured cost
profile, and dropping it silently is how #823 turned a ~125 s/call fast-twin
phase into ~3780 serial full-SVD fits (12-20 h) that round-1 plan-adherence
review blessed ("ridge, not MLP" ✓) — no check compared the plan's import
against the body-named twin. PASS requires the named helper imported+called,
OR a documented substitution. Record `Step 0.68: N/A — no ::fn-level helper
named` when neither body nor plan names any function/file-path-level helper
(module-level "reuse #M's pipeline" claims are the consistency-checker's
plan-time check, not this one).

**Hollow-verification-gate sub-check.** Whenever the diff carries or invokes
a verification / equivalence / vectorization gate (a `--verify-X` flag, an
`assert_*_equivalence` / `assert_matches_reference`-style self-check), or
modifies the dispatch path of a stage guarded by such a gate, confirm the
GATED function is the one the entrypoint actually dispatches on the live
path: trace flag → gate call → gated callee, then grep the dispatch path for
that same callee — object identity (`gated_fn is dispatched_fn`) OR
`__module__` + `__qualname__` match (a bare `__qualname__` match
false-verifies a same-name wrong-module sibling); for grep evidence, quote
both the import source and the call site proving the same object is
dispatched. A gate asserting on an unused sibling is a hollow gate — a
substantive finding at Major severity, blocker tag `hollow-verification-gate`
(SUBSTANTIVE, never `marker-shape` / `smoke-run-missing` / `git-provenance`,
never stripped by Step 5c-bis): its green PASS launders an unverified hot
loop as verified (incident #779: `--verify-vectorized` gated the unused
`vectorized_mlp_skill.assert_matches_reference()` while the dispatched ridge
hot loop — ~17k fits, 18-20h — had zero verification coverage; the launch
note claimed "vectorized" and rounds 6/7 PASSed). This sub-check fires
whenever such a gate is present — even when the named-helper trigger above is
otherwise N/A, and for any diff type, not only `type:experiment` (same
pattern as Step 0.67's work-conserving sub-check: the parent gate's N/A does
not close it). Record the gate→dispatch trace (gated fn, dispatched fn,
evidence `file.py:LINE`) or `hollow-gate sub-check: N/A — no verification
gate in diff`.
Sibling: Step 3.8 covers the BODY half of this family — a test that stubs the
production function itself launders a never-executed body exactly as a hollow
gate launders an unverified hot loop.

**Hub-call-scoping sub-check (any diff type).** When the diff INTRODUCES or
modifies a Hub verify / staging / existence-probe call against the ~1M-file
data repo, confirm it is prefix-scoped (`list_repo_tree(path_in_repo=<prefix>)`
for subtree listings, `file_exists` for single-path probes) with a bounded
outer retry on a first-page 429/5xx — an unscoped full-tree `list_repo_files`
/ `snapshot_download` there is a substantive Major (recipe:
`.claude/rules/gotchas.md` #833). Plan-time twin for REUSED, diff-untouched
helpers: `.claude/rules/artifact-reuse.md` check (i) leg (3) (#810: a reused
verify crawl wedged a live A100 run in 429 storms). Record
`Hub-call-scoping sub-check: N/A — no data-repo Hub calls in diff` when
absent. Full listings of the SMALL model repo are fine — data repo only.

## Step 0.69 detail — phase idempotency and inter-phase contract

A phased dispatcher that lacks skip-if-output-exists is the recurring paid-API
spend leak: a downstream vLLM crash restarts the pipeline from the top, re-running
a paid-Anthropic / paid-OpenAI phase whose artifacts already sit on HF. And a
consumer phase that asserts its input JSONL contract AFTER model initialization
turns a schema mismatch into a wasted GPU cycle (the #1689 shape: 79,800 rows
rendered, vLLM initialized, then `ValueError: The decoder prompt cannot be
empty` — 33% of rows resolved to `messages: [...]` with no `prompt_text` — after
the pod was billing). Both are code-review-time catchable from the dispatcher
diff alone. This gate is the review-side sibling of `.claude/rules/code-style.md`
line 53 (checkpoint-per-phase, ADVISORY-prose): the rule exists, and this gate
enforces it on dispatcher diffs.

**Trigger — does the diff add/modify a multi-phase dispatcher?** Grep the diff
for a phase-dispatch shape:

```bash
# Bash entrypoint dispatchers (issue<N>_dispatch.sh, one line per phase):
grep -nE '^phase_[a-z0-9_]+\s*\(\)|^case .* in$|_run_phase [a-z0-9_]+' <each dispatcher .sh in diff>
# Python phase-loop dispatchers (`for phase in PHASES`, `if args.phase == 'a'`):
grep -nE 'def phase_[a-z0-9_]+|PHASES *= *\[|args\.phase|--phase' <each dispatcher .py in diff>
```

Multi-phase = >1 phase name. Not a trigger — a single-entrypoint script or a
one-phase run: record `Step 0.69: N/A — diff carries no multi-phase dispatcher`
and proceed.

**Sub-check (1) — phase-level skip-if-output-exists.** For each phase, verify
ONE of the two patterns holds by reading the phase entry body:

- **(a) Sentinel/output-artifact skip.** The phase body checks a declared
  completion-sentinel or primary-output path at entry
  (`[ -e "$OUT/phaseA_done" ] && { echo "[phaseA] skip — sentinel exists"; return; }`,
  or a Python `if OUT.exists() and not args.force: return`). Bare file existence
  is ACCEPTABLE only when the phase writes its output atomically-then-renames
  (`os.replace` or `mv` of a tmp file) — otherwise the sentinel discipline of
  CLAUDE.md § Monitoring re-run discipline applies (never key "done" on a
  half-written file). Prefer a completion sentinel over a bare glob.
- **(b) First-class `--force` (or equivalently-named) flag.** The dispatcher
  accepts `--force` / `--rerun` / `--no-skip` / `--overwrite` (or the same as
  an env var like `FORCE_PHASE=1`) whose DEFAULT is OFF and whose value is
  threaded through the phase entry — so a deliberate rerun stays first-class.
  A phase that ALWAYS re-runs (no sentinel check AND no force flag) FAILs
  this sub-check.

Waiver form (mirrors `# CVD_PIN_EXEMPT: <reason>` from
`.claude/rules/gotchas.md`): a phase legitimately non-idempotent by nature (a
stochastic sampling phase whose output is per-run, an eval whose "done" state
is a WandB run id, etc.) carries `# PHASE_IDEMPOTENCY_EXEMPT: <reason ≥ 20 chars>`
on the phase entry's signature line. A waiver ≥ 20 chars is credited PASS.

**Verdict routing — sub-check (1):**

- Phase ships with (a) or (b) or a valid waiver → PASS this sub-check.
- **Phase makes paid API calls** (Anthropic/OpenAI/HF inference — grep the
  phase transitively for `anthropic.Anthropic`, `openai.OpenAI`,
  `api_dispatch.dispatch_judge_items` / `batch_judge` / `judge_completions_batch`,
  `openai.chat.completions.create`, or the plan §9 marks the phase `paid`) OR
  **holds a GPU** (grep for `vllm`, `AutoModel.from_pretrained`, `train_lora`,
  `LLM(`, `torch.cuda`, `accelerate launch`) AND ships without (a)/(b)/waiver →
  return verdict FAIL with a single `Critical` issue tagged `phase-not-idempotent`
  (SUBSTANTIVE — never stripped by Step 5c-bis), AND still read the diff and
  report substantive findings in the same pass (do not short-circuit — see
  Step 0.7):

  > `epm:experiment-implementation v<n>`'s dispatcher `scripts/<file>` phase
  > `<name>` makes paid API calls (or holds a GPU) but has no skip-if-output-exists
  > guard and accepts no `--force`/`--rerun` flag. A downstream crash re-runs it
  > from scratch, re-spending its API budget (or re-holding its GPU) each cycle
  > — the #1689 shape (4× re-runs of the same 3-condition × 3800-row Sonnet
  > phase across crash-fix relaunches). Re-post `v<n+1>` adding EITHER a
  > completion-sentinel check at phase entry (`[ -e "$SENTINEL" ] && return`)
  > OR a first-class `--force`-family flag (defaulting OFF), threaded through
  > the phase entry. Prefer a completion sentinel over bare file existence
  > (CLAUDE.md § Monitoring re-run discipline). A legitimately
  > non-idempotent phase carries `# PHASE_IDEMPOTENCY_EXEMPT: <reason ≥ 20c>`
  > on its entry.

- Phase is cheap CPU-only (no paid API, no GPU) AND ships without (a)/(b)/waiver
  → CONCERN bullet under "Issues Found"; NOT a standalone FAIL. Persist via
  `task.py raise-concern` per Step 0.8 / Rule 11 so the concern actually binds.

**Sub-check (2) — consumer inter-phase contract assertion.** For each phase
whose INPUT is another phase's persistent output (a JSONL / parquet / npz file
the earlier phase wrote), READ the consumer phase's entry:

- The consumer asserts every required input field non-empty (`assert
  row['prompt_text']` — never a silent `row.get('prompt_text', '')` chained to a
  filter), reports drop counts (`n_dropped > 0 → fail loud with the drop
  fraction`, per `.claude/rules/llm-judging.md` rule 9's drop-never-coerce
  discipline and CLAUDE.md § Critical Rules "Fail fast"), and does the check
  BEFORE any heavy initialization: `AutoModel.from_pretrained`, `LLM(`,
  `accelerate launch`, `torch.distributed.init_process_group`, first GPU-tensor
  allocation.
- **Verdict routing — sub-check (2):**
  - Contract assertion present + fail-loud + BEFORE model init → PASS.
  - Assertion present but AFTER model init → verdict FAIL with a single
    `Critical` issue tagged `consumer-contract-post-init` (SUBSTANTIVE, never
    stripped):

    > `epm:experiment-implementation v<n>`'s consumer phase `<name>` reads
    > `<producer_output.jsonl>` and initializes vLLM / AutoModel before checking
    > the input contract. A schema mismatch (missing field, empty row) then
    > wastes a pod cycle rather than seconds of CPU (the #1689 shape: 79,800
    > render rows, vLLM initialized, then died on 33% empty `prompt_text` — one
    > `.get()` with no assert). Re-post `v<n+1>` moving the assertion above the
    > model init call: `assert all(r['prompt_text'] for r in rows), f'{sum(1
    > for r in rows if not r.get(\"prompt_text\")):d} rows empty'` (fail loud;
    > no silent drop, no default fill).

  - Assertion silently drops / defaults / substring-matches → CONCERN bullet,
    persisted via `task.py raise-concern` (contract enforcement present but
    permissive enough that a schema mismatch could mask itself).
  - No assertion + consumer initializes heavy state → same FAIL as above.

**Fingerprint-of-degradation.** A gate that cannot NAME the expected phase
output artifact degrades to a judgement call (task-body constraint). This gate
credits an artifact name from THREE sources, in order: (i) the plan §9
`phase_outputs:` map (planner.md §9 requirement, see the sibling planner-side
edit), (ii) a `--out-root` / `--sentinel` flag the dispatcher exposes, (iii)
a plan-body `**Design:**` / `**Methodology:**` section explicitly naming the
output. If NONE of the three exist, record `Step 0.69: unable to verify —
plan/diff names no phase output artifact` (a CONCERNS bullet, NOT a FAIL — the
gate is designed to degrade gracefully; the sibling planner.md §9 edit is what
raises the artifact-name floor over time, without ratcheting this gate to a
false FAIL). This matches Step 0.65's "necessary-but-not-sufficient grep"
caution and Step 0.67's "plausible-but-unconfirmed" pattern (CONCERNS not FAIL).

Record the verdict as one line: `Step 0.69: PASS — <N> phases idempotent, <M>
consumers assert contract early`, `Step 0.69: FAIL — <phase> not idempotent /
<phase> contract post-init`, `Step 0.69: CONCERNS — <one-liner>`, or `Step
0.69: N/A — no multi-phase dispatcher in diff`.

Sibling: this gate reads dispatcher SHAPE for idempotency; Step 3.6 reads
long-loop RESTARTABILITY (>~1h serial loops must persist + resume) — the two
compose (a Step 0.69-idempotent phase's inner loop still owes Step 3.6's
intra-phase checkpointing).

## Step 0.70 detail — smoke-variable gating

**Trigger — bash dispatcher (`scripts/*dispatch*.sh`, any `.sh` in the diff)
carries a `<name>_smoke=` declaration OR a live `<name>="$<name>_smoke"`
assignment** (either grep hits):
`grep -nE '^[[:space:]]*(local +)?[a-z_][a-z_0-9]*_smoke='` /
`grep -nE '^[[:space:]]*(local +)?[a-z_][a-z_0-9]*="\$[a-z_][a-z_0-9]*_smoke"'`.
A sibling `<name>_full=` is NOT required — the load-bearing signature is a
live var pinned to `_smoke` with no `$SMOKE` fallback (pre-R13 #1689:
`conds_smoke="assistant_chat"` + `conds="$conds_smoke"`, NO `conds_full=` —
`git show 15906d680a^:scripts/issue1689_dispatch.sh` L134-135). No trigger
→ record `Step 0.70: N/A — diff carries no smoke-scoped variable`.

**Sub-check (1) — every live `<name>="$<name>_smoke"` has a `$SMOKE`-guarded
fallback in the SAME enclosing function/block.** ONE of: (a)
**bidirectional-pair** (preferred; `models`'s #1689 shape — declare both
variants, default `<name>="$<name>_full"`, then `[ -n "$SMOKE" ] &&
<name>="$<name>_smoke"`); (b) **in-line SMOKE guard** on the same variable
(`[ -n "$SMOKE" ] && <name>=...`, `if [ -n "$SMOKE" ]; then …; fi`,
`${SMOKE:+"$<name>_smoke"}`, or equivalent reading `$SMOKE` in the SAME
command); (c) **waiver** below. Otherwise-ungated → FAIL.

**Sub-check (2) — no hardcoded smoke-scoped literal masquerading as
production default.** Fires ONLY when a `<name>_smoke=` is declared in the
file. In the SAME enclosing function/block, flag any live loop-driving
assignment whose value is a hardcoded string equal to that `<name>_smoke`
variant (or a subset of `<name>_full` when declared) AND has no
`$SMOKE`-conditional override (pre-R13 `run_phase_fit_cells` L158
hardcoding the smoke `model_slug`). Waiver same as (1c).

**Sub-check (3) — orphaned `_full` (dead-code signal).** `<name>_full=`
declared with no `<name>="$<name>_full"` assignment anywhere → FAIL, tag
`smoke-var-orphan-full` (separately tagged so it never fuses with the
primary `smoke-var-ungated`).

**Waiver** (mirrors Step 0.69's `# PHASE_IDEMPOTENCY_EXEMPT:`):
`# SMOKE_VAR_UNGATED_EXEMPT: <reason ≥ 20 chars>` on the line above the
ungated assignment. Credits PASS.

**Verdict routing:** All (1)+(2) pass or waived AND no orphaned `_full` →
PASS. Any (1) or (2) FAIL → verdict FAIL, single `Critical` tagged
**`smoke-var-ungated`** (SUBSTANTIVE — never stripped by Step 5c-bis):

> `epm:experiment-implementation v<n>`'s dispatcher `scripts/<file>` at line
> `<L>` assigns `<var>="$<var>_smoke"` (or hardcodes its literal to a live
> loop-driving variable) with no bidirectional-pair fallback and no
> in-line `$SMOKE` guard — silently ships the smoke-scoped list as the
> production default. #1689 shape (2026-07-26, commit `15906d680a`):
> `conds="$conds_smoke"` with the `$SMOKE` gate on the sibling `models` but
> never on `conds` collapsed a 21-condition lattice to 1 through eight
> rounds. Re-post `v<n+1>` with the bidirectional-pair pattern (declare
> `<var>_full=`, default `<var>="$<var>_full"`, then `[ -n "$SMOKE" ] &&
> <var>="$<var>_smoke"`), OR add `# SMOKE_VAR_UNGATED_EXEMPT: <reason ≥
> 20c>`.

Any (3) FAIL → verdict FAIL, single `Major` tagged **`smoke-var-orphan-full`**
(SUBSTANTIVE): body names `<var>_full=` at `<file>:<L>` declared but never
assigned to a live `<var>`; re-post either assigning `<var>="$<var>_full"`
as the default (with optional `[ -n "$SMOKE" ] && <var>="$<var>_smoke"`) or
deleting the unused declaration.

Record one line: `Step 0.70: PASS — <N> smoke-scoped variables correctly
gated`, `Step 0.70: FAIL smoke-var-ungated — <var> ungated at <file>:<L>`,
`Step 0.70: FAIL smoke-var-orphan-full — <var>_full declared at <file>:<L>
but never used`, or `Step 0.70: N/A — diff carries no smoke-scoped
variable`.

**Fingerprint-of-degradation** (mirrors Step 0.69): bash source only; a
python dispatcher whose smoke gating lives in `args.smoke` records `Step
0.70: N/A — python dispatcher; smoke gating is arg-level`. Widening path
(python `args.smoke`, YAML/JSON `conditions_smoke:`) named in the plan
follow-ups.

## Step 0.71 detail — smoke blind-spot enumeration

**Grep triggers** (run over the diff's touched `.py`/`.sh` files):
`grep -nE 'if (not )?(ctx\.|args\.|self\.|cfg\.)?smoke\b' <file>` for branch
forms; `grep -nE 'smoke\s*=' <file>` for kwarg-gated gates
(`assert_split(..., smoke=ctx.smoke)`); ternaries
(`X if smoke else Y`). For each hit, classify:
**(a) substituted implementation** — the production import / model
constructor / API call sits only on the non-smoke side. Three sub-forms,
all (a): the plain if/else; the early-return form (`if smoke: return <toy>`
with the real import + constructor inline after the branch); and the
HELPER-WRAPPED form — `if smoke: return <toy>` followed by
`model = _load_model(...)` where a module-local lowercase helper holds the
production import. The helper-wrapped form is the REAL #1336 SLURM 4684
shape (`_load_sentence_transformer()` wraps
`from sentence_transformers import SentenceTransformer`, so the import is
invisible at the branch site) — follow lowercase callees one level into
module-local helpers when classifying;
**(b) downgraded gate** — an `assert` /
`raise` sits only on the non-smoke side — including the per-check
`if smoke: logger.info(...) else: raise AssertionError(...)` form with NO
early exit, the REAL #1336 SLURM 5005 shape — or the smoke side
early-returns before the gates
(`def assert_split(..., smoke=False): if smoke: return`).
Shrink-only smoke parameters (fewer cells / seeds / rows,
same code path) are NOT triggers — that class is owned by Step 0.6 coverage
and the #1611/#1727 gates.

**The check.** Fetch the plan's smoke section (worktree `plans/v*.md` /
`.claude/plans/issue-<N>.md`) and the implementation marker's `## Smoke run`
block. Every (a)/(b) branch must be NAMED in the `Smoke blind-spot
enumeration:` block of either surface — an entry stating what the smoke PASS
does not certify for that branch. The empty form is the literal
`none — smoke executes every production gate`; any (a)/(b) hit falsifies it
(FAIL, same tag — the enumeration is WRONG, which is worse than absent).

**FAIL template.** One Critical tagged `smoke-blind-spot-unenumerated`
(SUBSTANTIVE — never in the Step 5c-bis strip set):
`<file>:<L> — smoke-conditional <substituted-implementation|downgraded-gate>
branch not named by the blind-spot enumeration (<plan/marker ref>); the smoke
PASS certifies less than the plan presents. Remedy: add the enumeration line
(what production-only gate/import/implementation this branch hides) — or
exercise the import under smoke (an import-only probe on the smoke path: the
cheap move that would have prevented SLURM 4684) — or unify the branch away
(architectural parity).` A plan/marker with NO
enumeration block at all and ≥1 (a)/(b) branch in the diff FAILs the same
way, citing `.claude/rules/smoke-blind-spots.md`.

**Mechanical companion (advisory).** `uv run python
scripts/workflow_lint.py --check-smoke-blind-spots --smoke-blind-spot-scripts
<touched .py files> --smoke-blind-spot-plan <plan path>` — WARN-only AST
scan; it resolves module-local lowercase callees ONE level deep, so the
helper-wrapped #1336 shape fires, but deeper nesting, cross-module helpers,
dynamic dispatch, and non-`smoke`-named flags are its disclosed false
negatives — THIS lens is the binding gate for exactly those. Use it to seed
the grep, never as the verdict (naming-completeness is reviewer-owned).
