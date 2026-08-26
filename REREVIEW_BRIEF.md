# Task #2569 — Step 5 re-review (round 4), SHARED brief

Read this, then your shard's file list in your prompt.

## What you are reviewing

Round 3 returned FAIL on both sides. Two fix units (H1, H2) then ran on disjoint
file sets and closed 4 concerns; the orchestrator landed two more commits (a lint
waiver and a brief amendment). This is the re-review of that work.

**Review range:** `6193ee115b..origin/issue-2569`. ~8 files, ~71 KB at the payload
commits — under the ~300 KB budget where a single reviewer autocompact-thrashes, so
this round is **one reviewer per side, no sharding**. Review the round-scoped
range, NOT the branch-wide diff (1444 KB — it will thrash you).

The base is fixed (`6193ee115b`, the last pre-v4 commit); the tip is resolved at
review time rather than pinned, because pinning it excludes the brief-and-waiver
commits that land after the payload — including this brief.

```bash
WT=/home/thomasjiralerspong/explore-persona-space/.claude/worktrees/issue-2569
git -C "$WT" fetch origin issue-2569 --quiet
git -C "$WT" diff 6193ee115b..origin/issue-2569
```

Files: `scripts/issue2569_{figures,gateladder,xmodel_capture}.py`,
`tests/test_issue2569_{curve_der,figures,gateladder,xmodel}.py`,
`FIXROUND_BRIEF.md`.

## Plan path — resolve it, never hardcode a status folder

```bash
PLAN="$(cd /home/thomasjiralerspong/explore-persona-space && uv run python scripts/task.py find 2569)/plans/plan.md"
```

Run `task.py find` from the MAIN checkout. A worktree's `tasks/` tree is frozen at
its base commit and serves a stale plan with no error. `plan_version=v4`.

## Ledger state: 82 concern ids, 66 addressed, 16 open

```bash
cd /home/thomasjiralerspong/explore-persona-space
LEDGER="$(uv run python scripts/task.py find 2569)/concerns.jsonl"
```

The 16 open ids are RECORDED deferrals, not oversights. Seven are scope deferrals
from round 2 (`leg1-fixedpoint-neighbors-no-pb-producer`,
`leg3-rows-legs-need-pb-store`, `leg6-ft-arms-no-delta-tf-unit`,
`leg6-tbar-basis-mismatch`, `atlas-phasefits-correspondence-null-serial`,
`review-tip-drift-orchestrator-memory-commits`,
`weights-base-regime-omits-banked-map-fingerprint`); nine are carried from round 3
(`ans-len-fingerprint-download-toctou`,
`ans-len-local-fingerprint-not-content-stable`,
`atlas-aligned-spectra-key-ids-not-content`,
`dv3-single-arm-tolerance-not-producer-scoped`,
`dw-effective-rank-registered-lines-ungated`,
`dwfleet-align-units-key-omits-adapter-and-ft-sidecar-content`,
`h7-cross-basis-unit-asymmetry`,
`h7-low-evaluability-absent-from-disposition-string`,
`pd-pilot-single-survivor-pass-preexisting`).

Each carries its reasoning in the ledger's `evidence` field. Read it before you
start. If you think a deferral is WRONG, say so with evidence — that is a
legitimate finding, and one round-2 reviewer found a real defect exactly that way.
Do not simply restate a deferral as a new blocker.

## The four concerns this round closed — verify the fixes, do not trust the reports

- `h2b-two-of-five-yields-registered-verdict` — H1 raised
  `MIN_WELLPOSED_VERDICT_POINTS` from 2 to 3 in `issue2569_gateladder.py`, adding
  `UNDECIDABLE_INSUFFICIENT_WELLPOSED` and `same_sign_all: None` below the floor.
  Check the floor is REACHABLE (does the protocol-matched point set admit 3?), that
  the undecidable verdict reaches the artifact a reader consumes, and that the
  `MEAN_ABS_DR2_PASS` / `_KILL` bands are byte-unchanged from `4a48517b13`.
- `dw-degeneracy-labels-not-consumed` — degeneracy now renders as marker SHAPE
  (circle = well-separated, diamond = near-tied) with explicit `mew=1.0`. Verify by
  RENDERING: `set_paper_style("blog")` sets `lines.markeredgewidth: 0`, so an open
  marker without explicit `mew=` draws zero ink. Read the PNG.
- `pd-gate-pass-not-bound-to-live-regime` — H2 added `_GATE_BINDING_FIELDS`,
  `_require_gate`, `_gate_binding_from_store` in `issue2569_xmodel_capture.py`.
  Check the binding field set covers everything that changes the gate's meaning,
  and that exempt fields carry stated reasons.
- `select-stage-texts-resume-content-unpinned` — `source_content_sha256` in the
  `_stage_texts` sidecar plus `revision=rev` at every `hub.stage_hub_file` call.
  Check the sha is over CONTENT in a stable domain, not over identity.

Two orchestrator commits also land here: a `HUB_VERIFY_RETRY_EXEMPT` waiver at
`issue2569_xmodel_capture.py:408` (the flagged `api.list_repo_tree` is already
inside the `_walk` closure passed to `hub._retry_upload` at :418 — the checker's
AST scan cannot see through the closure), and a `FIXROUND_BRIEF.md` amendment.
Both are in scope; the waiver in particular deserves a skeptical read.

## What I most want from this round — the ensemble's measured split

Across four cross-side disagreements over three rounds, Codex won three, and all
three were the same species: **a verdict computed over an insufficient or wrong
denominator** (a dv3 flag true over an empty arm roster; an H7 criterion scored on
evaluable pairs where the plan registered all pairs; an H2b verdict from a mean
over a single delta). Claude's shards were the ones that verified hard
mathematical claims by EXECUTION — one wrote its own fp64 oracle and got
per-vector dots of 1.000000 against a rank-r SVD; another derived the Haar
identity independently and swept every consumer.

So: push hardest on **denominators, vacuity, and gate-precondition binding**, and
on **exactness claims you can verify by running the code**. Concretely, the four
defect classes this task keeps producing:

1. **Vacuous or underpowered verdicts** (six instances) — a computation emitting a
   well-formed number in a regime where the number is meaningless. Ask of every
   verdict: what is the denominator, and what does this return at n=1 or n=0?
   `all()` over one element is vacuously True.
2. **Identity-not-content resume keys** (four-plus) — a resume/cache key over
   paths, sizes, or mtimes rather than the bytes that change the output.
3. **Impossible-schema fixtures** (six) — a test fixture carrying a field the real
   producer never emits, so the test passes against a shape that cannot occur.
   Check every new fixture against the producer.
4. **Cross-file producer/consumer seams** (three) — a schema change landing in one
   file while its consumer in another still parses the old shape. The units worked
   disjoint file sets, so this seam is exactly where a defect can hide.

## Two duties, both from failures in earlier rounds

**Contract changes are DISPATCH REQUESTS, not notes.** If you find a producer
whose schema changed without its consumer updated, name the consumer FILE and
FUNCTION explicitly as a blocker. A round-2 unit reported such a change under
"Contract notes for downstream units"; the orchestrator summarized it into a
marker instead of dispatching a fix, and two blockers survived the round.

**Wait your own gates out synchronously.** You get one turn and nothing
re-invokes you when a background job finishes. If you run a lint or a test union,
wait it out inside the turn (foreground with an explicit `timeout`, or a bounded
`Monitor` until-loop — foreground `sleep` chains are hook-blocked). A killed or
still-running gate is INCONCLUSIVE, never clean. Run gates from the WORKTREE:
checks path-local to `scripts/` resolve against the cwd's tree, so a repo-root run
scans `main`, where a branch-only file is absent and the check trivially passes.

## Verdict

End with `PASS` or `FAIL`. On FAIL, one blocker per finding, each with the file,
the line, and the failure scenario (concrete inputs → wrong output), plus a
machine-readable row in EXACTLY this shape:

```
CONCERN:: <SEVERITY> <kebab-case-id> <one-line summary>
```

Severity FIRST, then the id, then the summary — space-delimited, no pipes.
`<SEVERITY>` is one of `BLOCKER` / `CONCERN` / `NIT`; the id must match
`^[a-z0-9][a-z0-9-]{1,79}$`. This is the format
`scripts/persist_verdict_concerns.py` parses (token 1 = severity, token 2 = id,
remainder = summary), and it rejects anything else as MALFORMED — an earlier
version of this brief specified a pipe-delimited id-first row, every row of a
real verdict was refused, and the orchestrator had to translate them by hand.
When there is nothing to persist, emit the single literal row `CONCERN:: none`.

Tag any blocker that is purely a mechanical contract (`marker-shape`,
`smoke-run-missing`, `git-provenance`) as such; everything else is substantive
and will not be stripped.
