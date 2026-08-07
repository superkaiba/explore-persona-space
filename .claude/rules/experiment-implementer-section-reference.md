---
paths:
  - ".claude/rules/experiment-implementer-section-reference.md"
description: >
  Full checklist detail for experiment-implementer.md's heaviest items
  (Before-writing-code item 5 smoke/sweep parity; After-implementation
  items 3 + 7), relocated from .claude/agents/experiment-implementer.md
  (per-spawn system-prompt cost; #1090/#2054 fixed-overhead deaths).
  Loaded ONLY via the explicit § pointer lines in
  experiment-implementer.md — the self-matching `paths:` glob keeps this
  file out of every other agent context.
---

# Experiment-implementer section reference (relocated checklist detail)

One H2 per relocated item, detail relocated verbatim from
`.claude/agents/experiment-implementer.md`. Read ONLY the section you need:
Grep the heading, then a chunked `Read` of that span (per the spec's
§ Context budget) — never the whole file. The OPERATIVE contract for every
item stays in experiment-implementer.md; this file carries the extended
recipes, verbatim templates, and incident grounding.

## Before-writing-code item 5 detail — smoke/sweep architectural parity

5. **Smoke/sweep architectural parity self-check.** Walk the plan's
   smoke-phase definition vs sweep-phase definition. **PREFER UNIFICATION:**
   if the plan unified the paths (smoke IS sweep with `--cells 1 --seeds 1`
   or equivalent single-cell parameterization — same dispatcher, same
   subprocess shape, same LAUNCH WIDTH (`--num_processes` / CVD
   composition — smoke never narrows the process shape; see the
   smoke-width entry in `.claude/rules/gotchas.md`, #1315/#1333), same
   env injection, same logging surface, same teardown sequence, AND the
   cell-subset parameterization threads through
   EVERY phase the dispatcher executes), the verdict is `PASS_UNIFIED`
   ONLY when the Axis 1 import-resolution leg passed AND every planned
   arm's per-arm-resolution row (Axis 2) reads `REAL` or `N/A` — a
   `FALLBACK` row makes the verdict `PASS_PARTIAL`.

   **Axis 1 — Import-resolution leg (execute on the REAL branch).**
   For every changed entrypoint in this round's diff (a `scripts/*.py`
   or `src/**/*.py` file the round touches), execute its deferred/lazy
   imports on the REAL branch. Bare `python -c 'import <module>'` fires
   ONLY module-level imports; the #1689 rounds 2/3/4 false-pass class
   was a `from foo import DispatchCall` NESTED inside a function body,
   which a top-level import never fires. Use ONE of these three shapes:
   (a) **PREFERRED — the dispatcher's `--import-check` mode**, which
   loads the entrypoint AND resolves every function-body `from … import
   …` + every `importlib` / `getattr` deferred import the entrypoint
   hits on its real code path (a one-line `if args.import_check: from
   foo import (DispatchCall, dispatch_calls); sys.exit(0)` at the top
   of `main()` is sufficient — the point is to NAME the deferred
   symbols). (b) **acceptable fallback** — `uv run python -c 'from
   scripts.issueNNNN_entrypoint import (dispatch_calls, DispatchCall,
   …every named function-body import…)'`; the `from <mod> import
   <names>` form fires those specific deferred names at module load,
   BUT only when every deferred import in the entrypoint's changed
   lines is enumerated in the `from ... import ...` list. A bare
   `import <mod>` is EXPLICITLY INSUFFICIENT here — it does not close
   this failure class. (c) **for pure top-level-import entrypoints**
   (NO `def foo(): from … import …` and NO `importlib` anywhere in the
   changed file's changed lines), a bare `uv run python -c 'import
   <mod>'` suffices — state this explicitly in `notes:` as
   `import-resolution-mode: top-level-only` so the reviewer can verify
   the changed lines contain no function-body imports. Record the
   EXACT command executed in the marker `notes:` under
   `import-resolution: <command>`; a deferred import behind a mock-only
   branch is the #1689 rounds 2/3/4 false-pass class — the smoke MUST
   exercise the real branch. If import resolution FAILS, the verdict
   is `FAIL_NO_CANARY` (a broken import is a coverage failure, not a
   fallback choice).

   **Axis 2 — Per-arm resolution attestation.** For every arm / rung /
   condition the PLAN §4 Design names (a `kind: experiment` plan lists
   these explicitly; a `kind: infra` plan typically names none), state
   in the marker `notes:` under a `per-arm-resolution:` sub-block —
   one row per plan-named arm:

   ```
   per-arm-resolution:
     <arm-name-1>: REAL — <one-line: which real computation ran>
     <arm-name-2>: FALLBACK — <one-line: what stub / bias-refit / default>
     <arm-name-3>: N/A — no computation path (<API-only | data-load-only | …>)
   ```

   An arm reads `REAL` when the smoke exercised its production
   computation path (not a stub, not a mock-only branch, not a
   fallback); `FALLBACK` when the smoke resolved to a placeholder
   (a bias-refit R², a `NotImplementedError`-guarded default, an
   early-return); `N/A — no computation path` when the arm is
   legitimately arm-less (API-only phase, data-loading probe, an arm
   with nothing to compute in the smoke slice — the vacuous case).
   The vacuous form for a `kind: infra` plan whose §4 names no arms
   is a single-line `per-arm-resolution: N/A — no plan-named arms`.
   A missing per-arm row for a plan-named arm is a marker-shape
   violation (verdict `FAIL_NO_CANARY`, not `PASS_PARTIAL`).
   **Per-phase subset threading is part of the PASS_UNIFIED
   definition, not optional:** list each phase the dispatcher runs
   (train, eval / cross-eval enumeration, anchor selection, analysis
   tolerance, upload) and name where each phase's cell list comes from
   — it must derive from the same `--cells`/override subset the smoke
   passes. A smoke whose subset shapes only the train loop while a
   downstream phase re-enumerates the full registered grid is NOT
   unified — verdict `FAIL_NO_CANARY` (incident #546 r1: train honored
   smoke overrides, cross-eval enumerated the full 120-cell grid,
   HF-404'd on never-trained adapters). **Same duty covers NON-cell smoke axes:**
   for every axis the smoke slices below production
   (questions, rows, steps, draws), verify the sliced size satisfies
   every downstream phase's minimum-N asserts — grep the consumers for
   `assert len(...) >=` / min-N `raise` shapes AND arithmetic floors
   (slicing, n-1 divisions), never from the plan's stub prose — and
   name the floor per sliced axis in `notes:`. Resize an under-floor
   slice up to the floor (recording it) where the plan permits; a
   slice you cannot bring to the floor makes the smoke un-passable by
   construction — verdict `FAIL_NO_CANARY` (incident #1315 r4:
   `questions[:1]` sat below `split_half_self_cosine`'s `len(qs) >= 2`
   and a PASS_UNIFIED smoke crashed at its LAST phase). Resizing keeps
   floors from firing in the MAIN smoke leg only — the gate branches
   themselves must still execute once in a separate degenerate probe
   (data-dependent-gates duty, "After implementation" item 3). If phase
   coverage holds AND the Axis 1 import-resolution leg passed BUT ≥1
   planned arm's per-arm-resolution row reads `FALLBACK`, the verdict
   is `PASS_PARTIAL arms_stubbed=<comma-list-of-fallback-arm-names>`.
   Step 6d.0 refuses to dispatch on `PASS_PARTIAL` for planned
   experiment arms — the round bounces to `status:planning` (mirroring
   `FAIL_NO_CANARY`) for the planner to either resolve the stubbed
   arms in the diff or re-authorize the stubs in a plan §4
   `### Authorized smoke stubs` block (one table row per arm:
   backticked arm name, why it cannot run at smoke, compensating
   control), landed through the plan-approval gate; after that
   authorization lands, re-post the marker as `PASS_AUTHORIZED_STUB
   arms_stubbed=<same list>` — Step 6d.0 grants it mechanically via
   `task.py check-authorized-stub` (rc=0 = GRANT; #2171). Self-tag
   `PASS_AUTHORIZED_STUB` directly (INSTEAD of `PASS_PARTIAL`) only
   when the CURRENT `plans/plan.md` already carries the block covering
   every FALLBACK-rowed arm.
   If the plan diverged
   (e.g., smoke uses in-process `train_one_cell`, sweep uses a subprocess
   wrapper) AND the plan §4 Design section justified the divergence in two
   sentences AND named which canary cell exercises the sweep path during
   smoke, the verdict is `PASS_CANARY canary_cell=<cell_id>` (same
   REAL/N/A per-arm invariant as `PASS_UNIFIED` — a `FALLBACK` row
   under `PASS_CANARY` is a marker-shape violation). If the plan
   diverged WITHOUT the canary section (or without the two-sentence
   justification), the verdict is `FAIL_NO_CANARY`.

   **Post the marker as a separate events.jsonl row BEFORE you EXIT this
   pre-flight phase, via:**
   ```
   uv run python scripts/task.py post-marker <N> epm:smoke-architecture-check \
     --note "verdict: PASS_UNIFIED
   notes: <one-line description of how smoke = sweep with one cell, naming
   each phase's cell-list source (e.g. train/eval/anchor all read --cells)
   and, per sliced non-cell axis, its smoke size vs the downstream min-N
   floor (e.g. questions=2 >= split-half floor 2)>
   import-resolution: <the EXACT command executed on the REAL branch —
   see Axis 1 shapes (a)/(b)/(c) above>
   per-arm-resolution:
     <arm-1>: REAL | FALLBACK <reason> | N/A — no computation path
     <arm-2>: …
   resume-matrix:
     <leg-name-1>: REAL — <one-line: which resume/topup/salvage leg was exercised, e.g. topup-record-recorded-miss> | FALLBACK — <one-line reason> | N/A — no resume/topup/salvage branch in this diff
     <leg-name-2>: …
   production-outroot-unit:
     <unit-name>: REAL — <one-line: which unit ran into eval_results/issue_<N>/> | FALLBACK — <one-line: production-shape unit infeasible at smoke, reason> | N/A — dispatcher writes no out-root (analysis-only / API-only)
   "
   ```
   Legal `verdict:` tokens: `PASS_UNIFIED` | `PASS_CANARY
   canary_cell=<id>` | `PASS_PARTIAL arms_stubbed=<comma-list>` |
   `PASS_AUTHORIZED_STUB arms_stubbed=<comma-list>` |
   `FAIL_NO_CANARY`. For `PASS_CANARY`, cite the plan §4 two-sentence
   justification in the `notes:` line. For `PASS_PARTIAL`, list the
   fallback-rowed arm names verbatim (as a set they must equal the
   arms whose `per-arm-resolution:` row reads `FALLBACK` — the
   `arms_stubbed=<comma-list>` set-equality scopes to
   `per-arm-resolution:` rows ONLY, NOT the `resume-matrix:` or
   `production-outroot-unit:` sub-blocks' own `FALLBACK` rows; the
   same set-equality scoping binds `PASS_AUTHORIZED_STUB`).
   For `FAIL_NO_CANARY`,
   post the marker AND additionally emit a one-line
   `<!-- workflow-fix-candidate v1 -->` block in your implementer report
   text suggesting the planner re-architect toward unification, then EXIT.

   Do NOT rely on an inline HTML-comment block in your report text —
   the `/issue` Step 6d.0 gate scans `events.jsonl` for a separate
   `epm:smoke-architecture-check` row, not for substrings inside another
   marker's payload.

   The planner revises toward unification on `FAIL_NO_CANARY` /
   `PASS_PARTIAL`; canary is the escape hatch when unification is
   genuinely impossible (e.g., per-cell vLLM allocation that can't
   reset in-process). Rationale: #397 rounds 9/10/10' PASSed smoke and
   crashed sweep because smoke didn't exercise the subprocess
   dispatcher; #1689 rounds 2/3/4 PASSed smoke behind `--mock-response`
   branches and stub fallbacks. Step 6d.0 refuses to dispatch on
   anything other than `PASS_UNIFIED`, `PASS_CANARY`, or a
   `PASS_AUTHORIZED_STUB` that `task.py check-authorized-stub`
   mechanically grants (rc=0; #2171).

   Additional smoke-contract requirements (Step 6d.0 gate refuses on
   missing evidence; every requirement below extends the
   smoke-architecture-check marker's per-leg attestation shape):

   - **Cross-phase data-contract smoke.** When any phase CONSUMES
     artifacts from a DIFFERENT issue / condition registry (parent
     matrices, sibling adapters, prior eval JSONs), the smoke runs the
     consumer against the producer's REAL output shape at tiny N — not
     component calls on synthetic fixtures (#518: bakeoff phase read
     #474's 16-condition matrix while #518 passed R1..R24 — `KeyError`
     11 h into production).
   - **Smoke drives the production entrypoint.** Invoke the launcher
     CLI with the production flag set (then scaled down), never the
     library functions directly — a function-level smoke misses branches
     the launcher never enters.
   - **Real-trainer-path smoke for callback-bearing training code.**
     When the diff passes `callbacks=[...]` to `train_lora` / any HF
     Trainer, the smoke MUST construct a real (SFT)Trainer and traverse
     `__init__ → on_init_end → on_train_begin → step → on_train_end`
     (tiny same-arch model, CPU, `max_steps=1-2`) — a
     dry-run/import-check substitute never fires `on_init_end` (#816:
     non-subclassed callback passed straight to production crash after
     53/53 sibling cells burned GPU-hours; gotchas.md HF Trainer
     callback entry).
   - **Tiny-real CPU e2e before the FIRST GPU launch of a multi-stage
     driver.** Mock-seam smokes surface shape bugs one per GPU cycle
     (#906 r11-r14: four bugs, four ~1.5h pod cycles). Run the FULL
     production path once on CPU with REAL library types at every
     internal seam; fake ONLY GPU-scale weights + the remote Hub
     boundary (GPU-bound phases: see item 3). When the pipeline
     INGESTS a real corpus (WildChat/LMSYS-class streaming), the
     **data-ingestion probe class** (#1092) binds too: a bounded
     tiny-real streaming probe against the REAL dataset — a kept cap
     AND a TOTAL-streamed-rows cap, `kept > 0` per dataset,
     per-filter rejection counters in the `done:` line (recipe:
     gotchas.md "Real-corpus streaming filters"). Record it under
     `## Smoke run` — Step 6d.0-bis refuses seam-stubbed evidence and
     synthetic-fixture-only evidence for a real-corpus ingestion phase.
   - **Resume-matrix smoke.** When the diff exposes ANY resumable
     re-entry branch — a resume predicate (done-file skip, sidecar-based
     terminal-verdict skip), a `--from-phase <name>` / `--resume` flag, a
     salvage/topup leg, a recorded-verdict re-read — the pre-launch
     smoke exercises EACH such leg at least once against a synthesized
     partial state: run the smoke, interrupt or seed the partial
     artifacts the leg re-reads, re-enter with the resume flag or the
     recorded-sidecar in place, confirm exit 0 + the leg's designed
     disposition (skip / re-emit / continue). Grep the diff for the
     branch class (`resume`, `topup`, `salvage`, `_record.json`,
     `if.*exists.*: return`, `--from-phase`) to enumerate the legs; a
     leg that cannot be brought to the smoke floor is declared
     `FALLBACK — <one-line reason>` in the marker's `resume-matrix:`
     sub-block (matching the per-arm-resolution vocabulary), mirroring
     the existing PASS_PARTIAL declared-escape shape. Rationale (#1947
     P0 launches 4-5, #1315 r6, #1112 r6): recorded terminal-verdict
     re-entry crashes, salvage input overwrites, resumed-process
     side-effect loss, partial-artifact resume — four distinct crash
     classes that fire only on the RE-ENTRY leg the smoke never
     exercises. Persisted memories:
     `feedback_resume_predicate_recorded_terminal_verdicts.md`,
     `feedback_salvage_inputs_pin_identity.md`,
     `feedback_registry_side_effect_lost_on_resume.md`,
     `feedback_partial_artifact_resume_and_trainer_ckpt_tokenizer.md`.
   - **Real production out-root unit.** Before the full launch, ONE
     real corpus/fit unit runs end-to-end at production shape writing
     to the PRODUCTION out-root (`eval_results/issue_<N>/...`), not a
     `/tmp/issue-<N>-smoke/` twin. This catches seams that fire only
     against the canonical path: `mkdir` of an out-root parent whose
     directory tree the first cell creates, registry-coupled metadata
     lookups keyed on the caller's own cell/arm ids
     (`_pfx_fit_core`-style `arm_method(arm_id)` misses), path
     predicates that gate on the canonical out-root prefix. Compose
     with the existing PASS_CANARY declared-escape when one cell at
     production shape is genuinely infeasible (GPU-scale weight
     materialization, multi-hour per-cell wall) — declare it as
     `production-outroot-unit: FALLBACK — <one-line reason>` in the
     marker. The unit's outputs are the FIRST cell of the out-root the
     full launch subsequently populates — no clobber of committed
     artifacts by construction (a first-launch out-root has none); a
     re-launch smoke into an already-committed out-root uses the
     scratch-dir redirect / restore-after-smoke fallback exactly as
     today (§ Smoke outputs never overwrite committed artifacts).
     Rationale (#1947 P4/P5 round 2): 8 fit units died on `KeyError:
     'imp-bare-con-sv-s42'` at the reused `_pfx_fit_core`'s registry
     lookup; smokes missed it because tiny fixtures used registry arm
     ids. Persisted memory:
     `feedback_reused_fit_core_registry_lookup_seam.md`.

## After-implementation item 3 detail — end-to-end smoke run per phase

3. **End-to-end smoke run PER PHASE.** For EACH distinct entrypoint the
   experiment pipeline executes — data-gen, training, eval (and any
   separate analysis / upload step) — run the script ONCE on a tiny real
   slice and confirm exit code 0 + a real artifact landed. Tiny slice
   means: 1 seed, the minimum contexts / cells, the base model or a tiny
   throwaway checkpoint, `max_steps=1` for training, a 1-example dataset
   for data-gen, etc. Eval rigs especially must be smoke-exercised
   end-to-end before code-review — a never-before-run eval script that
   was only import-checked or that piggy-backed on the training script's
   smoke is the canonical missing-phase case and code-reviewer will FAIL
   with `smoke-run-missing` (incident: #408 burned six relaunches catching
   one bug per cycle on a 203 KB eval rig that had never been run
   end-to-end). Record each phase as a `### <phase-name>` sub-section
   under `## Smoke run` in the report (see Report Format § (c) below).
   This catches the bulk of "experimenter discovers it crashes at
   startup / at eval" failures before the pod is even provisioned.
   A composed judge-instrument leg counts as wired only with the
   rule-27 parse-contract round-trip test
   (`.claude/rules/llm-judging.md`); a dry run proves routing only.

   **Per ARM CLASS, not just per phase.** When a phase's driver spans
   MULTIPLE ARM CLASSES (distinct source-context classes / recipe
   branches — e.g. persona-context vs bare-context arms), the tiny
   smoke covers AT LEAST ONE cell of EACH arm class, not one arm
   overall: per-arm seams (source-context construction, negative-panel
   assembly, `ModelOrganism` wiring) are invisible to a single-arm
   smoke however tiny-real its seams (#1090 fu5: a formatting-arm-only
   smoke passed; all 3 bare-context arms then died on the #527/#538
   panel-disjointness assert after a full 4×A100 GCE cycle). This is a
   coverage-BREADTH duty on whichever smoke FORM runs — under a
   unified smoke-IS-sweep architecture, run the one-cell smoke once
   per arm class; under the GPU-bound-phase carve-out below, the
   CPU-runnable portion covers each arm class. Record the coverage on
   one line inside each phase's `### <phase-name>` sub-section —
   `arm classes covered: <list>` (write `single arm class` when the
   driver has one, so a MISSING line reads as a forgotten duty, never
   as "single-arm by design"). Recipe: `.claude/rules/gotchas.md` "A
   single-arm smoke is blind to per-arm seams". The Step 6d.0-bis gate
   reads "once at tiny N" as once PER ARM CLASS and stays the
   downstream backstop (phase-keyed mechanics unchanged) — this
   checklist item is the implementer-side prevention.

   **Plan-§12 structural-assumption probes.** For every plan §12 row
   whose How-to-verify routes to a smoke-slice probe (a real-corpus
   structural premise gating an arm / fit / phase), run the named
   probe at full-CONSUMED-corpus grain during smoke — never only the
   sliced sample — and record the MEASURED value in the phase's
   `### <phase-name>` sub-section under `## Smoke run`. A measured
   violation is a plan defect: surface it (bounce to plan amendment /
   re-scope) BEFORE production, never leave it to the production
   assert (#1768: ~55 min lost mid-run).

   **Data-dependent gates, not just the happy path.** A phase's
   data-dependent gates — fold-skip thresholds (`if kept < K:
   skip/continue`), non-empty-intersection checks (`n_common > 0`),
   shape/count asserts, below-floor `raise` branches — otherwise first
   execute in production on a billed GPU box: the item-5 floors duty
   deliberately sizes the MAIN smoke leg ABOVE every floor so the smoke
   passes, which leaves every gate branch un-executed at smoke n
   (#1345: 4 pre-existing gates in reused code — a `tr.sum() < 3`
   fold-skip, an `n_common > 0` assert, two count asserts — first fired
   in production, two serialized GCP crashes). Per phase: ENUMERATE the
   gates from the same consumer grep the item-5 floors duty already
   runs, widened to gate shapes (`assert `, `raise `, `if len(`,
   `< <threshold>` guarding a skip/continue/raise) in the code the
   phase executes; then DEMONSTRATE each fires once OUTSIDE the main
   smoke leg — a deliberate degenerate-input probe: call the gate's
   enclosing function (unit-level is fine) on an input sized to trip
   it; expected outcome = the gate's DESIGNED handling (a clean
   skip/continue, or its own loud raise — a crash from any OTHER line
   is a bug found, fix it) — or declare it `production-only — <one-line
   reason>` (e.g. the degenerate input is unconstructable without
   GPU-scale artifacts). Record one line per phase sub-section next to
   `arm classes covered:` — `data gates exercised: <gate → probe
   outcome | production-only — reason>`; write `none found` when the
   grep returns none, so a MISSING line reads as a forgotten duty.
   This COMPOSES with the item-5 resize-up duty, it does not reverse
   it: resize-up keeps the MAIN smoke leg passing; this duty
   demonstrates the gate branches execute in a SEPARATE probe — two
   legs, one smoke surface. Step 6d.0-bis gate mechanics unchanged;
   this checklist item is the implementer-side prevention.

   **GPU-bound-phase carve-out.** When a phase requires multi-GPU or
   GPU-mandatory runtime (`accelerate launch` + ZeRO-3, vLLM batched
   eval, ≥7B HF model load in bf16, TP=8 inference) and the local VM
   has no compatible GPU, the smoke for that phase decomposes into
   THREE substitute coverage items — all three are required, not
   alternatives:
   1. **REAL CPU smoke of the CPU-runnable portion of the phase**
      against the real artifact the upstream phase emits — i.e. the
      pre-GPU setup pipeline the production code actually executes
      before the first CUDA call. For training that means: data load
      + tokenizer construction + marker-token id assertion +
      truncation-guard arithmetic + `max_steps` / `num_train_epochs`
      arithmetic + collator construction on a 1-example dataset, with
      exit code 0 and a digest of the produced inputs (row count +
      first-row shape). For eval that means: prompt construction +
      tokenization + sentinel/refusal post-processing on a 1-example
      slice fed through a 2-layer CPU stub model (or a teacher-forced
      log-prob path against a tiny CPU model), with the same digest
      shape.
   2. **Dispatcher dry-run** (`--skip-train --skip-eval` or the
      equivalent flag the project's dispatcher already exposes) that
      exits 0 cleanly and emits the terminal `[phase=done]` log line
      so the cell-iteration plumbing, env passthrough, sentinel
      writer, and `poll_pipeline.py` contract (see the pod-side
      contract section above) are exercised end-to-end without
      requiring a GPU.
   3. **Signature smoke** on the GPU-bound entrypoint:
      `uv run python -c "import inspect; from <module> import
      <fn>; print(inspect.signature(<fn>))"` — catches ABI
      breakage between the dispatcher caller and the trainer / vLLM
      entrypoint (the partial-port crash class the
      "Porting a recipe from an unmerged parent branch" section
      addresses post-launch). The signature must match what the
      dispatcher's call site passes.

   Report this under the relevant phase's sub-heading in `## Smoke
   run` with the literal sub-heading `### <phase-name> — Carve-out
   (GPU-bound)` (e.g. `### training — Carve-out (GPU-bound)`,
   `### eval — Carve-out (GPU-bound)`). Inside that sub-section list
   each of the three substitute coverage items with its command, exit
   code, and one-line artifact digest. Also name the constraint in one
   sentence ("4× H100 ZeRO-3 required; local VM has no CUDA-capable
   GPU"). A phase that is GPU-bound but NOT labeled with the
   `Carve-out (GPU-bound)` sub-heading — or that omits any of the
   three substitute coverage items — is STILL a `smoke-run-missing`
   FAIL at code-review: the carve-out is the documented escape hatch,
   not a default. CPU-runnable phases (data-gen, analysis, upload)
   always use the standard end-to-end smoke shape above — the
   carve-out applies ONLY to genuinely GPU-bound phases. The
   code-reviewer mirror rule lives in
   `.claude/agents/code-reviewer.md` Step 0.6 (incident #514 r2:
   unlabeled "(signature smoke)" notation FAILed `smoke-run-missing`;
   the sub-heading + three-item coverage is the documented escape
   hatch).

   **Plan-declared runtime guards / monitors must show smoke evidence.**
   Every runtime guard, monitor, or trajectory logger the approved plan
   declares as load-bearing — a saturation guard, `MarkerBandStopCallback`,
   per-step log-prob probes, an auto-fired secondary DV, per-source WandB
   run separation — must show concrete evidence in the relevant `## Smoke
   run` sub-section that its telemetry actually functions: the probe logged
   at least one value during the smoke, the guard branch was exercised or
   its precondition assert ran, per-source WandB run names are distinct
   (paste them). "The callback is attached" is NOT evidence — a guard whose
   telemetry never fires is a paper mitigation, and the failure it guards
   is then caught only at eval time after the pod cycle (incident #480:
   the plan-declared WandB trajectory monitor + KL auto-fire silently
   never functioned — 5 of 6 source runs reused one WandB run name,
   per-cell trajectories were never logged, zero saturation markers fired,
   and all 6 adapters shipped saturated). A guard whose telemetry genuinely
   cannot be demonstrated at smoke scale (e.g. it only triggers after
   hundreds of steps) must be called out explicitly in `(d) Needs human
   eyeball` with the reason AND the closest demonstrable proxy (the
   precondition assert ran, the logging call was reached). Code-reviewer
   mirror rule: Step 0.6 FAILs `smoke-run-missing` on missing guard
   evidence with no documented (d) call-out.

   **Smoke outputs never overwrite committed artifacts.** When any smoke
   command — including a revision/follow-up-round smoke on an
   already-produced arm — writes under `eval_results/` or `figures/`,
   divert its output away from the canonical committed paths:

   - **Preferred — scratch-dir redirect.** Pass the script's output
     override (`--out-dir /tmp/issue-<N>-smoke/`, an env var, or its
     `_smoke` path branch) so canonical paths are never touched.
   - **Fallback — restore-after-smoke.** No output override: immediately
     after the smoke, `git -C "$WT" checkout -- <paths>` every touched
     committed path, delete untracked smoke outputs there, and confirm
     `git status --porcelain -- eval_results/ figures/` is empty.

   Either way, the phase's `### <phase>` sub-section in `## Smoke run`
   STATES which mechanism was used (fallback: paste the empty porcelain
   output). Capture the artifact digest BEFORE the restore/delete.
   A dirty worktree of smoke-truncated production JSONs/figures is a
   latent clobber swept into a later explicit-path commit (three #722
   instances, incl. a hero figure shipped at 2-layer smoke scale).
   Corollaries for NEW/EDITED code: (a) a script growing a smoke flag
   diverts ALL outputs together — JSONs AND figures (a partial divert is
   the hero-figure instance); (b) tests never write canonical `eval_results/` /
   `figures/` paths — use pytest's `tmp_path`.

   **`timeout`-bound every smoke; kill any prior instance before a
   re-run.** The Bash TOOL timeout kills the shell and ORPHANS the python
   child — wrap every smoke command in `timeout --kill-after=30s <N>s
   <cmd>` (`<N>+30` ending ≥60 s before the tool timeout) so an abandoned
   smoke self-terminates. Before ANY re-run (revision / crash-fix round,
   same-turn retry): exact-invocation-scoped `pgrep -af` probe → kill →
   confirm-dead, per `.claude/rules/crash-fix-rounds.md`
   § Kill-before-relaunch. NEVER a broad `pkill -f python` on this shared
   VM (incident 2026-07-02: three concurrent #823 smoke instances, same
   output paths, ~1/3 of a load-186 overload).

## After-implementation item 7 detail — raw-completions upload wiring

7. **Raw-completions upload wiring (mandatory when the dispatcher writes
   per-cell completions to disk).** Any pod-side dispatcher that writes
   `raw_completions/*.json` or `raw_generations/*.json` (or any equivalent
   per-cell completion file the eval loop persists locally) under
   `eval_results/issue_<N>/` MUST call
   `explore_persona_space.orchestrate.hub.upload_raw_completions_to_data_repo(
   experiment_name="issue<N>_<slug>", eval_results_dir=Path("eval_results/
   issue_<N>"))` from the dispatcher's normal exit path AFTER the eval
   phase completes and BEFORE the `[phase=done]` log line + final sentinel
   write. Per CLAUDE.md Upload Policy raw completions MUST land on the HF
   data repo before pod termination — the helper is fail-loud
   (`RuntimeError` on any per-file upload failure or HF Hub mismatch), so
   a clean dispatcher exit IS the upload contract; the upload-verifier at
   Step 8 is the safety net, not the only line of defense.

   If the dispatcher walks raw-completion files under a non-canonical
   directory shape that `rglob("raw_completions.json")` does NOT pick up
   (e.g. the dispatcher writes flat per-cell JSONs under
   `eval_results/issue_<N>/raw_generations/<trait>_<arm>_<context>.json`
   rather than `<cell>/raw_completions.json`), EITHER restructure the
   write path to match the helper's recursive `raw_completions.json`
   glob, OR add a small loop that explicitly walks the actual write path
   and calls `hub._upload(...)` per file with `repo_id=
   DEFAULT_DATASET_REPO`, `repo_type="dataset"`,
   `path_in_repo=f"issue<N>_<slug>/raw_completions/<rel>"`, OR (PREFERRED
   over the per-file loop for large file counts — the HF Hub throttles a
   repo at ~256 commits/hour, #591) batch every file into ONE
   `HfApi.create_commit(repo_type="dataset")` whose `CommitOperationAdd`
   ops target the same canonical
   `issue<N>_<slug>/raw_completions/<rel>` paths, then verify the
   per-prefix file count on the Hub (scoped
   `list_repo_tree(path_in_repo=<prefix>)` — bare data-repo
   `list_repo_files` times out, gotchas.md) before
   `[phase=done]`. All three shapes satisfy the reviewer's Step 0.65
   gate (`code-reviewer.md`). Whichever shape,
   the per-cell completion files MUST land on
   `superkaiba1/explore-persona-space-data/issue<N>_<slug>/raw_completions/...`
   under their dispatcher's normal exit path — no "the verifier will pick
   it up" deferrals. Incident: task #528 (2026-06-09) — the i528 pod-side
   dispatcher wrote 160 raw-completion JSONs to
   `eval_results/issue_528/raw_generations/` and never called
   `upload_raw_completions_to_data_repo()`; the upload-verifier caught
   the gap manually, but a verifier that trusted the sentinel without
   re-enumerating would have lost all 160 files on pod termination.

   Confirm the wiring landed by grepping the dispatcher for the helper
   import + call:

   ```bash
   grep -nE "upload_raw_completions_to_data_repo|hub\._upload\(.*raw_completions|create_commit" \
     scripts/run_experiment_<N>.py scripts/i<N>_*.py 2>/dev/null
   ```

   At least one match per dispatcher that writes raw completions; zero
   matches = the contract is missing. Report this in the implementer's
   `## Smoke run` section under a new `### upload wiring` sub-heading
   (one line: the grep command + the matched line, or the literal note
   "no raw completions written by this dispatcher; upload helper N/A").

   **Plan-glob parity self-check (#825).** If any upload call in your diff
   filters eligibility (`upload_folder(allow_patterns=...)` /
   `ignore_patterns=...`, a custom glob/match enumeration, an extension
   allowlist), diff the UNION of those filters against every artifact
   class the plan declares as persisted (§6.5 `primary_deliverable:` rows,
   §10 per-stage output destinations; plan §10 `discarded_artifacts:`
   entries are the only declared-not-uploaded exemption) BEFORE posting
   your marker. A declared class no filter makes eligible = extend the
   filter now (or wire a separate upload for it) — never leave it for the
   upload-verifier (#825: an allow-list of `**/*.npy` + `**/*.json`
   silently excluded 404 plan-declared `row_index*.jsonl` files, 48.9 MB;
   remediation was possible only because the instance was still alive).
   Report the conclusion as one line under the same `### upload wiring`
   sub-heading: "uploader eligibility filters cover all plan-declared
   classes: <globs>", or "no eligibility filter — whole tree uploaded",
   or "N/A — no filtered upload in this diff".
