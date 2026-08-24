# #2544 — rig-reuse audit (orchestrator, pre-planning)

Written at `/issue 2544` Step 2 entry, before the planner is dispatched. Purpose: turn
the task body's prose reuse claim into a checked, file-and-line-level statement of what
is genuinely configuration and what is new code, so the planner sizes implementation
against measured facts rather than an impression.

Every claim below was verified live against the repo and the Hub on 2026-08-24.

## Verified present

**Pipeline (all on disk, `scripts/`):** `issue1902_common.py` (19 KB),
`issue1902_corpus.py` (39 KB), `issue1902_run.py` (93 KB), `issue1902_dispatch.sh`,
`issue1902_fits.py` (106 KB), `issue1902_figures.py` (34 KB). Plus follow-up rigs that
may hold reusable pieces: `issue1902_ladder_followup.py`, `issue1902_metric_ladder_ext.py`,
`issue1902_steer_probe.py`, `issue1902_smoke_fixture.py`, `issue1902_fig_9ater.py`,
`issue1902_followup_9ater.py`, `issue1902_preimage_check.py`.

**Shared analysis helpers:** `analysis/mapping_baselines.py` (identity+bias, kNN
retrieval — both mandated per cell), `paired_ci.py`, `null_battery.py`,
`vectorized_mlp_skill.py`.

**Corpus on the HF data repo** at `issue1902_stage_map/corpus/`: `corpus_single.jsonl`,
`corpus_multi.jsonl`, `clusters.json` (the cluster assignment the plan's cluster-grouped
6-fold CV needs), `manifest_stats.json`.

**Model ladder on the Hub:** `allenai/Olmo-3-1025-7B` carries 1,487 branches — exactly
the body's stated inventory. All 12 named base-ladder revisions resolve; zero missing.
All three post-training repos exist.

## Corpus sizing — settles planner open-decision 3

`manifest_stats.json` gives `n_single = 18000` (generic 16,000 · mathcode 1,200 ·
gsm8k 500 · mbpp 300) and `n_multi = 16000`. The stream scanned 71,284 rows to fill it.

With d = 4,096 and cluster-grouped 6-fold CV, n_train = 5/6 of the surviving set, so
clearing the n_train >= d floor requires the shared 15-rung non-degeneracy intersection
to retain **>= 4,916 of 18,000 rows — a 27% survival floor**. #1902 intersected 18,000
down to 16,391 (91%) across four checkpoints. Fifteen rungs including a random-init
floor control will cut deeper, and rung 0 in particular may retain almost nothing.

The planner should therefore state a pre-launch gate, not a post-hoc report: measure the
realized intersection at the P1 pilot, and if it falls below the floor either widen the
corpus (the builder is on disk and the source quotas were not exhausted — the stream
rejected 32,735 rows to `single_quota_full`, so more rows are available without changing
the recipe) or declare rung 0 exempt from the shared intersection and report it against
its own row set with the denominator change stated. Note that excluding a rung from the
shared intersection weakens control 4 (differential row dropout) exactly where the
identity-baseline read matters most, so the widen-the-corpus branch is preferable.

## The reuse claim is accurate about mechanism, optimistic about constants

The body says the pipeline "is parameterized on a checkpoint list plus a revision-pin map
and takes `revision=` on every model and tokenizer load, so the ladder is a configuration
change." The mechanism half is TRUE and is the load-bearing half:

- `resolve_revision(ckpt, pins)` / `load_revision_pins()` / `revision_pins_from_report()`
  exist in `issue1902_common.py` and the binding pins come from the P1 pilot report.
- `revision` appears at 19 sites in `common`, 37 in `run`, 5 in `fits`.
- Critically, **every load site pairs `MODEL_IDS[ckpt]` with `resolve_revision(ckpt, pins)`**
  (`issue1902_run.py:842`, `:970`, `:1464`, `:1705`). The checkpoint TOKEN is the
  discriminator throughout — not the model id. So twelve rungs sharing one repo id at
  twelve different revisions is safe by construction, which is the single biggest risk
  this design could have carried. No cache dir or store path is keyed by model id.

The constants half is 4-wide and needs a bounded widening:

1. `CKPTS: tuple[str, ...] = ("B", "S", "D", "R")` (`issue1902_common.py:44`) — a fixed
   4-tuple. `default_revision_pins`, `revision_pins_from_report` (which FAILS LOUD on any
   missing key) and the dims loop at `:122` all iterate it. Widening to 15 tokens is
   mechanical, and the fail-loud pin validation is a feature here: a missing rung pin
   cannot slip through.
2. `MODEL_IDS` (`:46`) — currently four distinct OLMo-2 repos. Becomes twelve entries
   pointing at `allenai/Olmo-3-1025-7B` plus three post-training repos.
3. **A hardcoded token literal at `issue1902_run.py:2014`:**
   `dims = C.model_dims(C.MODEL_IDS["R"], C.resolve_revision("R", pins))`. If the new
   ladder renames tokens (`r0`..`r14`), this raises `KeyError` at whatever phase reaches
   it. `:2207` (`C.MODEL_IDS[m]  # fail loud on an unknown checkpoint token`) is the
   deliberate fail-loud guard and is fine. Two dispositions, both acceptable: keep a
   token named `R` in the new ladder, or parameterize the literal. Whichever the
   implementer picks, the round owes the whole-tree symbol grep
   (`.claude/rules/crash-fix-rounds.md` § symbol-rename whole-tree grep duty) — sibling
   scripts drift until a later phase invokes them, and this rig has eleven siblings.
4. The smoke remap (`_SMOKE_MODEL_DIR`, `:58-60`) rebinds every checkpoint to one tiny
   local model over the real vocab. It comprehends `CKPTS` generically, so it widens for
   free — but note it makes all 15 rungs identical under smoke, so the smoke cannot
   exercise cross-rung difference. That belongs in the plan's smoke blind-spot
   enumeration (`.claude/rules/smoke-blind-spots.md`), which is a required plan section
   and where a substituted implementation must be disclosed.

## Genuinely new code (not configuration)

- The k-shot render and its **query-block-only pooling window**. The body notes the rig
  already separates context-window from prefix-window pooling, so this is closer to
  configuration than authoring — but it is the arm most likely to be silently wrong
  (pooling over exemplar tokens injects a constant into every context vector and moves
  R-squared mechanically), so it needs an explicit assertion, not a config flag alone.
- Per-rung degeneracy diagnostics (repetition rate, distinct-token ratio, answer-cloud
  effective rank and mean pairwise cosine, answer length, generation-cap-hit fraction).
- The shared non-degeneracy intersection across 15 rungs (#1902's version was 4-wide).
- The reduced cross-grid cell roster (43 cells: diagonal + one column + one row) in place
  of #1902's full 4x4 square.
- Per-rung repeat-generation noise ceiling on a pinned 1,000-row subset.

## Standing constraints the planner must carry

- **Ridge only.** Settled by the user 2026-08-24. No MLP, no nonlinear readout, in any
  arm. #1902 having run both is explicitly not a precedent.
- **Never pure GCV at n_train < d** (#1887). The shared fit cores enforce a dof cap by
  default; report the per-fit selector and selected lambda alongside every ridge read.
- **Both baselines in every cell** — identity+learned-bias and kNN retrieval with chance
  stated. They dissociate from R-squared in both directions (#722, #779), and for rungs 0-2
  they are the primary read rather than a formality.
- **Cap-hit fraction** is reported per generation pass with a pre-registered re-generation
  trigger, per the standing generation-cap rule.
