---
title: 'Auditing every git-tracked vLLM-LoRA eval driver for the cache-collision bug
  finds one new affected published result — #504''s saturated-anchor geometry was
  measured on the step-6 adapter, not step 25 — and the bug is still live on main
  (HIGH confidence)'
kind: analysis
tags: []
created_at: '2026-06-10T06:35:53Z'
has_clean_result: true
parent_id: 534
---
# Auditing every git-tracked vLLM-LoRA eval driver for the cache-collision bug finds one new affected published result — #504's saturated-anchor geometry was measured on the step-6 adapter, not step 25 — and the bug is still live on main (HIGH confidence)

<!-- clean-result-v2 -->

## Human TL;DR

**Headline.** the repo-wide cache-collision audit found exactly one published result that consumed wrong-adapter evals: the saturated-anchor geometry numbers (#504, labeled "step 25") were actually measured on the step-6 adapter — and the underlying bug is still unfixed on main.

**Takeaways.**
- vLLM caches LoRA adapters by integer id only. loop over checkpoints with one id and every read after the first silently comes from the first checkpoint — no error, no warning.
- 24 driver-and-run rows audited: 2 affected (one was the bug's own discovery run, already recovered before publication; one is new), 19 safe with named evidence, 3 never ran. nothing else published consumed a stale read.
- the affected numbers are real measurements of a genuinely saturated adapter — just at step 6 instead of step 25. the corrected story is more dramatic, not less: that learning rate saturated the bystanders within 6 training steps.
- the fix exists on a branch but never merged. any new multi-checkpoint trajectory eval launched from main re-hits the bug, so porting the fix is the top follow-up.

**How this updates me.** i now trust the published numbers outside that one grid — the audit is exhaustive over git-tracked vLLM drivers and the clean verdicts have named evidence. before promoting the affected write-up i need its maturity labels corrected, and i should treat "the calibration table looks flat across checkpoints" as a red flag worth chasing, because that flatness was sitting in the run's own committed artifact the whole time.

*(First pass — Thomas refines this before sending to the mentor.)*

## TL;DR

### Motivation

[#534](https://eps.superkaiba.com/tasks/534) found a quiet failure mode in our vLLM eval stack: the LRU adapter cache keys strictly on `lora_int_id`. A request whose id is already registered is "just touched" (the adapter path is never re-read), so an eval loop that reuses one id across different checkpoint paths silently serves the first-loaded weights for every later checkpoint. That run caught its own corruption and re-ran everything before publishing, but the same code pattern had been live in the repo for weeks, and a note from that era extrapolated "other headline results look unaffected" from a single spot check ([#530](https://eps.superkaiba.com/tasks/530)) without auditing the rest. The goal was to enumerate every historical eval that served LoRA adapters through vLLM, decide for each realized run whether it could have read the wrong adapter, and flag any published number that did.

### What I ran

A CPU-only audit, no training and no generation. I enumerated every git-tracked driver that constructs vLLM adapter requests: a grep at HEAD, a full-history pickaxe, a sweep over all branch tips, and hardened cross-greps (`enable_lora`, harness-style `lora_path=`, one repo-wide pass). That surfaced roughly 40 driver scripts, collapsing to 24 driver-and-realized-run rows once grouped by shared structure.

For each realized run I reconstructed the exact (id, path) request sequence from three sources: the driver code at the commit the run actually launched from, the launch markers, and the artifact metadata (per-cell git commits and timestamps). Each sequence then replayed through a small deterministic simulator that mirrors vLLM 0.11.0's cache semantics line-by-line (id hit means touch-without-reload; capacity is 1 by default), validated by 7 internal self-checks and calibrated on four controls with known ground truth (one documented stale-serve, three known-clean runs). It classified all four correctly; the affected class is attested by a single known case, so the calibration is stronger on the clean side. Verdicts come from a closed eight-value set; every row cites its evidence as file-and-line at a commit for code facts, artifact paths or launch markers for run facts, and replay output for cache-residency claims.

Scope: git-tracked vLLM-LoRA drivers only. In-process adapter swapping through PEFT (which keys on paths, not ids), merged-weight reuse, never-committed pod-local scripts, and non-vLLM harness defects are out of scope by design.

<details open>
<summary>3 example audit rows (driver → reconstructed request sequence → replay at capacity 1 → verdict)</summary>

| Driver | Reconstructed request sequence | Replay at capacity 1 | Verdict |
|---|---|---|---|
| trajectory eval over a 6-checkpoint grid (`eval_trajectory.py` via its per-cell dispatcher) | one engine; id 1 repeated over training fractions 0.08 to 1.0 | 5 of 6 stale; every request served the first-loaded adapter | Affected |
| cross-epoch predictor sweep (`issue532_predictor_stress.py`) | one engine; 16 distinct ids (epoch 1), then 6 repeated ids (epoch 2) | 0 stale; each repeat had 15 intervening loads that evicted it | Safe by eviction |
| single-adapter panel evals (about 20 scripts, grouped into one row) | one engine, one adapter, constant id | no repeat possible; the first load is always correct | Safe, single adapter |

Full table: `eval_results/issue_549/audit_table.json` (24 rows, committed on branch issue-549; lands on main at the task's auto-merge).

</details>

### Findings

#### The saturated-anchor geometry grid was measured on the wrong checkpoint

The saturated-anchor geometry result ([#504](https://eps.superkaiba.com/tasks/504), awaiting promotion) published partial correlations and bystander-saturation numbers labeled "checkpoint fraction 0.33 (step 25 of about 75)". Its eval driver consumed a 6-checkpoint index through one engine under a constant adapter id, so the audit's first question was which weights those reads actually came from. Four evidence legs, three independent in kind, give the same answer: every read came from the fraction-0.08 (step-6) adapter.

First, the code at the run's launch commit (`affdd82cb`) constructs one `LLM(max_loras=1)` before the checkpoint loop and `LoRARequest(..., lora_int_id=1, ...)` inside it. Second, the deterministic replay of that sequence marks 5 of 6 requests stale, every one served by the first-loaded fraction-0.08 adapter. Cherry-picked for illustration, the first two stale rows of the six-request replay (full evidence in `eval_results/issue_549/audit_evidence/i504_phase1_stale_serve.json`):

```
{"id": 1, "requested": "checkpoints/frac_0.16", "served": "checkpoints/frac_0.08", "stale": true}
{"id": 1, "requested": "checkpoints/frac_0.33", "served": "checkpoints/frac_0.08", "stale": true}
```

Third, the artifacts carry the signature of one set of weights generated six times: the rate at which base-model log-probs agree exactly between any two "checkpoints" is flat at every fraction distance in all 10 cells (mean adjacent rates 0.186 to 0.265 vs mean distant rates 0.193 to 0.256; no gradient), and the published per-cell source numbers are float-identical to the stale-served "fraction 0.33" trajectory entries in 8 of 8 regression cells (first cell: 6.550999160204265 in both files). Fourth, and this is the leg that upgrades the claim from inference to measurement: the run's own committed calibration artifact (`eval_results/issue_504/phase0_calibration_v4.json`) already showed it. The source implant strength (the marker's log-prob gain over the base model, in nats) across the six nominal fractions:

```
ckpt_frac:  0.08  0.16  0.33  0.50  0.75  1.00
source implant strength (nats):  5.47  5.89  5.01  6.08  5.63  7.17
```

Five of the six reads sit in a 1.1-nat band (5.0 to 6.1); the fraction-1.0 read, 7.17, is the one outlier, and even it is nowhere near the gradient a real 12-times-longer training sweep produces. The sequence is non-monotone and the chosen fraction 0.33 is the lowest of the six. The bystander-resolution read agrees to the last float (0.3296296296296296, with 178 of 540 in band) at fractions 0.08 and 0.16. A real sweep trains 12 times longer and changes something; two "different checkpoints" agreeing to the last digit means the calibration compared six reads of one adapter. The alternative, checkpoints that converged early and happen to look identical, is closed empirically: the post-fix re-run with distinct ids resolves checkpoint maturity strongly (source implant strength rising from −0.03 to +0.35 to +2.19 to +5.96 nats across fractions 0.25 to 1.0 on four distinct snapshots), with the caveat that the re-run used the de-saturated training recipe, so it shows the eval rig differentiates checkpoints generally; for this grid the code read and replay remain the primary legs.

What this does and does not invalidate: the published values are measurements of a saturated adapter, just the step-6 one. The partial correlations (shadow angle +0.335, nearest-neighbor distance −0.342) and the 91–96% bystander saturation all describe the step-6 adapter, not step 25. The qualitative "saturated anchor" claim survives. Every training-maturity label is wrong, and the calibration phase's "choice" of fraction 0.33 was vacuous by construction: it picked among six reads of one identical adapter, and the chosen read was the lowest of the six. The corrected label also pins down one new fact: that learning rate saturated the bystanders within 6 steps. A process lesson worth naming: this flatness was visible in the run's own committed artifact at the time, and nobody flagged it. The proposed correction (not executed; this audit does not edit published bodies) is a re-eval at true fractions through the fixed code path: about 4–6 GPU-hours eval-only if the per-fraction snapshots resolve on the HF model repo, else about 10 GPU-hours to retrain and re-eval.

#### One new hit across 24 rows: the audit-wide verdict matrix

The deliverable is the verdict table itself: 24 driver-and-realized-run rows covering every git-tracked vLLM-LoRA driver, each with a verdict from a closed eight-value set and cited evidence. Per the plan, the audit carries no measurement figures; the chart below just summarizes the counts.

![Horizontal bar chart of audit verdicts across 24 driver-and-realized-run rows: 11 safe via distinct ids, 4 safe via single adapter per engine, 3 never run, 2 affected by stale serving, 2 safe via quantified collision bounds, 1 safe via fresh process per cell, 1 safe via cache eviction.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/1a8f6c52416c6c6d1b8780eff9a07107a8da2d7c/figures/issue_549/verdict_counts.png)

> **Figure.** *Across 24 driver-and-realized-run rows, 2 were affected, 19 are safe with named evidence, and 3 were never run; 0 landed indeterminate.* Bars are exact counts from an exhaustive enumeration of git-tracked vLLM-LoRA drivers (no sampling, so no error bars). Red marks the two affected rows; gray marks drivers with no realized run to audit.

The figure and the table below carry the per-class counts. Those counts are ROWS; counting DRIVERS, the never-run set numbers 4, not 3: one never-run driver (zero artifacts on any ref) rides inside a grouped safe row, where there is no realized run to be affected. The second affected row is the bug's own discovery run: the documented positive control, recovered before publication by re-running all 40 cell-fraction evals at the fixed commit, so its published numbers come exclusively from the re-run.

| Verdict | Rows | What it covers | Basis |
|---|---|---|---|
| Affected | 2 | the saturated-anchor 6-checkpoint grid; the bug's discovery run | constant id over a multi-checkpoint loop in one engine; the discovery run recovered pre-publication, the grid did not |
| Safe: distinct ids | 11 | cross-evals, on-policy phase-4 evals, marker-spread evals, an in-flight panel | one id per adapter within each engine; several rows verified from committed id manifests (e.g. a 72-of-72-unique manifest artifact) |
| Safe: single adapter | 4 | about 20 single-adapter eval scripts (grouped) plus per-task diagnostics | one (id, path) per engine lifetime; the only possible repeat is same-id-same-path, a benign cache hit |
| Safe: fresh process | 1 | the subprocess dispatchers | construct no adapter requests; every cell gets a fresh eval process, so exposure is governed by the per-cell index the eval rows already cover |
| Safe: by eviction | 1 | the cross-epoch predictor sweep | repeated ids were evicted before recurrence (next finding) |
| Safe: probabilistic | 2 | two drivers that derive adapter ids by hashing | operative stale-serve bounds 5.5e-4 and 8e-4 (a collision must also be consecutive at capacity 1); hash randomization makes the realized ids unreproducible after the fact, so a bound is the only available ceiling |
| Never run | 3 | two never-launched drivers and one work-in-progress script | zero launch markers and zero artifacts on any ref |

The "0 indeterminate" hides a residual. The two hash-id rows are safe probabilistically with a summed operative bound of about 1.4e-3, and one distinct-id row rests on code structure plus calibration with its per-cell artifacts not relocated during the audit (the id scheme yields distinct ids for any realized subset, so no artifact could flip it unless the launcher overrode ids; the row carries the flag). Scope caveat carried from the design: "every git-tracked vLLM-LoRA driver" excludes PEFT in-process adapter swapping, merged-weight reuse, never-committed pod-local one-offs (invisible to any git dragnet by construction), and other harness defect classes.

#### Cache eviction protected the cross-epoch sweep

The localization-arm predictor sweep ([#532](https://eps.superkaiba.com/tasks/532)) reused adapter ids across epochs, the same pattern that bit the affected grid, which made it the audit's most at-risk clean candidate. It survived, and the replay shows why. The surviving artifacts (all 556 cells carry one git commit, timestamps spanning about an hour) collapse the run to one artifact-relevant process, whose 22 adapter loads (16 distinct epoch-1 ids, then 6 epoch-2 repeats) replay to 0 stale serves at cache capacity 1, because each repeated id had 15 intervening distinct loads that evicted it before it recurred. A counterfactual replay at capacity 16 yields 6 stale serves: the safety is genuinely by eviction, not id discipline.

The empirical leg reads best calibrated against its own noise floor. Per-cell maximum absolute log-prob differences between epoch-1 and epoch-2 reads, over 140 epoch-2 cells (7000 strictly-paired question rows):

```
min 9.1e-6   p25 4.76   median 6.62   p75 9.64   max 20.59   (nats)
```

The minimum cell shows the stored per-question log-prob computation is deterministic to about 1e-5 given fixed weights. Under stale serving all 140 cells would read near 1e-5, whereas 75% read above 4.7 nats: roughly six orders of magnitude between the stale-serve prediction and the observed data. One qualification this audit adds to the record: the audit design's text-identity threshold (0.9 exact-match for "affected") was mis-calibrated and the observed identity rate of 0.151 is not discriminating on its own. The affected grid's evidence shows same-weights vLLM regeneration reproduces identical outputs only about 19–27% of the time under batching nondeterminism, so a stale-served run would also sit far below 0.9. The replay carries the verdict; the string test only corroborates it.

Confidence splits by leg. The published epoch-1 headline is solid independently of the eviction argument, because its 16 ids are distinct and first-loaded, so those numbers could not have been mis-served at any cache capacity. The epoch-2 scope caveat is the moderate leg: the question-order pairing is pinned at code level rather than per-question text, one saturated diagonal cell is individually indistinguishable (it is itself the noise-floor calibration), and the diagonal dose-direction read rests on the sign of a near-zero median (about 2e-6 nats); the carrying direction check is the off-diagonal (−0.72, matching the reference run's epoch-2-lower direction).

#### The bug is still live on main

Re-verified at the time of this write-up, at main HEAD `21614a3b3` (which postdates the audit's recorded HEAD and both review rounds' checks): the fix commit `298877f9c` is not an ancestor of main, and the trajectory-eval driver still constructs the adapter request with a constant id inside its checkpoint loop:

```
lora_req = LoRARequest(lora_name=label, lora_int_id=1, lora_path=adapter_path)
```

Any future multi-checkpoint trajectory eval launched from main silently reproduces the failure documented above. This is the one output that requires a code change: port the fix to main and add a regression guard (a follow-up code task; editing eval drivers is outside this audit's write scope).

## Reproducibility

**Parameters:**

| Field | Value |
|---|---|
| Task kind | analysis — CPU-only audit; no training, no generation, no judge calls |
| Audited semantics | vLLM 0.11.0 `LRUCacheWorkerLoRAManager` (`lora/worker_manager.py` lines 240–267: id hit means touch, path never re-read; capacity = `max_cpu_loras` = `max_loras` = 1 when not overridden, as in every audited driver; ids below 1 raise at request construction) |
| Simulator | `scripts/i549_lru_simulate.py` — deterministic replay, 7 of 7 self-checks pass |
| Calibration | 4 of 4 known-ground-truth controls classified correctly through the same simulator used for production verdicts |
| Enumeration | grep at HEAD + full-history pickaxe + all branch tips + `enable_lora` / `lora_path=` cross-greps + one repo-wide grep; about 40 drivers, 24 driver-and-realized-run rows |
| Verdict enum | closed 8-value set: Affected / Safe-distinct-id / Safe-single-adapter / Safe-fresh-process / Safe-by-eviction / Safe-probabilistic / No-realized-run / Indeterminate |
| Learning rate / seeds | n/a — no training in this task |
| Hardware | orchestration VM (CPU only); no pod provisioned |
| Wall time | single working session, 0 GPU-hours |

**Artifacts:**

- Verdict table: `eval_results/issue_549/audit_table.json` (24 rows) — committed at commit `2c83c2a6846d566f7ea6ffe26b8b514283bbcabb` on branch issue-549; lands on main at the task's auto-merge.
- Evidence bundle: `eval_results/issue_549/audit_evidence/` (12 files: dragnet outputs A1–A4, per-driver code-context extracts B0_*, replay/comparison JSONs `i504_phase1_stale_serve.json`, `i532_per_process_replay.json`, `i532_ep1_ep2_comparison.json`, and `calibration_controls.json`) — same commit and branch.
- Hero-figure source: [figures/issue_549/verdict_counts.png](https://github.com/superkaiba/explore-persona-space/blob/1a8f6c52416c6c6d1b8780eff9a07107a8da2d7c/figures/issue_549/verdict_counts.png) (+ PDF + meta.json sidecar at the same commit).
- Inputs read (audit subjects — prior tasks' artifacts already in the repo or on the Hub, not reused training artifacts): `eval_results/issue_532/per_cell/` (556 cells), `eval_results/issue_504/phase0_calibration_v4.json` and `eval_results/issue_504/analyze_summary.json`, `eval_results/issue_534/c504v3_*/trajectory.json` (10 cells), `eval_results/issue_530/c504v3_*/trajectory.json` (10 cells), and the HF data repo's `issue504_geometry/phase1_trajectories/` (10 cells; per-cell metadata records launch SHA `affdd82cb`).
- Review trail: three code-review CONCERNs were deferred as non-blocking (the consumption check computes but does not assert; the audit-table rows are hand-built rather than derived from the evidence JSONs; the cross-epoch question-order pin is code-level, not text-level). They were bypassed for this write-up by re-extracting every load-bearing number directly from the committed evidence JSONs and primary artifacts rather than relying on the scripts' internal assertions.

**Compute:** 0 GPU-hours. CPU-only on the orchestration VM; no pod was provisioned for this task.

**Code:** branch issue-549, commits `00184f48bb857e939927520ac3146460c902bcdb` (simulator + per-run audits + calibration controls) and `2c83c2a6846d566f7ea6ffe26b8b514283bbcabb` (verdict table + evidence). The five scripts, pinned:

- [scripts/i549_lru_simulate.py](https://github.com/superkaiba/explore-persona-space/blob/2c83c2a6846d566f7ea6ffe26b8b514283bbcabb/scripts/i549_lru_simulate.py)
- [scripts/i549_calibration_controls.py](https://github.com/superkaiba/explore-persona-space/blob/2c83c2a6846d566f7ea6ffe26b8b514283bbcabb/scripts/i549_calibration_controls.py)
- [scripts/i549_audit_532.py](https://github.com/superkaiba/explore-persona-space/blob/2c83c2a6846d566f7ea6ffe26b8b514283bbcabb/scripts/i549_audit_532.py)
- [scripts/i549_audit_504.py](https://github.com/superkaiba/explore-persona-space/blob/2c83c2a6846d566f7ea6ffe26b8b514283bbcabb/scripts/i549_audit_504.py)
- [scripts/i549_build_audit_table.py](https://github.com/superkaiba/explore-persona-space/blob/2c83c2a6846d566f7ea6ffe26b8b514283bbcabb/scripts/i549_build_audit_table.py)

Reproduce (each command exits 0 and rebuilds its artifact; vLLM pinned at 0.11.0 via the repo's `uv.lock`):

```bash
uv run python scripts/i549_lru_simulate.py --self-test        # 7/7 checks pass
uv run python scripts/i549_calibration_controls.py            # 4/4 control groups pass
uv run python scripts/i549_audit_532.py && \
  uv run python scripts/i549_audit_504.py && \
  uv run python scripts/i549_build_audit_table.py             # rebuilds the 3 evidence JSONs + 24-row table
```

Success looks like: `eval_results/issue_549/audit_table.json` exists with verdict counts {Affected 2, Safe-distinct-id 11, Safe-single-adapter 4, Safe-fresh-process 1, Safe-by-eviction 1, Safe-probabilistic 2, No-realized-run 3}, and the affected grid's consumption check shows 8 of 8 float-identical values.
