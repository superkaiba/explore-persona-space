# Verification report — #2394 jailbreak-mining pilot hardened to paper-citable grade (task #2480)

**Closes F10's unpinned-provenance gap** (`docs/paper_context_answer_map/outline_critique_2026-08-22.md:243`): #2394 was
cited at paper grade (Results III, `c5_useful.tex`) from an unreviewed scratch report whose JSONs
lived only on `/mnt/eps-data` staging. This task commits those JSONs to the repo, re-derives every
cited number from the committed copy, and audits the two constructions a reviewer will attack.

**Scope honesty (read this first).** Every number below was re-read from the COMMITTED artifacts and
matched to the scratch report / paper to the quoted precision. This is reproduction-from-the-stored-
artifact, **not** an independent re-computation from raw activations (the two 460 MB `.npz` were not
loaded). It certifies the paper cites what the pipeline actually stored; it does not re-run the
pipeline.

## Provenance (SHA-pinned)

- **Committed result JSONs + provenance logs:** `eval_results/issue_2394/` at commit
  `89a01da7369366db09f2d8d6a3005f8084794fb4` (branch `issue-2480`) — 8 JSONs + 2 logs.
- **Verification + audit script:** `scripts/issue2480_verify.py` @ commit
  `62193fbca265a17007af81c4fefe3489692b1190` (round-2 hardening: audit verdicts DERIVED from
  executable predicates; exact budget/layer universes + value pins; gated toxicchat + transfer
  reads; 3-leg self-test). Re-run:
  `uv run python scripts/issue2480_verify.py` (verification + audit) and
  `uv run python scripts/issue2480_verify.py --self-test` (COLLECT-ALL + derived-verdict tests).
- **Analysis tensors (GPU-costly downstream inputs) persisted to HF:**
  `superkaiba1/explore-persona-space-data/issue2394_jbmine/analysis_tensors/`
  — `evil_compliance_ctxend.npz` (v_C context_end) + `evil_answer_t1.npz` (v_A answer-span), each
  459,833,394 bytes, `list_repo_tree`-verified post-upload (2 entries). Not committed to git.
- **Scratch report under audit:** `docs/scratch/jailbreak_mining_pilot.md` @ `cb1f5f836c` — left
  UNMODIFIED (this task flags discrepancies, it does not edit the scratch report).
- **Ground truth `_meta`:** dv = compliance (StrongREJECT-style); model `Qwen/Qwen2.5-7B-Instruct`
  @ `a09a35458c702b33eeacc393d103063234e8bc28`; v_C = context_end, v_A = t1; ridge λ = 100;
  eval `{n:1960, n_pos:98, base_rate:0.05}`.

## Step B — headline-number verification (COLLECT-ALL): 9/9 reproduce

Semantics: every claim is evaluated and its per-claim verdict emitted before the aggregate; a
reproduction MISS is recorded and the run continues (a miss is not a kill). Nonzero exit is reserved
for missing/corrupt/absent-key inputs. All 9 headline numbers reproduce to quoted precision.
The gate additionally asserts the EXACT quantification universes and value pins (round-2
hardening): budget universe {10, 20, 40, 80, 160, 320} wherever "ALL 6 budgets" is claimed, layer
universe {7, 11, 15, 19, 23, 27} wherever a whole-family extremum is claimed (arm-B max,
benign/in-domain R² ranges), `target == 0.8` exactly, and D_indomain = 50.99 / D_merged = 46.66
pinned at 2 dp — a subset/superset/empty sweep or a drifted D value can no longer PASS.

| Headline (reader-facing) | Committed key path | Stored | Verdict |
|---|---|---|---|
| **probe = oracle (EQUALITY read, MF-C)** — matched-L19 `A_probe_vC` vs `E_probe_vA_oracle` under the SAME `.eval` | `map_arms.layers["19"].{A_probe_vC,E_probe_vA_oracle}.pr_auc` | 0.9736 vs 0.9742 (Δ=0.00067) | **indistinguishable** (Δ inside AP noise at n_pos=98) |
| context probe **0.973** (hardneg pool, L27) — SEPARATE corroboration | `compliance_pilot.hardneg_failcomp_5pct.layers["27"].probe.pr_auc` | 0.9732 | ✓ (different pool n=3000 / different layer; corroboration, not the equality leg) |
| **fixed-direction map-then-project (arm B) ≤ 0.43** (MF-D) | `max(map_arms.layers[*].B_mapproj_{benign,indomain,merged}.pr_auc)` | 0.4252 (L15 indomain) | ✓ ≤ 0.43 (B family ONLY; at L19 the fitted arm C scores 0.945–0.971 and arm D 0.871–0.931 — neither is bounded by 0.43) |
| labels-to-PR-0.80 **≤10 vs 50.99 / 46.66** (L19) | `label_efficiency.layers["19"].n_to_reach_pr` | target=0.8, A=10, D_indomain=50.99, D_merged=46.66 | ✓ (A=10 is the smallest budget swept ⇒ true value **≤10**, an upper bound; the scratch report's "~47–51" band does NOT bracket D_merged = 46.66 — cite the stored values) |
| full-label ref **A ≈ 0.98** (MF-E; `c5_useful.tex` L124) | `label_efficiency.layers["19"].full_label_ref.A` | 0.9782 (n_train=1377) | ✓ |
| full-label ref **oracle ≈ 0.98** (MF-E) | `label_efficiency.layers["19"].full_label_ref.E_oracle` | 0.9828 | ✓ |
| **"ahead at every label budget"** (MF-E; `c5_useful.tex` L131) | `label_efficiency.layers["19"].curves` | A > {D_indomain, D_merged} at all 6 budgets [10..320] | ✓ (assert over `.curves`) |
| benign R² **−0.12 .. −0.88** | `map_arms.map_r2.benign` (6 layers) | min −0.876, max −0.119, mean −0.558 | ✓ |
| in-domain R² **+0.33 .. +0.62** | `map_arms.map_r2.indomain` | min 0.329, max 0.625 | ✓ (regularisation-limited: n_train=1377 < d=3584) |

**Naming discipline (MF-D).** The Goal's shorthand "map arms ≤0.43" means the **fixed-direction
map-then-project (arm B)** only. The fitted map arms are not bounded by 0.43: at L19, arm C
(probe on M·v_C, the reparametrization-identity arm, C ≈ A) scores 0.945–0.971 and arm D
(v_A-probe through M) scores 0.871–0.931. Every reader-facing artifact uses "arm B ≤ 0.43".

**Label-efficiency protocol.** The ≤10-vs-(50.99/46.66) advantage is measured under **5 stratified draws**
(`_meta.pos_frac_in_draw = 0.1`: 2 positive + 8 negative labels at budget 10), so it is conservative
under that protocol, not a natural-prevalence random-labeling estimate. Stored draw variability at
budget 10: A pr_auc_mean 0.834 (sd 0.064), D_indomain 0.619 (sd 0.134), D_merged 0.620 (sd 0.150).

## Step C — two-construction audit (CHANNEL-SCOPED)

Audit-channel verdicts are DERIVED by `issue2480_verify.py` from executable predicates: a False
predicate (counts-vs-`_meta` mismatch, a per-arm override key, absent split evidence, a
toxicchat/transfer read drift) renders that channel FAILED/UNVERIFIED in the script output and
flips the AGGREGATE line — never positive prose over a False predicate. The prose below mirrors
the script's VERIFIED output on the committed data.

### Construction 2 — same-family failed-jailbreak negatives

**Verdict (channel-scoped):** the construction REMOVES the benign-negative and context-identity
inflation channels — the negatives are genuine low-compliance jailbreak-family contexts on a
split-isolated pool. BUT it is an **extreme-groups** design, so the ABSOLUTE 0.973 is measured on a
gap-separated task; that is a SCOPE caveat on the absolute number, not inflation of the relative read.

- **Deterministic rule re-application (0-GPU, npz-free — NO RNG/seed replay).** Nothing in this
  audit consumes a seed: re-applying the producer's selection RULES
  (`scripts/issue1739_jbmine_compliance_pilot.py`) to the committed candidate compliance scores
  reproduces the producer `_meta` COUNTS and thresholds — `n_have` = 9953 DV contexts (evil_train
  7959 + evil_hh_rlhf 1994); the positives rule (compliance mean AND min-over-rollouts ≥ 90,
  top-150 by (min, mean), a deterministic sort) yields **150** (meta 150); failcomp candidates
  (mean ≤ 5.0) = **3787** (meta 3787); the negative COUNT via the producer's arithmetic
  = min(round(150·0.95/0.05), 3787) = **2850** ⇒ eval pool = 150 + 2850 = 3000 at base 0.05.
  Neither pool MEMBERSHIP is replayed: the exact 2,850-of-3,787 negative set is an npz-ordered
  `rng.choice` draw (replaying it requires the plan-forbidden `.npz` read), and the positive
  top-150 sort key ties at (100, 100) across the boundary (266 of the 564 eligible candidates),
  so the exact positive set depends on candidate ordering too. The excluded-middle verdict does
  not depend on either membership — every eligible positive has mean & min ≥ 90 and every
  failcomp candidate has mean ≤ 5, so no middle-band context can enter the pool under ANY
  tie/draw resolution.
- **Excluded-middle read (the reviewer's real attack).** Of the 9953 candidate contexts, **5182 are
  partial-compliers** with compliance mean in (5, 90) — more than half the population — and they are
  EXCLUDED from the eval pool by construction (positives require ≥ 90, negatives require ≤ 5). So the
  absolute 0.973 answers "separate always-comply (≥90) from failed-compliance (≤5)", NOT "detect
  always-comply among ALL same-family jailbreak contexts." State this as a scope caveat on the
  ABSOLUTE number; the probe-vs-map RELATIVE read on the shared pool is unaffected.
- **Split isolation.** Evidence is `map_arms._meta.split` ("per-context groups; MAP reserve /
  probe-train / test mutually disjoint") + `label_efficiency._meta.split` ("label pool == map reserve
  … EVAL disjoint, identical construction+seed to map_arms_results.json"). `compliance_pilot._meta`
  carries NO `split` key (consistency advisory only — it is not the split-isolation source).
- **toxicchat exclusion (stated, not flagged).** The `evil_toxicchat` block in
  `compliance_percontext.json` is EMPTY (0 entries); the producer loads only the shared file, so the
  evaluated pool excludes toxicchat BY CONSTRUCTION. This is expected, not a defect. The toxicchat
  sibling `compliance_percontext_toxicchat_probe.json` (671 contexts — the count is loaded and
  asserted by the executable audit) serves the transfer companion, a DIFFERENT pool.

**Untested residuals (named, not passed):**

- **(a) family-grain leakage — reported, not just pointed at.** The split is per-CONTEXT, so
  same-family contexts can span probe-train and test; per-context grouping is NOT family control
  (#810 LOFO lesson). The committed `transfer_results.json` gives the honest family-grain read at
  the prespecified L19 (the two usable-family directions; evil_toxicchat is `thin_target`, 13 test
  positives, directional only): `evil_train → evil_hh_rlhf` probe transfers **0.894** (vs 0.947
  within-family) but `evil_hh_rlhf → evil_train` drops to **0.623** (vs 0.982 within-family). Cross-
  family transfer is asymmetric and, in one direction, degrades substantially — the per-context 0.973
  overstates family-grain generalization. These four numbers are script-gated:
  `issue2480_verify.py` loads the committed `transfer_results.json` and asserts each at 3 sig figs
  (a drift renders `transfer_famgrain=MISMATCH` in the aggregate). Reviewers should read
  `transfer_results.json` for the full 6-direction sweep.
- **(b) lexical / surface artifact.** No text-only / lexical baseline exists in #2394's artifacts and
  fitting one is out of this task's 0-GPU read-only scope. The claim is DETECTION, not a
  representational-mechanism claim. **F10 stays partially open** for the paper session (a lexical
  control is the residual).
- **(c) phrasing.** "Same-family negatives necessarily make detection HARDER" is stronger than the
  committed diagnostics establish (same-family membership ≠ proven lexical/stylistic matching). The
  verdict is phrased "removes the benign-negative + identity-leakage channels", not "necessarily
  harder".

### Construction 1 — 5% base-rate composition

**Verdict (channel-scoped): NO differential base-rate inflation at the measured 5% prevalence.**

- **Matched prevalence across arms.** `map_arms.eval` is a SINGLE top-level block
  `{n:1960, n_pos:98, base_rate:0.05, k5:98}`; a programmatic scan of every arm dict across all layers
  shows each carries only `{roc_auc, pr_auc, hit@5pct, evals_to_find_20}` — NO per-arm
  `base_rate`/`eval`/`n` override. PR-AUC chance = base rate, applied EQUALLY to every arm on one eval
  block, so a 5% base rate cannot inflate the probe RELATIVE to arm B: the probe-vs-B gap (0.974 vs
  ≤0.43) is measured at matched prevalence.
- **Prevalence-invariant corroboration (ROC-AUC).** The probe-vs-B ORDERING also holds on the
  prevalence-invariant `roc_auc` at L19: probe 0.9964, oracle 0.9983 ≫ arm B max 0.8162. The
  **ordering is base-rate-robust**; the PR-unit gap **MAGNITUDE is 5%-specific** (it compresses at
  balanced prevalence — see below).
- **`balanced_benign` corroboration (VERIFIED from the committed JSON, not scratch-cited).** The
  balanced pool (n=300, base 0.50) is present with all arms: probe max 0.9895; map_then_project max
  0.870; rb_harmcomp max 0.850; rb_refusal max 0.808. (If the map/r_B arms were ever absent, the
  script DEMOTES this leg to "scratch-cited corroboration (UNVERIFIED)" — the plan's MF-F branch,
  now implemented and self-tested — while the matched-prevalence primary carries the verdict.)
  This is CORROBORATION on BENIGN negatives only
  — it changes prevalence AND negative composition at once (a two-variable control), so it is NOT a
  "definitive disconfirmation."
- **Absolute caveat.** PR-AUC at 5% base is a harder metric than at 50%, so 0.973 (≈19× base) is a
  strong absolute result and the low base rate makes it look impressive — but it is not a differential
  confound.

## Metric glosses (for the paper session)

- **`hit@5pct`** = precision at k = n_pos (numerically also recall at that cutoff here).
- **`evals_to_find_20`** = observed ranked-list depth to surface 20 positives (an observed count, not
  an expectation).
- **PR-AUC chance** = the base rate (0.05 here); ROC-AUC chance = 0.5 and is prevalence-invariant.
- **Negative R²** = transfer/reconstruction FAILURE ("worse than predicting the mean"), NOT a graded
  anti-prediction strength. The in-domain R² (+0.33..+0.62) keeps the regularisation-limited
  qualification: n_train=1377 < d=3584 (the merged map at n=4377 > d is the well-posed one).

## F10 status

- **Closed:** unpinned provenance (artifacts now committed + SHA-pinned; GPU-costly tensors persisted
  to HF); every cited headline number reproduces from the committed copy (9/9); both constructions
  carry explicit channel-scoped verdicts with the excluded-middle read and the family-grain transfer
  numbers.
- **Stays partially open for the paper session:** the lexical/surface-artifact control (residual (b))
  — DETECTION is established; a representational-mechanism claim would need a text-only baseline, out
  of this task's 0-GPU scope.
