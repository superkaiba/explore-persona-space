# Issue #2162 — CRITIQUE round 1 unioned findings (plan v1 → v2)

Panel: `statistics-critic` REVISE (1 blocker) · `methodology-baselines-critic` REVISE (2 blockers) · `efficiency-critic` APPROVE (0 blockers, 5 concerns) · `consistency-checker` WARN (0 BLOCK, 3 observations). Codex twins were confirmed no-shows (org quota exhausted until 2026-09-05), so each lens verdict is single-Claude and binding.

Plan under revision: `tasks/approved/2162/plans/plan.md` (v1). Manifest: `tasks/approved/2162/artifacts/planned_manifest.json`.

**Both `verify_plan.py` WARNs were adjudicated AWAY by the lenses that own them — do NOT "fix" either:**
- `verdict-lattice coherence` — FALSE POSITIVE. The §3 lattice is already an explicit disjoint-and-exhaustive iff-partition. The checker keyed on "wins" at plan lines 59/162, which are route-conflict SCIENTIFIC PREDICTIONS about a continuous balance-shift DV, not verdict labels. Leave the lattice alone.
- `reused-artifact fitness attestation` — PREMISE INACCURATE. §10 already carries the (h) legs for the WildChat bank, per-module item-(i) verdicts, item (j) N/A, item (k) with the commit SHA + re-check commitment, and item (l) with the `f_beh` boundary engaged; (a)–(g) are structurally N/A (no trained artifact reused). Optional one-line clarity add only: state "(a)–(g) N/A — no trained artifact reused; producing issue #2094 live, not retracted".

---

## BLOCKER 1 (methodology, must-fix) — the shuffled-donor null is contaminated BY CONSTRUCTION for binary-contrast types

**Defect.** §4.1 defines binary types as "3 distinct phrasing instantiations of the SAME A/B contrast" (e.g. `constraint_knowledge` = three wordings of has-internet vs no-internet). §4.2 arm 2 then draws the null target `V_slot(B_donor, ℓ)` from a DIFFERENT pair of the SAME type-cell, "constrained to a different value-PAIR where the pool allows". The constraint is on the value-PAIR, not on the donor VALUE. For every binary-phrasing type, all value-pairs share the same semantic B value, so the "null" installs a PARAPHRASE of the steered target. If the position genuinely carries the information, `F_null ≈ F_steered` and the type can NEVER separate. For 3-value types under lexicographic direction, recipients (v1,v3) and (v2,v3) each have exactly one legal donor pair sharing B=v3 — contaminating ~1/3 of null rows.

**Why conclusion-changing.** The registered verdict is the IUT `p = max(p_shuffled, p_crosstype)` with disjoint CIs against BOTH nulls (§3, §6), and Stage-2 survivor selection gates on the same test (§6 gate 5). So genuinely-carried binary types get systematic FALSE "not carried" reads, the §3 lattice mis-classifies them as "stored-but-unusable" (probe-positive + causal-negative), and they are excluded from Stage 2. NOT recoverable post-hoc: for binary types no valid same-type different-value donor exists in the bank at all, and re-registering the null after the run is precisely the post-hoc-stratification sin the task body exists to correct.

**Corroborating precedent (the parent already treats content-difference as the null's DEFINING property).** `experiments/issue2094/bank.py` L312–340 `donor_derangement` constrains matched-query donors cross-prefix-pair because "a same-prefix-pair donor's prefix-end Δ duplicates the recipient's"; L380–400 `type_b_donor_delta` uses the OTHER prefix's centroid.

**Required fix (all three parts):**
- (a) Constrain the derangement on donor-B-**VALUE** ≠ recipient-B-value, not donor value-pair.
- (b) Per type, where the construct genuinely admits 3+ distinct VALUES (format: bullets / prose / a third format; language: 3 languages), build distinct value SETS instead of 3 phrasings of one contrast.
- (c) For irreducibly binary types (`constraint_knowledge`, `refusal_boundary`, the conflict cells), pre-register NOW — before the run — one of two options.

**ORDERING CONSTRAINT ON (c) — read this carefully.** The task Goal's registered criterion is that a positive "clears BOTH nulls". A cross-type-only single-null test for some types DEVIATES from that criterion and would narrow what the user approved. Therefore:
1. STRONGLY PREFER the criterion-preserving routes: fix (b) so the type has genuinely distinct values (then the same-type null is valid), or the directional A-side donor with its changed meaning stated explicitly.
2. Use the cross-type-only single-null variant ONLY where the construct is irreducibly binary AND no valid distinct-value or A-side donor exists. If you must use it for ANY type, say so EXPLICITLY and prominently in a new "Goal-criterion deviations" line in §12, naming each affected type — the orchestrator must evaluate whether that trips the material-design-change resurface trigger before the plan advances.

**Mechanizable acceptance:** a committed bank unit test iterating the REALIZED derangement and asserting donor-B-value ≠ recipient-B-value per assignment, plus a per-type null-mode declaration in `bank_manifest_2162()`.

---

## BLOCKER 2 (methodology, must-fix) — the uniform 3-direct + 9-neutral carrier split makes item/fact types unreachable BY ARITHMETIC

**Defect.** The separation exclusion is per pair on its own anchors (§6 exclusion 1). For fact/list types, a neutral WildChat carrier's ceiling (generate-under-B) and floor BOTH express neither the name/date/list-item, so `Δ_ceiling ≈ Δ_floor ≈ 0` and the pair is excluded. Only type-ENGAGING pairs can survive. Max separable pairs for such a type = 3 value-pairs × 3 direct-probe carriers = **9** — below BOTH the pre-registered survival floor (≥12, §6) and the exact signed-rank attainability floor (n ≥ 11 at α′ = 0.05/31, §6). The registered confirmatory test is therefore unreachable by arithmetic for exactly the types **prediction 2** lives on. Same structure hits the conditional-policy types: `refusal_boundary` only manifests on boundary-engaging queries, `constraint_knowledge` on lookup-dependent ones.

**Why conclusion-changing.** The headline policy-vs-item contrast — half the core claim — collapses to one-sided ("policy transfers" but "items untestable"), and the 2×2 mislabels item types. Three internal contradictions confirm the defect is real: the plan's own MDE remedy (24→36 pairs) cannot reach these types because every added pair is a neutral-carrier pair that cannot survive; the plan's own gate-3 arithmetic puts 25% survival (exactly the 9/36 direct-only outcome) in the "catastrophic" detection band (§7 gate 3); and the body's own type-14 definition ("stated, directly queried") contradicts giving that type 9 carriers on which the fact is irrelevant.

**Required fix.** Per-type-CLASS carrier allocation. Item + conditional-policy types get a MAJORITY of type-ENGAGING carriers — more direct probes, or WildChat queries FILTERED for engagement (e.g. medical-adjacent for `refusal_boundary`) — sized so best-case survivors ≥ the n ≈ 27 target that yields MDE 0.20 (hard minimum: ≥12 floor, with the realized-MDE caveat carried). Keep shared neutral carriers only to the extent the cross-type-donor matching and the leave-one-carrier-out folds require them. The plan's own `icl_task_mapping` exception (§4.1) already demonstrates per-type carrier flexibility, so this needs no new machinery. **Re-derive gate 3's stratification + pass bar after the reallocation.**

**Mechanizable acceptance:** a plan/bank check computing per type `n_engaging_pairs = value_pairs × engaging_carriers` and asserting ≥ the registered floor.

---

## BLOCKER 3 (statistics, must-fix) — per-type varied-span LOCUS is unregistered, so prefix-end degeneracy cannot be established from the plan

**Defect.** §4.1 symbolically traces exactly TWO degenerate cells (`query_content`@prefix-end, `persona_role_header`@prefix-end) and refines Holm m on them, but defers the varied-span LOCUS of the other 19 types to the unwritten bank module ("the bank module carries the authoritative full set"). For `language_implied` (body Factor-1 row 10: "the user writes in Spanish / in English; no instruction") the natural single-turn instantiation varies the FINAL user query's language — then A and B share the ENTIRE prefix, `v_pe(A) = v_pe(B)`, the steered patch installs the recipient's own state, and the registered P2 comparison at that cell is ≡ floor noise BY CONSTRUCTION while being narrated as a prefix-end test under prediction 5. `language_implied`@prefix-end is currently COUNTED in P2's m = 15. Same latent risk for any type whose stated span is appended to the final user turn rather than placed in a prior turn or the system region: `user_emotion`, `user_expertise`, `fact_*`.

**Why conclusion-changing.** The α direction is safe (an extra never-significant test only makes Holm stricter). The damage is a GUARANTEED-NULL cell read as "type X is not carried at prefix-end" inside a CONFIRMATORY family.

**Required fix (both parts, cheap — no power parameter changes, so no §9 re-cost):**
- (a) Add a per-type **span-locus column** to the §4.1 bank spec with values `prefix-side` | `final-query` | `generation-header`, REQUIRING every non-pre-declared type's varied span to sit strictly BEFORE the final user turn. Pin `language_implied`'s instantiation explicitly (prior-turns-vary, or all-user-text-varies — either keeps the prefix-end arm live).
- (b) Register a runtime degeneracy guard at P1 capture: assert `v_pe(A) ≠ v_pe(B)` AND `v_ce(A) ≠ v_ce(B)` per pair for every cell not pre-declared degenerate; auto-flag `degenerate_self` otherwise. Family m is adjusted ONLY by the pre-declared constructional drops — never by a runtime discovery (that would be post-hoc).

**Mechanizable acceptance:** a bank-level check that A/B token ids are identical (or not) before the final-user-turn boundary per pair, plus a cosine == 1.0 state-identity assert at capture.

---

## CORRECTNESS FIXES (consistency-checker; fold in — cheap and they prevent a false record shipping to the report)

4. **Delete the phantom parent-deviation claim (independently verified FALSE at source).** §4.2 arm 2, §11, and §12 assumption 12 claim "#2094 installed the raw donor state at replace cells" and frame per-layer norm-matching as a stricter planner refinement. The parent's REALIZED code contradicts this: `scripts/issue2094_run.py` `_donor_payload` (L942–979, main-resident) norm-matches the donor STATE `V_B(donor)` to the recipient's `V_B` norm at replace cells — its docstring states this explicitly and records it as the round-2 code-review resolution of concern `replace-null-donor-realization` (the parent PLAN's Δ-centric wording was the incoherent text that review FIXED). So #2162 MATCHES the parent's realized null regime. The design needs NO change — correct the TEXT (one sentence) and re-ground §12-12's raw-donor-recount contingency, which currently rests on a false premise. Leaving it would ship a false deviation caveat into the report.

5. **Name the fork-base constant flips in §4.6.** `scripts/issue2162_run.py` forks `scripts/issue2094_run.py`, whose module constants pin the PARENT's values for two declared divergences — `MAX_NEW_TOKENS = 1024` and `GRID_TEMPERATURE = 0.0  # greedy grid` (L95–99, verified) — and whose grid has NO per-pair×arm draw seam at all (K exists only for anchors, `ANCHOR_DRAWS = 10`). A verbatim fork silently inherits greedy / 1024 / 1-draw: exactly the silent-pin channel. The plan states the TARGET values unambiguously (§0, §4.3, §11) but never names the three required fork-base changes. Add one line in §4.6 naming them — `GRID_TEMPERATURE → 1.0`, `MAX_NEW_TOKENS → 2048`, new grid K=5 per-pair×arm draw loop — so plan-adherence and code-correctness have a mechanical gate.

6. **Fix the §4.3 pointer nit.** §4.3 says anchor K=10 is "named in § Divergences", but that list carries only items (1)–(7); the anchor-K deviation (from the BODY's ≈5k K=5 arithmetic, not from the parent — the parent's `ANCHOR_DRAWS = 10` matches) is actually named in §11. Either add it to the Divergences list or fix the pointer.

## PLAN-TEXT CLEANUPS (efficiency + statistics, non-blocking; do them while you are in the file)

7. **Finish the truncated sentence in the §9 P3 basis cell** (plan v1.md L300): it currently reads `folded: (21.8 + 6.2 + 3.7 − P2-share 1.6) ≈ wait: V_a+margin for anchors ride P3 blocks too; total P2..P3 GPU-h below`. The arithmetic that actually closes is four lines down; clean up the row text.
8. **Harmonize the judge-call telemetry.** Components 68.3k + 132.4k + 9.4k + 0.4k ≈ 210.5k vs the stated ≈203k total (§9). Telemetry only, no gate rides on it — make them consistent.
9. **Derive worker count from realized GPU count.** The §10 Repro-card launch line hardcodes `--num-workers 8` (L402) while §8/§9 promise re-sharding off realized width on a 4× capacity miss. Specify that the dispatcher derives worker count from `nvidia-smi -L`, not a hand-edited flag.
10. **Prefer a shared block queue over static round-robin.** Recency/load blocks carry longer prefills, so per-block cost is not equal; at ~29 blocks/worker the skew is bounded (~±6 min) but a shared block queue with 8 persistent workers is strictly work-conserving and costs nothing.
11. **Reorder the gate-3 slice to the front of P2** — generate the 6-pairs-per-cell gate slice FIRST and dispatch the SYNC judge while the remaining anchors generate, so the ~15–20 min 8×H100 gate wait overlaps P2's tail instead of idling (~2–3 GPU-h of currently-unbooked idle, silently absorbed by the contingency row). Optionally book P1's 7-idle-GPU stretch and the gate wait as explicit idle rows so Σ(wall × width) reconciles with the GPU-h column per pod.
12. **Strengthen two report-side narration commitments** (add to §6 / §12 as pre-commitments, since the report inherits them):
    - Probe-negative must be narrated as "not linearly decodable at n = 24 / d = 3584 (12 carrier groups)", NEVER the body's stronger "a probe at chance means not encoded" — probe false negatives at this n mislabel the 2×2's *absent* vs *used-but-not-decoded* quadrants. Also report the realized max-selected permutation band's upper bound next to the AUC ceiling per type × slot.
    - The MDE line `1.02/√n` is a SINGLE-test calculation, but the registered "causal-positive" requires the Holm IUT AND disjoint CIs against BOTH nulls — joint power at n = 27 sits below 0.80, so the realized joint MDE is plausibly ~0.23–0.25 rather than 0.20. Pre-commit that the P7 realized-MDE report is computed for (or explicitly annotated against) the full registered conjunction.
    - Break the cross-type null out BY DONOR TYPE (donor ids are already recorded per row) wherever that null looks elevated — a donor type whose content moves the recipient's rubric (e.g. an `instr_format` donor into a `verbosity` recipient) inflates it. Conservative for positives, but it should be visible.
    - Add an explicit `untestable-causal` label (not just prose) for any sub-floor cell in the 2×2 figure, so an underpowered cell is never rendered as "stored-but-unusable".
    - One line next to the pre-exclusion counts noting that the 0.5 separation bar selects on a K=10 anchor estimate, so near-threshold kept pairs carry slightly upward-biased denominators — a small, arm-symmetric, conservative-for-positives attenuation of F.

---

## Constraints on the revision

- **GPU-hour ceiling.** v1 is approved at 54 GPU-h against a 100 GPU-h cap. Blockers 1 and 3 are GPU-neutral by construction. Blocker 2 is a REALLOCATION of carriers, not necessarily an increase — hold the total at or below the current 54 where you can. **If the revised estimate exceeds 100 GPU-h, say so explicitly and prominently: that re-parks the task at `plan_pending` for re-approval rather than proceeding.** Re-state the verbatim `Estimated GPU-hours (total): <number>` line either way.
- **Do NOT re-open settled scope.** The 21-type roster and the scope boundaries were settled interactively with the user on 2026-08-06 and are recorded in the body's Provenance. Fix the null construction, the carrier allocation, and the span-locus registration — do not drop or add types, and do not add a nonlinear readout/map/probe (linear-by-default is a standing rule).
- **Manifest.** Update `artifacts/planned_manifest.json` if and only if the revision changes it, and summarize the diff in your return. Flag explicitly if condition-set MEMBERSHIP changes (that is a resurface trigger). Re-validate against `.claude/skills/issue-v2/planned_manifest.schema.json`.
- **Return contract.** Return the new plan version path, the manifest validation result, the verbatim GPU-hours line, a per-blocker statement of what changed, and any Goal-criterion deviation you were forced into. Do NOT paste plan or manifest bodies into your return.
