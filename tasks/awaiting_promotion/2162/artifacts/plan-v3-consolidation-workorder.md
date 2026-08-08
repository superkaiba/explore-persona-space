# Issue #2162 — plan v2 → v3 consolidation work order (round-2 non-blocking items)

**Round 2 outcome: ALL LENSES CLEARED.** `statistics-critic` PASS · `methodology-baselines-critic` PASS · `efficiency-critic` APPROVE · `consistency-checker` WARN (0 BLOCK). Codex twins were confirmed no-shows both rounds (org quota exhausted until 2026-09-05), so each verdict is single-Claude and binding.

**This is NOT a blocker round.** Every critic explicitly stated its remaining items are non-blocking and that none warrants a revision round on its own. You are landing them together because several are report-side PRE-COMMITMENTS that the downstream `report-verifier` grades the report against, and one is a body-fidelity error on USER-SETTLED content. Do NOT re-open any design decision, re-derive any statistic, or re-litigate anything either round PASSed. Make these edits and nothing else.

Plan to edit: `tasks/approved/2162/plans/plan.md` (symlink → `v2.md`). Body (authority for item 1): `tasks/approved/2162/body.md`. Manifest: `tasks/approved/2162/artifacts/planned_manifest.json`.

**Hard invariants — do not disturb:**
- `Estimated GPU-hours (total): 54` must remain verbatim, both occurrences. NONE of these edits changes compute.
- Holm family m stays `31/15/28`. No edit here touches a registered test count.
- Manifest condition-set MEMBERSHIP stays exactly the 13 current conditions. Metrics stay 8. Figures stay 15 unless item 4 genuinely needs a view added — if so, say so explicitly in your return.
- `Goal-criterion deviations: NONE` (§12) stays true.
- The value CYCLE, the per-class carrier allocation, and the span-locus registry are all PASSed constructions. Do not restructure them.

---

## Item 1 (body fidelity — flagged independently by BOTH consistency-checker and methodology-baselines-critic) — `fact_user_name` drops a user-settled value

**Verified at source by the orchestrator.** Body row 12 (`body.md` L108) specifies three values: `"My name is Alice." / "…Bob." / "…Priya."`. Plan v2 §4.1 row 12 (`v2.md` L106) reads `Alice / Priya / Marcus`. `Bob` appears ZERO times in the plan; `Marcus` appears ZERO times in the body. This is the ONE type where the body already supplied a complete 3-value set, and it is the one type where a value was swapped out — while §12-21 simultaneously asserts the body's exemplars are kept verbatim.

**FIX: revert to the body's own set — `Alice / Bob / Priya`.** Do not keep Marcus and flag it; the body settled this roster and there is no gap to fill. Preserve the cycle direction convention and the row's class/locus columns. If `Bob` was swapped for a token-length reason, that reason does not survive contact with the body being authoritative — revert regardless, and if you believe length matching genuinely matters, note it as an observation in §12 without changing the value.

## Item 2 (provenance honesty) — the value-slot labeling claim is inaccurate for ≥3 rows

§4.1 (L82) and §12-21 claim "body Factor-1 exemplars kept verbatim as v1/v2; each v3 a flagged planner addition". Two lenses found this is not true row-wise: for `refusal_boundary` and `verbosity` the planner's ADDITION sits at **v2** with the body's second exemplar at **v3**. Mechanically harmless (the cycle realizes all unordered pairs, so every body-named contrast survives as a directed pair), but §12-21 points the user's approval attention at the v3 column only, so the flag as written hides where the additions actually are.

**FIX:** correct the claim to state the actual addition SLOT per type (a short parenthetical per affected row in the §4.1 table, or a one-line correction in §12-21 naming the rows whose addition is not at v3). After item 1, re-check every row: the claim must be true as written for all 21 types.

## Item 3 (risk register) — add the `constraint_knowledge` weak-separation row to §8

`constraint_knowledge`'s (no-internet ↔ KB-only) directed pair is a likely weak-separation class: on live-lookup carriers BOTH values decline, so ceiling/floor separation rides on whether the answer references the knowledge base. This is the same class as the `refusal_boundary` (none ↔ disclaimer) weakness the plan ALREADY names in §8 — it was simply missed.

**FIX:** one §8 risk row, same shape as the existing `refusal_boundary` row, with the same mitigation (the pre-registered separation exclusion + `untestable-causal` labeling handle it gracefully — a thin cell is labeled underpowered, never rendered as a false verdict). Purpose is that a thin cell at P7 is expected rather than a surprise.

## Item 4 (pre-commitment extension) — extend the by-donor breakout to the SHUFFLED-donor null for the two ordinal value sets

§6 report-side pre-commitment 3 currently wires the by-donor-type breakout to the CROSS-TYPE null only (manifest figure `crosstype_null_by_donor`). For the two graded/ordinal value sets — `refusal_boundary` (none / disclaimer+refer / decline+refer) and `constraint_knowledge` (no-internet / live-web / KB-only) — the SHUFFLED-donor null can carry partial rubric credit, because an adjacent-value answer scores mid on the target rubric. That ELEVATES the null and depresses separation (conservative for positives, so not a validity threat), but it is currently invisible.

**FIX:** extend pre-commitment 3 so the shuffled-donor null is ALSO broken out by donor VALUE for these two types specifically, to be reported if either null looks elevated at P7. Donor pair + value ids are already recorded per row, so no new data capture is needed. If this needs a manifest figure view, add it and say so in your return; if the existing `crosstype_null_by_donor` transform can carry a shuffled-donor facet, prefer that.

## Item 5 (scope caveat) — name the translation dilution per type, and the translationese caveat

For `language_implied` and `instr_language`, the realized carrier text is a base-model TRANSLATION of the WildChat rows, not the raw tier-1/2 rows — so F for the language types is measured on model-translated text rather than native user queries. This is already declared in three places (§4.1 table row 10, the §4.1 tier note, §12-22) and was adjudicated JUSTIFIED (forced by the minimal-pair design: content-matched cross-language pairs cannot be sourced from any real corpus; within-pair floor/ceiling normalization bounds the confound; the langid companion validates at runtime).

**FIX (wording only, two parts):** (a) one clause in the tier declaration naming the PER-TYPE dilution explicitly, so the report caveat inherits it rather than depending on a reader joining three separate declarations; (b) a §6 report-side pre-commitment that the language-type reads carry the translationese scope caveat.

## Item 6 (new caveat created by the B2 fix) — cross-CLASS F comparability

The per-class carrier allocation means prediction 2's policy-vs-item contrast now compares types measured on DIFFERENT carrier regimes: policy types on 9 neutral WildChat carriers, item types on 12 hand-written engaging ones. Per-type verdicts remain within-cell and clean, and the confound runs in the CONSERVATIVE direction for prediction 2 (item types get their best-case carriers, so a null there is the strongest available form of the claim). But the existing §4.1 realism caveat covers DATA TIER only, not cross-class comparability.

**FIX:** extend the caveat by one clause to cover cross-CLASS F comparability, and add the matching §6 report-side pre-commitment so the report states it wherever the policy-vs-item contrast is drawn.

## Item 7 (internal consistency) — harmonize the gate-3 sync duration

§9 (L366) quotes the gate-3 sync judge slice at "≈ 2 min at 3 keys × 2 procs"; §7 (L295) says "≈ 15 min". No gate rides on either figure and both sit far inside P2's ~72-min wall, so the overlap conclusion is insensitive — but the two lines should not disagree.

**FIX:** pick the defensible figure and state it consistently in both places. (The ~2 min is the throughput-table dispatch estimate for ~9.1k calls; the ~15 min presumably includes generation of the slice + judge round-trip + verdict. If they measure different spans, LABEL each span rather than forcing one number.)

## Item 8 (report-side narration) — read null levels against the `query_content` control

Both null arms install carrier-mismatched states for class-E cells (same-type null is carrier-shuffled; cross-type falls back to seeded draws) while the steered arm is carrier-matched by construction. The rubric-keyed DV makes this second-order — a wrong-carrier donor expresses the wrong VALUE either way, which is what the rubric scores — and this was structurally present in v1 too (~92% of v1's derangement draws were carrier-mismatched by chance), so it is NOT a revision-introduced defect and needs no design change. `query_content` is the built-in control that quantifies the pure query-swap effect.

**FIX:** one §6 report-side pre-commitment that null levels are read AGAINST `query_content`'s F when narrating specificity, so a carrier-mismatch contribution to null level is visible rather than assumed away.

---

## Not for you — recorded here so it is not lost

The `efficiency-critic` flagged one IMPLEMENTATION-panel item that is deliberately NOT a plan edit: the shared block queue's claim-file mechanism (§4.6, `O_CREAT`-exclusive claims under `/workspace/issue2162_out/claims/`) needs stale-claim reclamation on the resume path — a claim file with no matching done-checkpoint must be reclaimable (claim-age or pid-liveness keyed), or a crashed worker leaves its in-flight block claimed forever and the grid completes SHORT until the `grid_done.json` manifest check catches it. The orchestrator carries this into the Step 4 implementation brief. You may add a one-line requirement to §4.6 if it belongs in the spec, but do not design the mechanism.

## Return contract

Return: the new plan version path; the manifest validation result + whether figures/conditions/metrics counts changed; the verbatim `Estimated GPU-hours (total): <number>` line (must still be 54); a per-item one-line statement of what you changed; and confirmation that Holm m is still 31/15/28, condition membership is unchanged, and `Goal-criterion deviations: NONE` still holds. Persist via `uv run python scripts/task.py new-plan-version 2162 --file <path>`. Do NOT paste plan or manifest bodies into your return.
