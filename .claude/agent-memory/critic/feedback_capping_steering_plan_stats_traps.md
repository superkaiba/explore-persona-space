---
name: capping-steering-plan-stats-traps
description: "Traps in activation-capping/steering defense plans (#2203): footprint-matched random null per cap arm; incoherent-row listwise drop = post-treatment selection; harm-judge waves need rule-28 api-refusal registration; norm-matching is INERT for replace ops (edit magnitude unmatched); CI-overlap lattice adjudication vs the registered paired contrast"
metadata:
  type: feedback
---

Recurring findings on inference-time capping/steering defense plans
(first seen #2203, assistant-axis capping reproduction of arXiv
2601.10387).

1. **Footprint-matched random-direction null.** A norm-matched random cap
   with τ re-computed for v_rand is necessary but NOT sufficient: the null
   must also match each gated arm's intervention FOOTPRINT (position set ×
   decode steps). A ctx-position-only `cap_ctx_randnull` gating an
   all-token cap arm understates generic-disruption effects by orders of
   magnitude of edited positions → false "clears the null band" on the
   all-token arm. Since τ = 25th pct of the direction's own projections,
   the clamped-state fraction matches by construction, so the added arm is
   cheap and exactly matched.
2. **Incoherent-row listwise drop = post-treatment selection.** Dropping
   rows flagged incoherent from the PAIRED contrast conditions the eval
   set on the heaviest arm's coherence — removes exactly the prompts where
   the cap bit hardest, underestimates the all-token reduction, and biases
   TOWARD "single-position recovers most of the effect". Fix: judge +
   persist ALL generated rows with per-row coherence flags; all-rows read
   primary (or dual-registered), coherence-dropped read as labeled
   sensitivity. Judge parse-fails stay legitimate drops (rule 9) — do not
   bundle them with model-side incoherence in one listwise clause.
3. **Harm-content judge waves: register the rule-28 api-refusal class.**
   Any Batch-API wave judging harmfulness of jailbreak/adversarial
   completions is the #1739 censoring class (34.1% refusal draws,
   outcome-correlated: highest-harm rows censored first → harm rate biased
   DOWN on high-harm arms). The rule-26 pilot gate does NOT key on this
   class by design. A plan naming only drop-never-coerce + transport-retry
   + pilot gate has NOT covered it — require per-arm `n_api_refusal`
   reporting + the targeted SYNC re-issue remediation
   (`scripts/issue1739_evilood_refusal_rejudge.py` reference).

4. **Norm-matching is INERT for axis-REPLACE ops — edit magnitude is the
   unmatched quantity (Methodology lens, #2203 v8).** `axis_replace`
   applies `v̂·(⟨h_def,v̂⟩ − ⟨h,v̂⟩)` using only the UNIT direction, so a
   "norm-matched random direction" control matches nothing the op reads.
   Along a random unit v̂ in high d the per-position edit concentrates
   near `‖h_def−h‖/√d`, while the real axis captures a large share of the
   displacement — the random-replace control is magnitude-unmatched
   SMALLER by construction. Asymmetric consequence: only the
   axis-specific-CONFIRMED read is at risk ("any equally-large coherent
   perturbation would work" not excluded); Falsified is immune a
   fortiori. Analyzer-weighable (not REVISE) iff per-position
   |Δproj| = |proj_after − proj_before| telemetry is persisted for both
   arms — demand the A-vs-R |Δproj| distributions beside any Confirmed
   read. Single-seed v_rand (n=1 direction draw) is the same family:
   Concern + wording fix ("one seeded draw"), never a REVISE when it is
   deliberate parent parity.

5. **CI-overlap lattice adjudication vs the registered paired contrast
   (Statistics lens REVISE, #2203 v8).** A registered verdict lattice that
   reads Confirmed ⇔ "arm A's reduction CI entirely above control R's
   reduction CI" and Falsified ⇔ "CIs overlap" — while the SAME §3
   registers a per-prompt paired A-vs-R contrast — is fundamentally
   miscalibrated: disjointness ≈ demanding p≲0.005, overlap grants
   Falsified at p>~0.005, both reduction CIs are positively correlated
   through the SHARED baseline sample, and the pairing that cancels the
   baseline is thrown away. Run the arithmetic at the plan's own committed
   effect size: in v8, Confirmed needed a reduction gap > hw_A+hw_R ≈
   0.044 under realistic discordance while the whole committed effect was
   0.036 — the H1-expected axis-specific world registers FALSIFIED, while
   the paired per-prompt difference CI at the same numbers is [0.011,
   0.061] (cleanly axis-specific). Fix is a one-clause rewrite (adjudicate
   every CI clause on the paired difference D = A-reduction − R-reduction,
   cluster-bootstrapped on already-persisted rows; zero re-cost). Verify
   the per-item join inputs actually exist in the committed comparator
   JSONs before crediting the pairing (v8: `harm.mean_scores` keyed
   `<arm>-jb-<idx>` + `cluster_ids` + baseline judge JSON — present; the
   arm-name key prefix must be stripped at join time).

**Why:** items 1-3 survived an otherwise strong v4 plan that got τ
re-computation, held-out band freeze (selection symmetry fix 2), pilot
gate, and per-prompt score persistence right; item 4 is the replace-op
sibling the v8 amendment surfaced (v8 handled 1-3 correctly); item 5 is
the v8 Statistics-lens REVISE — the lattice named two per-arm CIs where
its own registered contrast was the paired difference.

**How to apply:** any plan with cap/steer arms vs a random-direction null,
a coherence gate feeding a paired analysis, or a judged harm DV over
adversarial completions. See also [[selection-symmetric-nulls]] lineage
entries.
