# Reconciler memory index — one line per file; open the file when the hook matches the disagreement.

## Pre-registered verdict logic / plan-stage (read the first two on any plan-stage stats/methodology split)

- [Pre-registered gate defects ≠ analyzer-recoverable](feedback_claude_gate_unit_vs_preregistered_verdict_logic.md) — data-recoverability never rescues a defective registered gate/kill/lattice; barred-amendment tell; 35+ incidents. REVISE/FAIL.
- [Gate-design flaw (REVISE) vs recoverable robustness read (APPROVE)](feedback_gate_design_vs_recoverable_robustness_read.md) — REVISE only on affirmative misfire / barred amendment / run-time-only capture loss; else APPROVE. 21 datapoints.
- [Live-replay proposed mechanical checks vs the named offender](feedback_live_replay_proposed_mechanical_checks.md) — run the plan's regex/check/glob against the incident artifact it claims to catch; mutation-walk batteries; PASS-audit leg. #869/#867/#932/#947.
- [Claude APPROVEs unwired lint-check plans](feedback_claude_approves_unwired_lint_check_plans.md) — direct-call-only tests, no_flags wiring unpinned; negative-assertion spans need count(anchor)==1 + a negative test. #963/#979 REVISE.
- [Claude APPROVEs on internal-fn read, missing route filter](feedback_claude_approves_daemon_interface_read_missing_route_filter.md) — verify the CONSUMER-VISIBLE route + grep for the capability claimed absent; kill-fires-on-true-state = REVISE. #956.
- [Plan-verbatim text vs the plan's own binding Must-Fix](feedback_plan_verbatim_text_vs_plan_binding_mustfix.md) — faithful-to-§4 never rescues when MF/canonical text contradicts it; deviation clause bars prose-discretion rescue. #870/#864/#902/#915.
- [Plan "procedure re-writes state Y" safety claims](feedback_claude_accepts_plan_state_rewrite_safety_claim.md) — grep the literal state write + watcher arms (PARK vs pod-safety) + park-then-teardown ORDER. #908/#919.
- [Sign-blind |ρ| rule classed as analyzer concern](feedback_claude_signblind_decision_rule_as_analyzer_concern.md) — |ρ| PASS + directional Confirmed narration = Must-Fix direction pin. #540.
- [Cross-lens defect re-filed per lens](feedback_cross_lens_defect_refiled_per_lens.md) — home lens settled REVISE ⇒ judge the disputed lens on its own question; out-of-scope → APPROVE. #546.
- [Audit acceptance anchored to own instrument](feedback_audit_acceptance_anchored_to_own_instrument.md) — "all N hits of <registered grep>" misses hand-rolled sites; self-referential acceptance → REVISE. #536.
- [Merge-step plans without merge-tree](feedback_claude_approves_merge_step_without_mergetree.md) — run `git merge-tree` yourself; blanket conflict policies wrong for the SHARED conflicting files. #555.
- [Unclosed cherry-pick imports + unimplemented data pin](feedback_claude_plan_cherrypick_closure_and_pin.md) — grep import closure + loaders' `revision=` vs stated pin; "no foreign src/" forbidding the fix → REVISE. #547.
- [Orthogonal partial-state flag in inherited analyzer](feedback_claude_misses_orthogonal_partial_state_flag.md) — enumerate ALL terminal states inherited code emits; same threshold number ≠ same statistic. #546.
- [Plan-named branch with no acceptance test](feedback_claude_underclasses_unverified_branch_test_gap.md) — prescribed `except (A, B)` with one arm pinned: trace the dropped arm; silent-success escape = REVISE. #736.
- [Codex BLOCKER on unsatisfiable plan directive](feedback_codex_unsatisfiable_plan_directive.md) — hash-seed nondeterminism made the gate unmatchable; deterministic-anchor + disclosure correct; no round-2 fix exists. #511.
- [Reroute w/o consumer-pointer trace](feedback_claude_approves_reroute_without_consumer_pointer_trace.md) — grep consumers of canonical ids + env-gate setters; RECONSTRUCTING ids ≠ consuming the return. #564.
- [Codex trim-license on modal-lane control](feedback_codex_approves_trim_license_on_modal_lane_control.md) — "trim if preferred" on the ONLY default-lane control + topology-misleading assumption → REVISE. #578.
- [Codex skips data-construction arithmetic](feedback_codex_skips_data_construction_arithmetic.md) — mix quotas (#543), Goal-cell coverage (#545), "weighable from text" the rig discards (#603) — trace the artifact yourself.
- [Disk assumption framed as a code claim](feedback_claude_trusts_disk_assumption_framed_as_code_claim.md) — "all N have valid body.md" needs a per-folder ls, not a code read; re-run the enumeration. #724.
- [Caller-topology claims without wrapper trace](feedback_claude_trusts_caller_topology_claims_without_wrapper_trace.md) — live-execute the path-selection predicate vs PeftModel/wrappers; fact-checker CONFIRMED ≠ trace. #868.
- [Labeled-split over disjoint registries](feedback_claude_approves_labeled_split_over_disjoint_registries.md) — intersect registry keys with label slugs yourself; 0-hit = unimplementable; trace branch reachability under shipped params. #901.

## Claude code-review misses (FAIL-leaning calibration)

- [Global-resolution fold over a per-behavior ladder](feedback_claude_passes_global_resolution_over_per_behavior_ladder.md) — single slug/flag collapse vs plan's per-unit ladder; committed D0 evidence dictates the mixed topology; coverage-incomplete→unit-FAIL is the silent kill-verdict accomplice. #1739 leg-2 r1 FAIL.
- [Persist-before-reduce: ordering not existence](feedback_persist_before_reduce_ordering_vs_existence.md) — file exists ≠ persisted before the remote judge; derived expected-set ≠ hollow gate; prior-round CONCERN severity binds. #906 r9.
- [Headline decision statistic not produced](feedback_claude_misses_headline_decision_statistic_not_produced.md) — grep the named estimator's call site; PASSing other items never covers it (#841 r1/r2 FAIL). COUNTER #922 r2: a locally-equivalent-by-linearity helper is PASS — verify equivalence first.
- [Under-classed silent failures](feedback_claude_underclasses_silent_failures.md) — real silent-failure bug + CONCERNS/Minor → FAIL; classify by what the bug DOES, not fix size; incl. #1098 guard-waiver fail-open. 13-incident ledger.
- [Plan-complete diff violating the lane's declare→satisfy artifact contract](feedback_claude_plan_complete_diff_violates_lane_artifact_contract.md) — new execution path: trace WHO WRITES the declared sentinel; finalize exit-3 strands billing pod. #909 FAIL.
- [Same-file siblings (sibling-scan family)](feedback_claude_misses_same_file_siblings.md) — must-fix walks miss the bug CLASS: sibling paths, render branches, resamplers, figure-vs-analyze layers (now Step 3.7).
- [Producer/consumer contract mismatches](feedback_claude_misses_producer_consumer_key_mismatch.md) — round-trip literal contracts: JSON key path-vs-inline, custom_id f-strings, consumer regex, builder default vs plan holdout.
- [Verify inherited producer capture semantics](feedback_verify_inherited_capture_semantics_before_crediting_slot_claim.md) — read the PRODUCER's write code, not plan prose/consumer docstring; #658 span convention broke #812's slots. #812 r1.
- [Dispatcher-wiring correctness bugs](feedback_claude_misses_dispatcher_wire_bugs.md) — optional args never passed, fallbacks never invoked, defaults neutering plan elements on the canonical invocation. #504/#517/#520.
- [Scaffolded-but-not-plumbed pipelines](feedback_claude_scaffolded_pipeline_not_plumbed.md) — orphaned helpers, readers with no writer, HALT gates never invoked; grep the production wiring site. #397/#508/#516.
- [Cross-branch Python module dep](feedback_claude_misses_cross_branch_python_module_dep.md) — lazy imports of sibling-branch-only modules pass import-checks, die on the pod; ls the worktree. #501.
- [Fix regressions](feedback_claude_misses_fix_regressions.md) — a REPLACING check can be weaker on the original Must-Fix class; replicate the old bad input through the NEW check. #389/#554.
- [Floor-vs-raise divergence from reference](feedback_claude_misses_floor_vs_raise_divergence.md) — new script floors where the cited reference RAISES; the rationalizing comment is the smell. #532.
- [In-code-comment invariance claims](feedback_claude_misses_invariant_comment_smell.md) — comment asserts an invariant the expression doesn't deliver; trace it for the covered case. #505 r2.
- [Predictor formula mismatch as Nit](feedback_claude_treats_predictor_formula_mismatch_as_nit.md) — same field name, different formula corrupts the by-name cross-arm aggregator; "unchanged this round" ≠ scope argument. #518.
- [Plan-vs-parent semantic refinement](feedback_claude_inherits_parent_with_plan_semantic_refinement.md) — plan refines a parent construct; impl carries parent code verbatim; data can't measure the target. #506 r3.
- [Estimand divergence excused via plan pseudocode](feedback_claude_excuses_estimand_divergence_via_plan_pseudocode.md) — named estimand governs over a shortcut exact only off the primary cohort → FAIL. #539.
- [CONCERNS on headline-poisoning bug pre-pod-launch](feedback_claude_concerns_on_pre_pod_launch_headline_bug.md) — "fix next round before launch" is unenforced; wrong headline estimator on launch path → FAIL. #522.
- [CONCERNS on missing tiny-N smoke](feedback_claude_concerns_on_smoke_gate.md) — `--help`/import/dry-run-only evidence is the canonical substantive FAIL; "Wave X IS the smoke" needs Wave X run. #492/#551.
- [DRY_RUN smoke misses CUDA init](feedback_claude_dry_run_smoke_misses_cuda_init.md) — dry-runs never exercise CUDA init; unconditional CVD assignment piles shards on GPU 0. #488 r3.
- [Synthetic-fixture smoke masks args-grid bug](feedback_claude_synthetic_fixture_smoke_masks_args_grid_bug.md) — validator iterates module constants, --smoke sets another grid; demand artifact-chained smokes. #517 r2.
- [Overcorrected critic concern into blocker](feedback_claude_overcorrects_critic_concern_to_blocker.md) — guardrail over-rejects the plan-envisioned PASS case and the test codifies it; re-read the plan §s. #514 r3.
- [Green tests over verifier semantics](feedback_claude_trusts_green_tests_over_verifier_semantics.md) — artifact IS a verifier: read check bodies vs prescribed scope; stub fixtures pin nothing. #454/#608/#564/#565.
- [Best-effort upload made load-bearing](feedback_claude_misses_besteffort_upload_made_loadbearing.md) — new HF-fetch consumer of a warn-only upload; teardown loses the never-uploaded artifact → FAIL. #613.
- [Fabricated plan-adherence checkmarks](feedback_claude_fabricates_rf_walkdown_checkmark.md) — ✓ with plausible justification a grep disproves; rg the literal new AND prior values before believing any ✓.
- [Nonexistent backstop semantics](feedback_claude_cites_nonexistent_backstop_semantics.md) — read the named gate's ACTUAL fail condition before crediting a downgrade. #594.
- [Comment-tail waiver spoofs on raw-scan guards](feedback_claude_misses_comment_tail_spoof_on_rawscan_guards.md) — replay `<destructive> # <waiver>` shapes yourself; in-round fail-open ≠ documented fail-closed trade-off. #897 FAIL.

## Codex code-review overreach (PASS-leaning calibration)

- [Hardening beyond minimal-port contract](feedback_codex_hardening_beyond_minimal_port_contract.md) — registered change set + writer-reachability + execution-test the demanded fix; persist residue via raise+defer-concern; grep quoted "acceptance criteria" VERBATIM (#952/#958 invented them). 21 variants.
- [Step 0.6 literal vs purpose](feedback_codex_step_06_literal_vs_purpose.md) — PASS when a demonstrated pre-launch gate runs the changed code for real; FAIL when the first real run would be production. #551/#560.
- [Marker format/existence conflated with code](feedback_codex_conflates_marker_format_with_code.md) — marker-shape nits / stale-file reads with "diff not reviewed" → discard, verify Claude independently, PASS.
- [Pre-existing / stale state litigated in round N](feedback_codex_litigates_pre_existing_in_round_n.md) — git-provenance family: trunk pre-existence, scope drift, stale-worktree "deletions", stale UNTRACKED copies (#922: `git show <pin>:` is authoritative).
- [Over-read plan prose](feedback_codex_overreads_plan_prose.md) — synthesized quotes, 1-D companions inflated to grids, contextual sentences read unconditional, prose names vs pinned Source impls.
- [Methodology choice as code bug](feedback_codex_methodology_choice_as_bug.md) — implementer picked plan-listed option (b); or the flag targets the plan's own registered rule; grep the plan's exact wording. #480/#543.
- [Registered noise-floor statistic flagged as unit mismatch](feedback_codex_flags_registered_noise_floor_statistic.md) — read the PLAN's registered SOURCE fn + parent usage; faithful re-impl = out-of-scope, PASS. #661 r2.
- [Fabricated code citation for a silent-shrink BLOCKER](feedback_codex_fabricated_code_citation_silent_shrink.md) — quoted file:line that greps 0 hits; real path runs fail-loud validators first → PASS. #667.
- [Fail-loud diagnostic blocker](feedback_codex_fail_loud_diagnostic_blocker.md) — side-channel failure LOUD + headline artifact preserved = PASS-class with mandatory standing recs.
- [Meta-test blocker on verified fix](feedback_codex_meta_test_blocker_on_verified_fix.md) — negative-control-missing on a regression test whose shared code path catches reverts → CONCERN, never a cap-3 pivot. #505 r3.
- [Gitignored worktree artifacts as production state](feedback_codex_gitignored_artifacts_as_production_state.md) — reachability walk: git propagation, canonical-flow creation, pre-existence. #543/#570/#601.
- [Blocker on sandbox-unreadable pod artifact](feedback_codex_blocker_on_sandbox_unreadable_pod_artifact.md) — "could not read" ≠ unverifiable: producer exit-assert + pinned revision + parent proof → demote. #534.
- [Env-var orphan unreachable](feedback_codex_env_var_orphan_unreachable.md) — trace the import chain to the actual entry point; zero consumers = dead orphan, PASS + remove rec. #488 r2.
- [Raw-branch-diff misses Step 10d surgical merge](feedback_codex_raw_branch_diff_misses_surgical_merge.md) — outside-scope "reverts" bounded by the additive checkout; PASS + surgical-merge rec. #511.
- [FAIL on absent OTHER-plan-section wiring](feedback_codex_plan_section_in_scope.md) — verify the element is INSIDE the round brief + reachable; un-invoked plan gaps are PASS+CONCERNS. #505 r6.
- [Chat-template blocker without measurement-rig cross-check](feedback_codex_chat_template_blocker_without_measurement_xcheck.md) — parent rig used the SAME surface; saturation in the parent CSV is the fingerprint; inheriting it is correct. #509 r3.

## Clean-result-critic + interp-critic calibration

- [Claude clean-result-critic under-applies SPEC text](feedback_claude_clean_result_critic_underapplies_spec_text.md) — mechanical pre-pass PASS ≠ spec compliance; 25 BLOCKING spec-text rules + 22 Codex over-fire DISCARDs. #923 r1+r2, #833 r3.
- [Bare-#N refs + missing reuse path (Lens 2/5)](feedback_clean_result_bare_issue_refs_and_reuse_path.md) — grounded FAILs Claude PASSes; discard the L4/L6/L12/L15 Codex over-fires as SPEC misreads. #722 r1.
- [Lens 7 statistical framing missed](feedback_claude_misses_lens7_statistical_framing.md) — re-scan for named tests, ±, derived intervals in prose/captions before trusting a Claude PASS. #378.
- [Cross-loop CI conflict (interp vs clean-result)](feedback_cross_loop_ci_conflict.md) — interp-required CIs / house-style captions survive Codex's Lens-7 FAIL; add "Do NOT remove" lines. #478/#509/#464.
- [Lens 9 raw+processed exception](feedback_codex_misses_lens9_raw_processed_exception.md) — adjacent raw + processed pair counts as ONE figure. #480 r1.
- [Grandfathered re-gate prose re-litigated](feedback_codex_relitigates_grandfathered_regate_prose.md) — byte-unchanged previously-adjudicated findings are out of scope; diff the set-body commit. #464.
- [Plain-English REPLACEMENT re-litigated after keyword ban](feedback_codex_relitigates_replacement_register_after_keyword_ban.md) — ban lists are ENUMERATED keywords, not concepts; never ban the phrasing a prior reconcile prescribed. #715 r2 PASS.
- [Silent plan deviations missed](feedback_claude_misses_silent_plan_deviations.md) — implementer-flagged deviations absent from the body; diff plan vs body row-by-row (Lens 5/13). #520.
- [Body factual errors misframed as confidence risk](feedback_claude_interp_misframes_as_confidence_risk.md) — confidence covers UN-addressed risk, never a body asserting the OPPOSITE of the flagged alternative. #492.
- [Interp completeness vs honesty-only bar](feedback_interp_completeness_vs_honesty_bar.md) — Codex PASSes on honesty alone; verified sibling-artifact misses bearing on the headline → REVISE. #549.
- [Codex passes when sandbox blocks data](feedback_codex_passes_when_sandbox_blocks_data.md) — "could not fetch" disclaimers zero-weight the PASS; re-verify vs the JSONs; ALWAYS Read pinned PNGs. #381/#601.
- [Recount with silent matcher variant](feedback_codex_recount_with_silent_normalization.md) — recompute under the PLAINEST rule; sibling numbers cohering under ONE matcher vindicate the body. #833 r2.
- [Fix WORDING verified, not artifact TRUTH](feedback_codex_verifies_fix_wording_not_artifact_truth.md) — re-derive fix-round disclosure claims from the pinned JSONs/figures yourself. #778 r2 REVISE.
- [Correct numeric claim failed via wrong statistic / JSON-only search](feedback_codex_fails_correct_numeric_claim_wrong_statistic_or_artifact.md) — multi-number simultaneous match pins the statistic; search the .pt artifacts the footer names. #920 r2 PASS.
